"""Reasoning navigation agent — the only module that talks to Claude.

Given the user's message, a screenshot of the current viewer frame, and a small
camera description, ask Claude (vision + structured output) to return an
``answer`` plus a list of camera/highlight ``actions`` the frontend executes.

Kept deliberately small and provider-isolated so the LLM backend is swappable.
Model + key come from ``config`` (``backend/.env``).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .. import config

try:  # anthropic is optional at import time so tests/CI without it still load
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore


def _make_client():
    if anthropic is None or not config.ANTHROPIC_API_KEY:
        return None
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


_client = _make_client()


SYSTEM_PROMPT = """You are a navigation + reasoning agent embedded in a real-time 3D \
Gaussian-splat viewer. The user sees a live rendering of a reconstructed 3D scene and \
talks to you about it. You are given the CURRENT camera view as an image plus a short \
description of the camera. You can move the camera and highlight objects by returning a \
list of ACTIONS, and you answer the user's question in `answer`.

Coordinate conventions:
- Any action that refers to an object in the scene MUST include `target_2d`: the [x, y] \
location of that object's CENTER in the CURRENT image, normalized to [0,1] with (0,0) = \
top-left and (1,1) = bottom-right. Estimate it from the image.
- `amount` is RELATIVE and unitless, normally 0.5-2.0; 1.0 means one moderate step / a \
~30 degree turn / a comfortable zoom. The client scales it to the scene automatically.

Action types (return only the ones you need):
- move:    {"type":"move","direction": forward|backward|left|right|up|down, "amount":n}
- rotate:  {"type":"rotate","direction": clockwise|counterclockwise, "amount":n}
- zoom:    {"type":"zoom","direction": in|out, "amount":n}
- fly_to:  {"type":"fly_to","target_2d":[x,y]}      (move toward + frame an object)
- look_at: {"type":"look_at","target_2d":[x,y]}     (turn to face an object)
- highlight:{"type":"highlight","target_2d":[x,y],"label":"short name"}
- clear_highlight: {"type":"clear_highlight"}
- reset_view: {"type":"reset_view"}
- set_splat_scale: {"type":"set_splat_scale","amount":0.5}  (shrink splats) \
  or {"type":"set_splat_scale","amount":2.0}  (enlarge splats)
- set_background: {"type":"set_background","label":"black"}  (color name: \
  black, white, navy, midnight, gray, slate, dark)
- set_brightness: {"type":"set_brightness","amount":1.5}  (1.0=normal, >1=brighter)

Rules:
- "go to / move to / take me to X" -> if X is visible, emit `fly_to` with its target_2d. \
If X is NOT visible in the current image, do NOT guess coordinates: emit a `rotate` or \
`move` to explore and say in `answer` that you are looking for it.
- "what is this / what does X do / explain X / find X" -> reason from the image, emit a \
`highlight` on it (with target_2d), and put a concise, specific explanation in `answer`. \
If you are unsure what an object is or does, say so plainly rather than inventing detail.
- `answer` is ALWAYS required: one or two friendly sentences describing what you did and/or \
the answer to the question. Never leave it empty.
- If the user asks to change the appearance of the world (bigger/smaller splats, different \
background, brighter/darker), emit the corresponding modification action."""


# Structured-output schema. No min/max constraints (validated client-side).
ACTIONS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "actions"],
    "properties": {
        "answer": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "move",
                            "rotate",
                            "zoom",
                            "fly_to",
                            "look_at",
                            "highlight",
                            "clear_highlight",
                            "reset_view",
                            "set_splat_scale",
                            "set_background",
                            "set_brightness",
                        ],
                    },
                    "direction": {
                        "type": "string",
                        "enum": [
                            "forward",
                            "backward",
                            "left",
                            "right",
                            "up",
                            "down",
                            "clockwise",
                            "counterclockwise",
                            "in",
                            "out",
                        ],
                    },
                    "amount": {"type": "number"},
                    "target_2d": {"type": "array", "items": {"type": "number"}},
                    "label": {"type": "string"},
                },
            },
        },
    },
}


def _history_to_messages(history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert prior {role, text} turns to Claude messages (text only, last ~6)."""
    out: List[Dict[str, Any]] = []
    for turn in history[-6:]:
        role = turn.get("role")
        text = (turn.get("text") or "").strip()
        if role in ("user", "assistant") and text:
            out.append({"role": role, "content": text})
    return out


def run_agent(
    message: str,
    screenshot_b64: Optional[str],
    camera: Dict[str, Any],
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Return {"answer": str, "actions": [..]}. Never raises for normal errors."""
    if _client is None:
        return {
            "answer": (
                "The navigation agent isn't configured yet — set ANTHROPIC_API_KEY "
                "in backend/.env and restart the server."
            ),
            "actions": [],
        }

    user_content: List[Dict[str, Any]] = []
    if screenshot_b64:
        user_content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            }
        )
    cam_note = (
        f"Camera mode={camera.get('mode')} fov_rad={camera.get('fov')}. "
        "The scene is roughly centered at the origin, Y is up."
    )
    user_content.append({"type": "text", "text": f"{cam_note}\n\nUser: {message}"})

    messages = _history_to_messages(history) + [
        {"role": "user", "content": user_content}
    ]

    try:
        resp = _client.messages.create(
            model=config.AGENT_MODEL,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            output_config={
                "format": {"type": "json_schema", "schema": ACTIONS_SCHEMA}
            },
            messages=messages,
        )
    except Exception as exc:  # network / API errors -> friendly fallback
        return {
            "answer": f"Sorry, the agent hit an error talking to the model: {exc}",
            "actions": [],
        }

    if getattr(resp, "stop_reason", None) == "refusal":
        return {"answer": "I can't help with that request.", "actions": []}

    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text = block.text
            break
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {
            "answer": "I had trouble forming a response. Could you rephrase that?",
            "actions": [],
        }

    if not isinstance(data, dict):
        data = {}
    actions = data.get("actions")
    if not isinstance(actions, list):
        actions = []
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = "Done."
    return {"answer": answer, "actions": actions}
