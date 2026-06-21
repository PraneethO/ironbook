"""Reasoning navigation agent — the only module that talks to Claude.

Given the user's message, a screenshot of the current viewer frame, and a small
camera description, ask Claude (vision + structured output) to return an
``answer`` plus a list of camera/highlight ``actions`` the frontend executes.

Kept deliberately small and provider-isolated so the LLM backend is swappable.
Model + key come from ``config`` (``backend/.env``).
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import sentry_sdk

from .. import config

try:  # anthropic is optional at import time so tests/CI without it still load
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

# Langfuse is optional — degrades gracefully to no-ops when not configured.
try:
    from langfuse import get_client as _lf_get_client
    from langfuse import observe as _lf_observe
    _langfuse_ok = True
except Exception:  # pragma: no cover
    _langfuse_ok = False
    _lf_get_client = lambda: None  # type: ignore[assignment]

    def _lf_observe(func: Any = None, **kw: Any) -> Any:  # type: ignore[misc]
        def dec(f: Any) -> Any:
            return f
        return dec if func is None else func


def _make_client():
    if anthropic is None or not config.ANTHROPIC_API_KEY:
        return None
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _make_async_client():
    if anthropic is None or not config.ANTHROPIC_API_KEY:
        return None
    return anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)


_client = _make_client()
_async_client = _make_async_client()


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
background, brighter/darker), emit the corresponding modification action.
- When you emit a `fly_to`, `look_at`, or `highlight` action for a specific object, set \
`diagram` to a compact ASCII/Unicode schematic (≤10 lines, ≤24 chars wide) that \
illustrates the object's structure using box-drawing characters (─ │ ┌ ┐ └ ┘ ╔ ╗ ╚ ╝ ═ ║). \
Label key sub-parts inline. Keep it minimal and readable. \
For any response where no object is being focused, set `diagram` to ""."""


# Structured-output schema. No min/max constraints (validated client-side).
ACTIONS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "diagram", "actions"],
    "properties": {
        "answer": {"type": "string"},
        "diagram": {"type": "string"},
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


def _build_user_content(
    screenshot_b64: Optional[str],
    message: str,
    camera: Dict[str, Any],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if screenshot_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": screenshot_b64,
            },
        })
    cam_note = (
        f"Camera mode={camera.get('mode')} fov_rad={camera.get('fov')}. "
        "The scene is roughly centered at the origin, Y is up."
    )
    content.append({"type": "text", "text": f"{cam_note}\n\nUser: {message}"})
    return content


def _extract_partial_answer(partial_json: str) -> str:
    """Pull the answer string out of an incomplete JSON as it streams in."""
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', partial_json)
    if not m:
        return ""
    s = m.group(1)
    s = s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\").replace("\\t", "\t")
    return s


@_lf_observe(name="agent-act")
def run_agent(
    message: str,
    screenshot_b64: Optional[str],
    camera: Dict[str, Any],
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Return {"answer": str, "actions": [..]}. Never raises for normal errors."""
    sentry_sdk.set_tag("agent.model", config.AGENT_MODEL)
    sentry_sdk.add_breadcrumb(
        category="agent",
        message="agent pipeline started",
        data={
            "message_preview": message[:200],
            "has_screenshot": bool(screenshot_b64),
            "camera_mode": camera.get("mode"),
            "history_turns": len(history),
        },
        level="info",
    )

    _lf = _lf_get_client()
    if _lf:
        _lf.update_current_trace(
            tags=["navigation", "vision-grounded"],
            metadata={"model": config.AGENT_MODEL},
            input={
                "message": message,
                "has_screenshot": bool(screenshot_b64),
                "camera_mode": camera.get("mode"),
                "history_turns": len(history),
            },
        )

    if _client is None:
        result: Dict[str, Any] = {
            "answer": (
                "The navigation agent isn't configured yet — set ANTHROPIC_API_KEY "
                "in backend/.env and restart the server."
            ),
            "actions": [],
        }
        if _lf:
            _lf.update_current_trace(output=result)
        return result

    # ── Build context ─────────────────────────────────────────────────────────
    user_content = _build_user_content(screenshot_b64, message, camera)
    messages = _history_to_messages(history) + [
        {"role": "user", "content": user_content}
    ]

    # ── Call Claude ───────────────────────────────────────────────────────────
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
    except Exception as exc:
        sentry_sdk.capture_exception(exc)
        if _lf:
            _lf.update_current_trace(output={"error": str(exc)})
        return {
            "answer": f"Sorry, the agent hit an error talking to the model: {exc}",
            "actions": [],
        }

    if getattr(resp, "stop_reason", None) == "refusal":
        result = {"answer": "I can't help with that request.", "actions": []}
        if _lf:
            _lf.update_current_trace(output=result)
        return result

    # ── Parse response ────────────────────────────────────────────────────────
    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text = block.text
            break
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        result = {
            "answer": "I had trouble forming a response. Could you rephrase that?",
            "actions": [],
        }
        if _lf:
            _lf.update_current_trace(output=result)
        return result

    if not isinstance(data, dict):
        data = {}
    actions = data.get("actions")
    if not isinstance(actions, list):
        actions = []
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = "Done."
    diagram = data.get("diagram") if isinstance(data.get("diagram"), str) else ""
    action_types = [a.get("type") for a in actions if isinstance(a, dict)]

    # ── Token usage ───────────────────────────────────────────────────────────
    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    cache_write = getattr(usage, "cache_creation_input_tokens", 0)

    # Sentry: measurements surface in the Performance dashboard charts.
    sentry_sdk.set_measurement("agent.input_tokens", input_tokens, "none")
    sentry_sdk.set_measurement("agent.output_tokens", output_tokens, "none")
    sentry_sdk.set_measurement("agent.action_count", len(actions), "none")

    for action in actions:
        if isinstance(action, dict):
            sentry_sdk.add_breadcrumb(
                category="agent.action",
                message=f"action dispatched: {action.get('type')}",
                data={k: v for k, v in action.items() if k != "target_2d"},
                level="info",
            )

    sentry_sdk.add_breadcrumb(
        category="agent",
        message="agent pipeline completed",
        data={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "action_count": len(actions),
            "action_types": action_types,
        },
        level="info",
    )

    # Langfuse: full prompt/response/token trace.
    if _lf:
        _lf.update_current_trace(
            output={"answer": answer, "action_types": action_types},
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "action_count": len(actions),
                "model": config.AGENT_MODEL,
            },
        )

    return {"answer": answer, "diagram": diagram, "actions": actions}


async def stream_agent(
    message: str,
    screenshot_b64: Optional[str],
    camera: Dict[str, Any],
    history: List[Dict[str, str]],
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async generator yielding SSE-style dicts: {type:'text',delta:str} then {type:'done',...}."""
    if _async_client is None:
        yield {
            "type": "done",
            "answer": "Navigation agent not configured. Set ANTHROPIC_API_KEY in backend/.env.",
            "diagram": "",
            "actions": [],
        }
        return

    user_content = _build_user_content(screenshot_b64, message, camera)
    messages_list = _history_to_messages(history) + [{"role": "user", "content": user_content}]

    accumulated = ""
    last_answer = ""

    try:
        async with _async_client.messages.stream(
            model=config.AGENT_MODEL,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": ACTIONS_SCHEMA}},
            messages=messages_list,
        ) as stream:
            async for text_chunk in stream.text_stream:
                accumulated += text_chunk
                partial = _extract_partial_answer(accumulated)
                if len(partial) > len(last_answer):
                    delta = partial[len(last_answer):]
                    yield {"type": "text", "delta": delta}
                    last_answer = partial
    except Exception as exc:
        sentry_sdk.capture_exception(exc)
        yield {
            "type": "done",
            "answer": f"Sorry, the agent hit an error: {exc}",
            "diagram": "",
            "actions": [],
        }
        return

    try:
        data = json.loads(accumulated) if accumulated else {}
    except json.JSONDecodeError:
        data = {}

    answer = data.get("answer") if isinstance(data.get("answer"), str) else ""
    answer = answer.strip() or last_answer or "Done."
    diagram = data.get("diagram") if isinstance(data.get("diagram"), str) else ""
    actions = data.get("actions") if isinstance(data.get("actions"), list) else []

    yield {"type": "done", "answer": answer, "diagram": diagram, "actions": actions}
