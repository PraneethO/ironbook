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

import sentry_sdk

from .. import config

try:  # anthropic is optional at import time so tests/CI without it still load
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

# Langfuse is optional — degrades gracefully to no-ops when not configured.
try:
    from langfuse.decorators import langfuse_context as _lf_ctx
    from langfuse.decorators import observe as _lf_observe
    _langfuse_ok = True
except Exception:  # pragma: no cover
    _langfuse_ok = False

    class _NoopCtx:
        def update_current_trace(self, **kw: Any) -> None: pass
        def update_current_observation(self, **kw: Any) -> None: pass

    _lf_ctx = _NoopCtx()  # type: ignore[assignment]

    def _lf_observe(func: Any = None, **kw: Any) -> Any:  # type: ignore[misc]
        def dec(f: Any) -> Any:
            return f
        return dec if func is None else func


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


@_lf_observe(name="agent-act")
def run_agent(
    message: str,
    screenshot_b64: Optional[str],
    camera: Dict[str, Any],
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Return {"answer": str, "actions": [..]}. Never raises for normal errors."""
    sentry_sdk.set_tag("agent.model", config.AGENT_MODEL)

    with sentry_sdk.start_span(op="ai.pipeline", name="ironbook.agent.act") as pipeline_span:
        pipeline_span.set_attribute("gen_ai.request.model", config.AGENT_MODEL)
        pipeline_span.set_attribute("ai.has_screenshot", bool(screenshot_b64))
        pipeline_span.set_attribute("ai.message_length", len(message))
        pipeline_span.set_attribute("ai.history_turns", len(history))
        pipeline_span.set_attribute("ai.camera_mode", camera.get("mode", "unknown"))

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

        _lf_ctx.update_current_trace(
            tags=["navigation", "vision-grounded"],
            metadata={"model": config.AGENT_MODEL},
        )
        _lf_ctx.update_current_observation(
            input={
                "message": message,
                "has_screenshot": bool(screenshot_b64),
                "camera_mode": camera.get("mode"),
                "history_turns": len(history),
            },
            model=config.AGENT_MODEL,
        )

        if _client is None:
            pipeline_span.set_attribute("ai.skipped", True)
            result: Dict[str, Any] = {
                "answer": (
                    "The navigation agent isn't configured yet — set ANTHROPIC_API_KEY "
                    "in backend/.env and restart the server."
                ),
                "actions": [],
            }
            _lf_ctx.update_current_observation(output=result, level="WARNING")
            return result

        # ── Phase 1: build context ────────────────────────────────────────────
        with sentry_sdk.start_span(op="ai.prepare_input", name="build_context") as prep_span:
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
                prep_span.set_attribute("ai.screenshot_bytes", len(screenshot_b64))

            cam_note = (
                f"Camera mode={camera.get('mode')} fov_rad={camera.get('fov')}. "
                "The scene is roughly centered at the origin, Y is up."
            )
            user_content.append({"type": "text", "text": f"{cam_note}\n\nUser: {message}"})

            messages = _history_to_messages(history) + [
                {"role": "user", "content": user_content}
            ]
            prep_span.set_attribute("ai.context_turns", len(messages))

        # ── Phase 2: call Claude (AnthropicIntegration auto-spans this) ───────
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
            pipeline_span.set_status({"code": 2, "message": str(exc)})
            sentry_sdk.capture_exception(exc)
            _lf_ctx.update_current_observation(
                output={"error": str(exc)},
                level="ERROR",
                status_message=str(exc),
            )
            return {
                "answer": f"Sorry, the agent hit an error talking to the model: {exc}",
                "actions": [],
            }

        if getattr(resp, "stop_reason", None) == "refusal":
            pipeline_span.set_attribute("ai.refusal", True)
            result = {"answer": "I can't help with that request.", "actions": []}
            _lf_ctx.update_current_observation(output=result, level="WARNING")
            return result

        # ── Phase 3: parse response + extract actions ─────────────────────────
        with sentry_sdk.start_span(op="ai.parse_output", name="extract_actions") as parse_span:
            text = ""
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    text = block.text
                    break

            try:
                data = json.loads(text) if text else {}
            except json.JSONDecodeError:
                parse_span.set_attribute("ai.parse_error", True)
                result = {
                    "answer": "I had trouble forming a response. Could you rephrase that?",
                    "actions": [],
                }
                _lf_ctx.update_current_observation(output=result, level="WARNING")
                return result

            if not isinstance(data, dict):
                data = {}
            actions = data.get("actions")
            if not isinstance(actions, list):
                actions = []
            answer = data.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                answer = "Done."

            action_types = [a.get("type") for a in actions if isinstance(a, dict)]
            parse_span.set_attribute("ai.action_count", len(actions))
            parse_span.set_attribute("ai.action_types", ",".join(t for t in action_types if t))

        # ── Token usage + final span attributes ───────────────────────────────
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_write = getattr(usage, "cache_creation_input_tokens", 0)

        pipeline_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        pipeline_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        pipeline_span.set_attribute("gen_ai.usage.cache_read_tokens", cache_read)
        pipeline_span.set_attribute("gen_ai.usage.cache_write_tokens", cache_write)
        pipeline_span.set_attribute("ai.action_count", len(actions))
        pipeline_span.set_attribute("ai.action_types", ",".join(t for t in action_types if t))

        # Measurements show up as performance metrics in Sentry charts.
        sentry_sdk.set_measurement("agent.input_tokens", input_tokens, "none")
        sentry_sdk.set_measurement("agent.output_tokens", output_tokens, "none")
        sentry_sdk.set_measurement("agent.action_count", len(actions), "none")

        # One breadcrumb per executed action so the trace tells the full story.
        for action in actions:
            if isinstance(action, dict):
                sentry_sdk.add_breadcrumb(
                    category="agent.action",
                    message=f"action dispatched: {action.get('type')}",
                    data={k: v for k, v in action.items() if k != "target_2d"},
                    level="info",
                )

        _lf_ctx.update_current_observation(
            output={"answer": answer, "action_types": action_types},
            usage={"input": input_tokens, "output": output_tokens},
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

        return {"answer": answer, "actions": actions}
