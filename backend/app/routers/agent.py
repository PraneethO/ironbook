"""Reasoning navigation agent endpoint (CONTRACT extension).

POST /api/agent/act        — returns a complete {answer, diagram, actions} JSON.
POST /api/agent/act/stream — SSE stream: text deltas then a final done event.
"""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import config
from ..models import AgentActRequest, AgentActResponse
from ..services import agent_llm

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/act", response_model=AgentActResponse)
def act(req: AgentActRequest) -> AgentActResponse:
    result = agent_llm.run_agent(
        req.message,
        req.screenshot_b64,
        req.camera.model_dump(),
        [t.model_dump() for t in req.history],
    )
    return AgentActResponse(**result)


@router.post("/act/stream")
async def act_stream(req: AgentActRequest) -> StreamingResponse:
    async def generate():
        async for event in agent_llm.stream_agent(
            req.message,
            req.screenshot_b64,
            req.camera.model_dump(),
            [t.model_dump() for t in req.history],
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class VoiceConfig(BaseModel):
    deepgram_key: str
    model: str = "nova-2"


@router.get("/voice-config", response_model=VoiceConfig)
def voice_config() -> VoiceConfig:
    """Return the Deepgram key from server config so it never ships in the JS bundle."""
    return VoiceConfig(deepgram_key=config.DEEPGRAM_API_KEY)
