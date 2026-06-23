"""Reasoning navigation agent endpoint (CONTRACT extension).

POST /api/agent/act  — given the user's message + current viewer screenshot +
camera, returns a friendly `answer` and a list of `actions` (move / rotate /
zoom / fly_to / look_at / highlight / clear_highlight / reset_view) for the
frontend to execute against the splat viewer.
"""
from __future__ import annotations

from fastapi import APIRouter
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


class VoiceConfig(BaseModel):
    deepgram_key: str
    model: str = "nova-2"


@router.get("/voice-config", response_model=VoiceConfig)
def voice_config() -> VoiceConfig:
    """Return the Deepgram key from server config so it never ships in the JS bundle."""
    return VoiceConfig(deepgram_key=config.DEEPGRAM_API_KEY)
