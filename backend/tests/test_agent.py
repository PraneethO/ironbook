"""Tests for the reasoning navigation agent endpoint.

The LLM is always mocked — the suite never hits the network and never requires
ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import app.services.agent_llm as agent_llm


def test_act_passes_through_actions(client, monkeypatch):
    captured = {}

    def fake_run(message, screenshot_b64, camera, history):
        captured["message"] = message
        captured["screenshot_b64"] = screenshot_b64
        captured["camera"] = camera
        captured["history"] = history
        return {
            "answer": "Highlighting the fountain.",
            "actions": [
                {"type": "highlight", "target_2d": [0.5, 0.4], "label": "fountain"},
                {"type": "fly_to", "target_2d": [0.5, 0.4]},
            ],
        }

    monkeypatch.setattr(agent_llm, "run_agent", fake_run)

    body = {
        "message": "highlight the fountain and move to it",
        "screenshot_b64": "AAAA",
        "camera": {"mode": "orbit", "fov": 1.0, "eye": [0, 0, 5],
                   "target": [0, 0, 0], "bounds": {"min": [-1, -1, -1], "max": [1, 1, 1]}},
        "history": [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}],
    }
    r = client.post("/api/agent/act", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"] == "Highlighting the fountain."
    assert len(data["actions"]) == 2
    assert data["actions"][0]["type"] == "highlight"
    assert data["actions"][0]["target_2d"] == [0.5, 0.4]
    # request was forwarded intact
    assert captured["message"].startswith("highlight the fountain")
    assert captured["screenshot_b64"] == "AAAA"
    assert captured["camera"]["mode"] == "orbit"
    assert len(captured["history"]) == 2


def test_act_works_without_screenshot(client, monkeypatch):
    monkeypatch.setattr(
        agent_llm, "run_agent",
        lambda *a, **k: {"answer": "ok", "actions": []},
    )
    r = client.post("/api/agent/act", json={"message": "reset the view"})
    assert r.status_code == 200
    assert r.json() == {"answer": "ok", "actions": []}


def test_run_agent_unconfigured_is_friendly(monkeypatch):
    # No client (missing key) -> friendly answer, no actions, no network.
    monkeypatch.setattr(agent_llm, "_client", None)
    out = agent_llm.run_agent("anything", None, {"mode": "orbit"}, [])
    assert out["actions"] == []
    assert isinstance(out["answer"], str) and out["answer"]


def test_actions_schema_is_strict():
    # Structured-output schema must be strict (additionalProperties False, required).
    schema = agent_llm.ACTIONS_SCHEMA
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"answer", "actions"}
    item = schema["properties"]["actions"]["items"]
    assert item["additionalProperties"] is False
