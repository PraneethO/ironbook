"""Tests for the built-in demo scene (bundled bike, no upload needed)."""
from __future__ import annotations


def test_demo_creates_ready_project(client):
    r = client.post("/api/projects/demo")
    assert r.status_code == 201, r.text
    proj = r.json()
    assert proj["name"] == "Demo Bike"
    assert proj["status"] == "ready"
    assert proj["has_asset"] is True

    info = client.get(f"/api/projects/{proj['id']}/asset/info")
    assert info.status_code == 200
    assert info.json()["splat_count"] > 1000  # the real bundled scene


def test_demo_is_reused_not_duplicated(client):
    a = client.post("/api/projects/demo").json()
    b = client.post("/api/projects/demo").json()
    assert a["id"] == b["id"]  # repeated clicks reuse the one demo project
