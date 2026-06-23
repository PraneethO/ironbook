"""End-to-end API tests against CONTRACT.md using FastAPI TestClient."""
from __future__ import annotations

import io

from tests.conftest import make_image_bytes, upload_payload


# --- health ---------------------------------------------------------------


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # No GPU/COLMAP here, so it's the on-device depth reconstructor when the
    # model is present, otherwise the procedural fallback. Either way it must
    # match whatever the app actually selected.
    from app.reconstruction import select_backend

    assert body["reconstruction_backend"] == select_backend().name
    assert body["reconstruction_backend"] in {
        "brush", "msplat", "gaussian_3dgs", "depth", "fallback", "colmap_gsplat"
    }


# --- project CRUD ---------------------------------------------------------


def test_project_crud_lifecycle(client):
    # create
    r = client.post("/api/projects", json={"name": "My World"})
    assert r.status_code == 201
    proj = r.json()
    pid = proj["id"]
    assert proj["name"] == "My World"
    assert proj["status"] == "draft"
    assert proj["photo_count"] == 0
    assert proj["has_asset"] is False
    assert proj["thumbnail_url"] is None

    # get
    r = client.get(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert r.json()["id"] == pid

    # list contains it
    r = client.get("/api/projects")
    assert r.status_code == 200
    assert any(p["id"] == pid for p in r.json())

    # delete
    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 204

    # gone
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_list_newest_first(client):
    ids = []
    for name in ["a", "b", "c"]:
        ids.append(client.post("/api/projects", json={"name": name}).json()["id"])
    listed = [p["id"] for p in client.get("/api/projects").json()]
    # newest (c) first
    assert listed[0] == ids[-1]
    assert listed.index(ids[2]) < listed.index(ids[0])


def test_get_missing_project_404(client):
    assert client.get("/api/projects/deadbeef").status_code == 404
    assert client.delete("/api/projects/deadbeef").status_code == 404


# --- uploads + validation -------------------------------------------------


def _new_project(client, name="P"):
    return client.post("/api/projects", json={"name": name}).json()["id"]


def test_upload_accepts_good_images(client):
    pid = _new_project(client)
    r = client.post(f"/api/projects/{pid}/uploads", files=upload_payload(10))
    assert r.status_code == 200
    report = r.json()
    assert report["accepted"] == 10
    assert report["photo_count"] == 10
    assert report["rejected"] == []
    assert 0.0 <= report["coverage_score"] <= 1.0
    assert 0.0 <= report["quality_score"] <= 1.0
    assert report["ready_to_reconstruct"] is True

    # uploads listing has metadata + thumbnail_url
    r = client.get(f"/api/projects/{pid}/uploads")
    items = r.json()
    assert len(items) == 10
    for it in items:
        assert it["thumbnail_url"]
        assert it["width"] > 0 and it["height"] > 0
        assert "sharpness" in it

    # cover thumbnail served
    r = client.get(f"/api/projects/{pid}/thumbnail")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")

    # project now reflects photo_count + thumbnail_url
    p = client.get(f"/api/projects/{pid}").json()
    assert p["photo_count"] == 10
    assert p["thumbnail_url"] == f"/api/projects/{pid}/thumbnail"


def test_upload_rejects_bad_types(client):
    pid = _new_project(client)
    files = [
        ("files", ("notes.txt", b"hello", "text/plain")),
        ("files", ("clip.mp4", b"\x00\x00\x00", "video/mp4")),
        ("files", ("good.jpg", make_image_bytes(1), "image/jpeg")),
    ]
    r = client.post(f"/api/projects/{pid}/uploads", files=files)
    report = r.json()
    assert report["accepted"] == 1
    reasons = {x["filename"]: x["reason"] for x in report["rejected"]}
    assert "notes.txt" in reasons
    assert "clip.mp4" in reasons
    # friendly: no jargon
    assert "ffmpeg" not in reasons["clip.mp4"].lower()
    assert "video" in reasons["clip.mp4"].lower()


def test_too_few_photos_not_ready(client):
    pid = _new_project(client)
    r = client.post(f"/api/projects/{pid}/uploads", files=upload_payload(3))
    report = r.json()
    assert report["accepted"] == 3
    assert report["ready_to_reconstruct"] is False
    assert any("at least" in w.lower() for w in report["warnings"])


def test_below_recommended_warns(client):
    pid = _new_project(client)
    r = client.post(f"/api/projects/{pid}/uploads", files=upload_payload(10))
    report = r.json()
    assert report["ready_to_reconstruct"] is True
    # below 20 => recommendation + side-view hint
    assert any("side views" in w.lower() for w in report["warnings"])


def test_blurry_warning(client):
    pid = _new_project(client)
    r = client.post(f"/api/projects/{pid}/uploads", files=upload_payload(10, sharp=False))
    report = r.json()
    assert any("blurry" in w.lower() for w in report["warnings"])
    # blurry images => lower quality score
    assert report["quality_score"] < 0.5


def test_upload_missing_project_404(client):
    assert (
        client.post("/api/projects/nope/uploads", files=upload_payload(1)).status_code
        == 404
    )


# --- reconstruct job (inline) --------------------------------------------


def test_reconstruct_runs_to_ready(client):
    pid = _new_project(client)
    client.post(f"/api/projects/{pid}/uploads", files=upload_payload(10))

    r = client.post(f"/api/projects/{pid}/reconstruct")
    assert r.status_code == 200
    job = r.json()
    # inline => completes immediately
    assert job["project_id"] == pid
    assert job["status"] == "ready"
    assert job["progress"] == 1.0
    keys = [s["key"] for s in job["stages"]]
    assert keys == [
        "preprocessing",
        "pose_estimation",
        "structure",
        "optimization",
        "compression",
        "viewer_asset",
    ]
    assert all(s["status"] == "done" for s in job["stages"])
    assert job["logs"] and job["logs"][0]["level"] == "info"

    # job endpoint reflects ready
    jr = client.get(f"/api/projects/{pid}/job").json()
    assert jr["status"] == "ready"

    # project status ready + has_asset
    p = client.get(f"/api/projects/{pid}").json()
    assert p["status"] == "ready"
    assert p["has_asset"] is True


def test_reconstruct_too_few_photos_rejected(client):
    pid = _new_project(client)
    client.post(f"/api/projects/{pid}/uploads", files=upload_payload(3))
    r = client.post(f"/api/projects/{pid}/reconstruct")
    assert r.status_code == 400
    assert "at least" in r.json()["detail"].lower()


def test_job_before_reconstruct_404(client):
    pid = _new_project(client)
    assert client.get(f"/api/projects/{pid}/job").status_code == 404


# --- assets ---------------------------------------------------------------


def _reconstructed_project(client):
    pid = _new_project(client)
    client.post(f"/api/projects/{pid}/uploads", files=upload_payload(10))
    client.post(f"/api/projects/{pid}/reconstruct")
    return pid


def test_asset_bytes_and_count(client):
    pid = _reconstructed_project(client)
    r = client.get(f"/api/projects/{pid}/asset")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    data = r.content
    assert len(data) > 0
    assert len(data) % 32 == 0

    info = client.get(f"/api/projects/{pid}/asset/info").json()
    assert info["format"] == "splat"
    assert info["bytes"] == len(data)
    assert info["splat_count"] == len(data) // 32
    assert "min" in info["bounds"] and "max" in info["bounds"]
    assert len(info["bounds"]["min"]) == 3
    # floor near y=0
    assert info["bounds"]["min"][1] >= -0.5


def test_asset_404_before_ready(client):
    pid = _new_project(client)
    assert client.get(f"/api/projects/{pid}/asset").status_code == 404
    assert client.get(f"/api/projects/{pid}/asset/info").status_code == 404


def test_asset_missing_project_404(client):
    assert client.get("/api/projects/nope/asset").status_code == 404
    assert client.get("/api/projects/nope/asset/info").status_code == 404


# --- share ----------------------------------------------------------------


def test_share_url_shape(client):
    pid = _new_project(client)
    r = client.get(f"/api/projects/{pid}/share")
    assert r.status_code == 200
    assert r.json()["url"] == f"http://localhost:5173/view/{pid}"


def test_share_missing_project_404(client):
    assert client.get("/api/projects/nope/share").status_code == 404


def test_thumbnail_404_when_none(client):
    pid = _new_project(client)
    assert client.get(f"/api/projects/{pid}/thumbnail").status_code == 404
