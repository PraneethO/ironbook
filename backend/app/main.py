"""Gaussian Splat World — backend API.

FastAPI app implementing CONTRACT.md. Photos in, a navigable `.splat` 3D world
out, with friendly staged progress. Runs on a CPU fallback reconstructor here;
the COLMAP+gsplat path plugs in behind the same interface when a GPU exists.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .routers import agent, health, projects
from .services import store

app = FastAPI(title="Gaussian Splat World API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_BASE_URL, "*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(projects.router)
app.include_router(agent.router)


@app.on_event("startup")
def _on_startup() -> None:
    config.ensure_storage()
    # Recover the project index if it drifted from disk (e.g. after a crash).
    store.rebuild_index_from_disk()
