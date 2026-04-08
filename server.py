"""
server.py — FastAPI HTTP server for the Content Moderation RL Environment.

Exposes the OpenEnv API over HTTP so the hackathon validator and external
agents can interact with the environment without importing Python directly.

Endpoints:
    GET  /health          → liveness probe (HuggingFace Spaces requirement)
    POST /reset           → env.reset()  → Observation JSON
    POST /step            → env.step()   → {observation, reward, done, info}
    GET  /state           → env.state()  → internal state dict
    GET  /action_space    → list of valid action strings

Usage:
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dataset.dataset_generator import generate_dataset
from env.moderation_env import ContentModerationEnv
from env.action_model import ModerationAction
from env.observation_model import Post

# ---------------------------------------------------------------------------
# Global environment state (one session for simplicity)
# ---------------------------------------------------------------------------

_env: ContentModerationEnv | None = None
_posts: list[Post] = []


def _get_env() -> ContentModerationEnv:
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )
    return _env


# ---------------------------------------------------------------------------
# Lifespan: pre-generate dataset on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _posts
    seed = int(os.getenv("DATASET_SEED", "42"))
    n = int(os.getenv("DATASET_SIZE", "500"))
    _posts = generate_dataset(n=n, seed=seed)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Content Moderation RL Environment",
    description=(
        "OpenEnv-compatible environment exposing a social-media content "
        "moderation task over HTTP. Implements reset / step / state."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    action: str = Field(
        ...,
        description="Moderation action: ALLOW | REMOVE | WARN_USER | ESCALATE_TO_HUMAN",
        examples=["REMOVE"],
    )


class ResetRequest(BaseModel):
    shuffle: bool = Field(default=True, description="Shuffle the post queue on reset")
    seed: int | None = Field(default=None, description="Optional RNG seed")
    n_posts: int = Field(default=200, ge=1, le=500, description="Number of posts in episode")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness probe")
def health() -> dict[str, str]:
    """Returns HTTP 200 so HuggingFace Spaces marks the container healthy."""
    return {"status": "ok", "environment": "content_moderation_rl", "version": "1.0.0"}


@app.get("/", summary="Root — API info")
def root() -> dict[str, Any]:
    """Root endpoint — returns API overview."""
    return {
        "name": "Content Moderation RL Environment",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/reset", "/step", "/state", "/action_space", "/episode_summary"],
        "docs": "/docs",
    }


@app.post("/reset", summary="Reset environment and get first observation")
def reset(req: ResetRequest = ResetRequest()) -> dict[str, Any]:
    """
    Initialise (or re-initialise) the environment episode.

    Returns the first Observation as a JSON object.
    """
    global _env
    posts_slice = _posts[: req.n_posts]
    if not posts_slice:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    _env = ContentModerationEnv(posts=posts_slice, seed=req.seed, shuffle=req.shuffle)
    obs = _env.reset()
    return {
        "observation": obs.model_dump(),
        "episode_info": {
            "n_posts": len(posts_slice),
            "shuffle": req.shuffle,
        },
    }


@app.post("/step", summary="Submit a moderation action")
def step(req: StepRequest) -> dict[str, Any]:
    """
    Apply `action` to the current post and advance the episode.

    Returns: observation, reward, done, info
    """
    env = _get_env()
    try:
        action = ModerationAction(req.action.upper())
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{req.action}'. Valid: {ModerationAction.values()}",
        )

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", summary="Get current internal environment state")
def state() -> dict[str, Any]:
    """
    Return the environment's internal state snapshot.

    Useful for debugging. Does not expose hidden ground-truth labels.
    """
    env = _get_env()
    return env.state()


@app.get("/action_space", summary="List valid moderation actions")
def action_space() -> dict[str, Any]:
    """Return all valid action strings and their indices."""
    return {
        "actions": [
            {"name": a.value, "index": ModerationAction.index_of(a)}
            for a in ModerationAction
        ]
    }


@app.get("/episode_summary", summary="Get episode summary (after done=True)")
def episode_summary() -> dict[str, Any]:
    """Return accuracy and per-category breakdown for the current episode."""
    env = _get_env()
    return env.episode_summary()
