"""
action_model.py — Action space for the Content Moderation RL environment.

Defines the four possible moderation actions and a Pydantic wrapper model.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class ModerationAction(str, Enum):
    """
    The four moderation actions available to the RL agent.

    - ALLOW           : Post is deemed safe; no action taken.
    - REMOVE          : Post is immediately removed from the platform.
    - WARN_USER       : A warning is sent to the post author; post remains visible.
    - ESCALATE_TO_HUMAN: Flagged for human review (used for borderline / complex cases).
    """

    ALLOW = "ALLOW"
    REMOVE = "REMOVE"
    WARN_USER = "WARN_USER"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"

    # Convenience helpers -------------------------------------------------
    @classmethod
    def values(cls) -> list[str]:
        """Return all action value strings."""
        return [a.value for a in cls]

    @classmethod
    def index_of(cls, action: "ModerationAction") -> int:
        """Return the integer index of an action (useful for gym-style envs)."""
        return list(cls).index(action)

    @classmethod
    def from_index(cls, idx: int) -> "ModerationAction":
        """Return the action at the given integer index."""
        members = list(cls)
        if idx < 0 or idx >= len(members):
            raise ValueError(f"Action index {idx} is out of range [0, {len(members)-1}]")
        return members[idx]


class Action(BaseModel):
    """Pydantic wrapper around a ModerationAction, as submitted by the agent."""

    action: ModerationAction = Field(
        ...,
        description="The chosen moderation action",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this action (optional metadata; does not affect reward).",
    )

    def __str__(self) -> str:
        return self.action.value
