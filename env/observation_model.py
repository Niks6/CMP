"""
observation_model.py — Pydantic models for observations, posts, and reward info.

These models define the data contract between the environment, agent, and graders.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PostCategory(str, Enum):
    """Ground-truth category of a post, used internally by the environment."""
    BENIGN = "benign"
    SPAM = "spam"
    HATE_SPEECH = "hate_speech"
    MISINFORMATION = "misinformation"


class Post(BaseModel):
    """
    Raw post data as it arrives in the moderation queue.

    This is the *internal* representation used by the environment and graders.
    The agent only ever sees an `Observation` (which omits the ground-truth category).
    """

    post_id: int = Field(..., description="Unique post identifier")
    post_text: str = Field(..., description="The text content of the post")
    toxicity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Toxicity score in [0, 1]"
    )
    report_count: int = Field(..., ge=0, description="Number of user reports")
    user_reputation: float = Field(
        ..., ge=0.0, le=1.0, description="Author reputation in [0, 1]; higher is better"
    )
    image_flag: bool = Field(
        ..., description="True if the post contains a potentially harmful image"
    )
    category: PostCategory = Field(..., description="Ground-truth moderation category")
    spam_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Spam-likelihood score in [0, 1]"
    )
    misinfo_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Misinformation-likelihood score in [0, 1]"
    )


class Observation(BaseModel):
    """
    What the RL agent observes at each step.

    Derived from a Post but *excludes* the ground-truth category so the agent
    must learn entirely from observable signals.
    """

    post_id: int = Field(..., description="Unique post identifier")
    post_text: str = Field(..., description="The text content of the post")
    toxicity_score: float = Field(..., ge=0.0, le=1.0, description="Toxicity score in [0, 1]")
    report_count: int = Field(..., ge=0, description="Number of user reports")
    user_reputation: float = Field(
        ..., ge=0.0, le=1.0, description="Author reputation in [0, 1]; higher is better"
    )
    image_flag: bool = Field(..., description="True if the post contains a potentially harmful image")
    spam_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Spam-likelihood score")
    misinfo_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Misinformation score")
    step_index: int = Field(default=0, description="Current step index within the episode")
    posts_remaining: int = Field(default=0, description="Number of posts left in the queue")

    @classmethod
    def from_post(cls, post: Post, step_index: int = 0, posts_remaining: int = 0) -> "Observation":
        """Construct an Observation from a Post, stripping ground-truth labels."""
        return cls(
            post_id=post.post_id,
            post_text=post.post_text,
            toxicity_score=post.toxicity_score,
            report_count=post.report_count,
            user_reputation=post.user_reputation,
            image_flag=post.image_flag,
            spam_score=post.spam_score,
            misinfo_score=post.misinfo_score,
            step_index=step_index,
            posts_remaining=posts_remaining,
        )


class RewardInfo(BaseModel):
    """Structured reward information returned alongside the scalar reward."""

    value: float = Field(..., description="Scalar reward value")
    reason: str = Field(..., description="Human-readable explanation for the reward")
    correct: bool = Field(..., description="Whether the action was considered correct")
    post_category: str = Field(..., description="Ground-truth category of the post")
    action_taken: str = Field(..., description="The moderation action that was taken")
