"""
hate_speech_task.py — Task 2 (Medium): Hate Speech Moderation.

The agent must identify hate speech and apply appropriate actions:
  - Severe hate speech (toxicity > 0.8) → REMOVE
  - Moderate hate speech (toxicity 0.5–0.8) → WARN_USER
  - Benign content → ALLOW

Medium difficulty because hate speech varies in toxicity and requires nuance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from env.observation_model import Post, PostCategory
from env.action_model import ModerationAction
from env.moderation_env import ContentModerationEnv
from dataset.dataset_generator import generate_task_dataset
from graders.hate_grader import HateGrader


@dataclass
class HateSpeechTask:
    """
    Task 2 — Hate Speech Moderation (Medium difficulty).

    The agent sees hate speech posts of varying severity mixed with benign content.
    The grader rewards both correct identification AND appropriate severity calibration.

    Grading:
        A weighted score accounting for severity-appropriate action selection.
    """

    name: str = "hate_speech_moderation"
    difficulty: str = "medium"
    description: str = (
        "Moderate a queue containing hate speech posts of varying severity "
        "alongside benign content. Severe hate speech requires removal; "
        "moderate hate speech warrants a user warning."
    )
    n_posts: int = 200
    benign_ratio: float = 0.35
    seed: int = 123
    _dataset: Optional[list[Post]] = field(default=None, init=False, repr=False)

    def get_dataset(self) -> list[Post]:
        """Return (and cache) the task dataset."""
        if self._dataset is None:
            self._dataset = generate_task_dataset(
                category=PostCategory.HATE_SPEECH,
                n=self.n_posts,
                benign_ratio=self.benign_ratio,
                seed=self.seed,
            )
        return self._dataset

    def make_env(self, shuffle: bool = True) -> ContentModerationEnv:
        """Create a ContentModerationEnv loaded with this task's dataset."""
        return ContentModerationEnv(
            posts=self.get_dataset(),
            seed=self.seed,
            shuffle=shuffle,
        )

    def grade(
        self,
        actions: list[ModerationAction],
        posts: list[Post],
    ) -> float:
        """
        Grade a sequence of agent actions.

        Args:
            actions: List of ModerationAction, one per post.
            posts:   Posts acted on in the same order.

        Returns:
            float in [0.0, 1.0].
        """
        return HateGrader().grade(actions=actions, posts=posts)
