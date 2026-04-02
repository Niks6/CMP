"""
spam_task.py — Task 1 (Easy): Spam Detection.

The agent must distinguish spam posts from benign content.
Difficulty is easy because spam posts have distinctive signals:
  - High spam_score
  - Specific keyword patterns
  - High report_count
  - Low user_reputation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from env.observation_model import Post, PostCategory
from env.action_model import ModerationAction
from env.moderation_env import ContentModerationEnv
from dataset.dataset_generator import generate_task_dataset
from graders.spam_grader import SpamGrader


@dataclass
class SpamTask:
    """
    Task 1 — Spam Detection (Easy difficulty).

    The agent sees a mixed queue of spam and benign posts. The goal is to
    REMOVE spam and ALLOW benign content.

    Grading:
        score = (correctly_removed_spam + correctly_allowed_benign) / total_posts
    """

    name: str = "spam_detection"
    difficulty: str = "easy"
    description: str = (
        "Detect and remove spam posts from a mixed moderation queue. "
        "Spam posts feature promotional language, suspicious links, and low user reputation."
    )
    n_posts: int = 200
    benign_ratio: float = 0.40
    seed: int = 42
    _dataset: Optional[list[Post]] = field(default=None, init=False, repr=False)

    def get_dataset(self) -> list[Post]:
        """Return (and cache) the task dataset."""
        if self._dataset is None:
            self._dataset = generate_task_dataset(
                category=PostCategory.SPAM,
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
        Grade a sequence of agent actions against ground-truth post categories.

        Args:
            actions: List of ModerationAction, one per post (in order).
            posts:   The posts that were acted on (in the same order as actions).

        Returns:
            float in [0.0, 1.0] — fraction of posts handled correctly.
        """
        return SpamGrader().grade(actions=actions, posts=posts)
