"""
misinformation_task.py — Task 3 (Hard): Misinformation Detection.

The hardest task. Misinformation posts can look innocuous in text but have
high misinfo_score signals. The agent must:
  - ESCALATE_TO_HUMAN or REMOVE misinformation
  - ALLOW benign/spam-like content that is not misinformation

Hard because misinformation overlaps visually with spam and toxicity signals
are not reliable indicators — the agent must learn to rely on misinfo_score
and report_count together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from env.observation_model import Post, PostCategory
from env.action_model import ModerationAction
from env.moderation_env import ContentModerationEnv
from dataset.dataset_generator import generate_task_dataset
from graders.misinformation_grader import MisinformationGrader


@dataclass
class MisinformationTask:
    """
    Task 3 — Misinformation Detection (Hard difficulty).

    The queue contains misinformation posts mixed with benign and spam content.
    The agent must learn to escalate or remove misinformation while avoiding
    false positives on benign content.

    Grading:
        Precision-recall balanced score for misinformation detection.
    """

    name: str = "misinformation_detection"
    difficulty: str = "hard"
    description: str = (
        "Identify and escalate misinformation posts in a complex mixed queue "
        "containing benign, spam, and misinformation content. Misinformation "
        "requires ESCALATE_TO_HUMAN or REMOVE; benign content must be allowed through."
    )
    n_posts: int = 200
    benign_ratio: float = 0.35
    seed: int = 999
    _dataset: Optional[list[Post]] = field(default=None, init=False, repr=False)

    def get_dataset(self) -> list[Post]:
        """Return (and cache) the task dataset."""
        if self._dataset is None:
            # Mix of misinformation, benign, and some spam sprinkled in via noise
            self._dataset = generate_task_dataset(
                category=PostCategory.MISINFORMATION,
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
        return MisinformationGrader().grade(actions=actions, posts=posts)
