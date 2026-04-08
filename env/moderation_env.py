"""
moderation_env.py — Main Content Moderation RL Environment (OpenEnv interface).

The environment presents the agent with a queue of social media posts one at a time.
At each step the agent submits a ModerationAction and receives:
  - the next Observation
  - a scalar reward
  - a `done` flag
  - an info dict

The environment follows the OpenEnv specification:
    reset()  -> Observation
    step()   -> (Observation, float, bool, dict)
    state()  -> dict
"""

from __future__ import annotations

import random
from typing import Any, Optional

from env.observation_model import Observation, Post, RewardInfo
from env.action_model import Action, ModerationAction
from reward.reward_function import compute_reward


class ContentModerationEnv:
    """
    OpenEnv-compatible Content Moderation Reinforcement Learning Environment.

    Args:
        posts:   A list of Post objects forming the moderation queue for this episode.
        seed:    Optional random seed for reproducibility.
        shuffle: Whether to shuffle the post queue on `reset()`.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        posts: list[Post],
        seed: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        if not posts:
            raise ValueError("ContentModerationEnv requires at least one post.")

        self._posts_source: list[Post] = posts
        self._shuffle: bool = shuffle
        self._rng = random.Random(seed)

        # Episode state (populated by reset())
        self._queue: list[Post] = []
        self._step_idx: int = 0
        self._current_post: Optional[Post] = None
        self._episode_rewards: list[RewardInfo] = []
        self._history: list[dict[str, Any]] = []
        self._done: bool = True  # Must call reset() before first step

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment and return the first observation.

        Returns:
            Observation: The first post's observable features.
        """
        self._queue = list(self._posts_source)
        if self._shuffle:
            self._rng.shuffle(self._queue)

        self._step_idx = 0
        self._episode_rewards = []
        self._history = []
        self._done = False

        self._current_post = self._queue[self._step_idx]
        return Observation.from_post(
            self._current_post,
            step_index=self._step_idx,
            posts_remaining=len(self._queue) - self._step_idx - 1,
        )

    def step(
        self, action: ModerationAction | Action | str
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Apply a moderation action and advance to the next post.

        Args:
            action: A ModerationAction enum member, an Action Pydantic model,
                    or a string matching a ModerationAction value.

        Returns:
            observation: Next Observation (or the current final obs if done).
            reward:      Scalar reward for this action.
            done:        True when all posts in the queue have been processed.
            info:        Dict with detailed reward info and step metadata.

        Raises:
            RuntimeError: If called before `reset()` or after the episode has ended.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_post is None:
            raise RuntimeError("No current post. Call reset() first.")

        # Normalise action type
        moderation_action = self._parse_action(action)

        # Compute reward using the current post's ground-truth category
        reward_info = compute_reward(self._current_post, moderation_action)

        # Record history for this step
        step_record: dict[str, Any] = {
            "step": self._step_idx,
            "post_id": self._current_post.post_id,
            "post_text": self._current_post.post_text,
            "category": self._current_post.category.value,
            "action": moderation_action.value,
            "reward": reward_info.value,
            "correct": reward_info.correct,
            "reason": reward_info.reason,
        }
        self._history.append(step_record)
        self._episode_rewards.append(reward_info)

        # Advance to the next post
        self._step_idx += 1
        done = self._step_idx >= len(self._queue)
        self._done = done

        if not done:
            self._current_post = self._queue[self._step_idx]
            next_obs = Observation.from_post(
                self._current_post,
                step_index=self._step_idx,
                posts_remaining=len(self._queue) - self._step_idx - 1,
            )
        else:
            # Episode over — return the *last* observation again (with posts_remaining=0)
            next_obs = Observation.from_post(
                self._queue[self._step_idx - 1],
                step_index=self._step_idx - 1,
                posts_remaining=0,
            )

        info: dict[str, Any] = {
            "reward_info": reward_info.model_dump(),
            "step": self._step_idx - 1,
            "total_posts": len(self._queue),
            "cumulative_reward": sum(r.value for r in self._episode_rewards),
            "correct_actions": sum(1 for r in self._episode_rewards if r.correct),
        }

        return next_obs, float(reward_info.value), done, info

    def state(self) -> dict[str, Any]:
        """
        Return the current internal environment state.

        Returns:
            Dict containing episode progress, queue stats, and running totals.
        """
        return {
            "step_index": self._step_idx,
            "total_posts": len(self._queue),
            "posts_remaining": max(0, len(self._queue) - self._step_idx),
            "done": self._done,
            "cumulative_reward": sum(r.value for r in self._episode_rewards),
            "correct_actions": sum(1 for r in self._episode_rewards if r.correct),
            "total_actions": len(self._episode_rewards),
            "current_post_id": self._current_post.post_id if self._current_post else None,
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def episode_summary(self) -> dict[str, Any]:
        """
        Summarise the completed (or in-progress) episode.

        Returns:
            Dict with total reward, accuracy, and per-category breakdown.
        """
        total = len(self._episode_rewards)
        correct = sum(1 for r in self._episode_rewards if r.correct)
        accuracy = correct / total if total > 0 else 0.5
        # Clamp strictly within (0, 1) — validator requires scores not equal to 0.0 or 1.0
        accuracy = max(0.01, min(0.99, accuracy))

        # Per-category breakdown
        from collections import defaultdict
        cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        for record in self._history:
            cat = record["category"]
            cat_stats[cat]["total"] += 1
            if record["correct"]:
                cat_stats[cat]["correct"] += 1

        per_category = {
            cat: {
                "accuracy": (v["correct"] / v["total"]) if v["total"] > 0 else 0.0,
                **v,
            }
            for cat, v in cat_stats.items()
        }

        return {
            "total_posts": total,
            "correct_actions": correct,
            "accuracy": accuracy,
            "total_reward": sum(r.value for r in self._episode_rewards),
            "per_category": per_category,
        }

    @property
    def history(self) -> list[dict[str, Any]]:
        """Step-by-step history of the current episode."""
        return list(self._history)

    @property
    def action_space(self) -> list[ModerationAction]:
        """All available moderation actions."""
        return list(ModerationAction)

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return len(ModerationAction)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(action: ModerationAction | Action | str) -> ModerationAction:
        """Normalise various action representations to a ModerationAction enum."""
        if isinstance(action, ModerationAction):
            return action
        if isinstance(action, Action):
            return action.action
        if isinstance(action, str):
            try:
                return ModerationAction(action.upper())
            except ValueError:
                raise ValueError(
                    f"Unknown action string '{action}'. "
                    f"Valid actions: {ModerationAction.values()}"
                )
        raise TypeError(f"Unsupported action type: {type(action)}")
