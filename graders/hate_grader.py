"""
hate_grader.py — Programmatic grader for Task 2: Hate Speech Moderation.

Grading is severity-aware:
  - Severe hate speech (toxicity > 0.8) → correct if REMOVE (full credit)
  - Moderate hate speech (toxicity 0.5–0.8) → correct if WARN_USER or REMOVE (full credit)
  - Benign content → correct if ALLOW or WARN_USER

Final score = weighted_correct / total_posts
"""

from __future__ import annotations

from env.observation_model import Post, PostCategory
from env.action_model import ModerationAction


class HateGrader:
    """
    Grader for the Hate Speech Moderation task.

    Severity-calibrated: rewards agents that choose severity-appropriate actions,
    not just those that blindly remove all hate speech.
    """

    SEVERE_TOXICITY_THRESHOLD: float = 0.8
    MODERATE_TOXICITY_THRESHOLD: float = 0.5

    def grade(
        self,
        actions: list[ModerationAction],
        posts: list[Post],
    ) -> float:
        """
        Grade agent actions with severity awareness.

        Args:
            actions: Ordered list of ModerationAction choices.
            posts:   Corresponding Post objects.

        Returns:
            float in [0.0, 1.0].
        """
        if len(actions) != len(posts):
            raise ValueError(
                f"Length mismatch: {len(actions)} actions vs {len(posts)} posts."
            )

        if not posts:
            return 1e-4

        correct = 0
        for action, post in zip(actions, posts):
            if post.category == PostCategory.HATE_SPEECH:
                if post.toxicity_score > self.SEVERE_TOXICITY_THRESHOLD:
                    # Severe: must REMOVE
                    if action == ModerationAction.REMOVE:
                        correct += 1
                elif post.toxicity_score > self.MODERATE_TOXICITY_THRESHOLD:
                    # Moderate: WARN_USER or REMOVE both acceptable
                    if action in (ModerationAction.WARN_USER, ModerationAction.REMOVE):
                        correct += 1
                else:
                    # Mild hate speech: WARN_USER is best
                    if action == ModerationAction.WARN_USER:
                        correct += 1
            else:
                # Benign post: ALLOW or WARN_USER acceptable
                if action in (ModerationAction.ALLOW, ModerationAction.WARN_USER):
                    correct += 1

        raw = correct / len(posts)
        return max(1e-4, min(1 - 1e-4, raw))

    def detailed_report(
        self,
        actions: list[ModerationAction],
        posts: list[Post],
    ) -> dict:
        """
        Return a detailed breakdown by hate speech severity tier.

        Returns:
            Dict with per-tier accuracy and overall score.
        """
        tiers: dict[str, dict] = {
            "severe_hate": {"total": 0, "correct": 0},
            "moderate_hate": {"total": 0, "correct": 0},
            "mild_hate": {"total": 0, "correct": 0},
            "benign": {"total": 0, "correct": 0},
        }

        for action, post in zip(actions, posts):
            if post.category == PostCategory.HATE_SPEECH:
                if post.toxicity_score > self.SEVERE_TOXICITY_THRESHOLD:
                    tier = "severe_hate"
                    ok = action == ModerationAction.REMOVE
                elif post.toxicity_score > self.MODERATE_TOXICITY_THRESHOLD:
                    tier = "moderate_hate"
                    ok = action in (ModerationAction.WARN_USER, ModerationAction.REMOVE)
                else:
                    tier = "mild_hate"
                    ok = action == ModerationAction.WARN_USER
            else:
                tier = "benign"
                ok = action in (ModerationAction.ALLOW, ModerationAction.WARN_USER)

            tiers[tier]["total"] += 1
            if ok:
                tiers[tier]["correct"] += 1

        total = len(posts)
        overall_correct = sum(t["correct"] for t in tiers.values())
        score = overall_correct / total if total > 0 else 0.0

        per_tier = {
            tier: {
                "accuracy": (v["correct"] / v["total"]) if v["total"] > 0 else 0.0,
                **v,
            }
            for tier, v in tiers.items()
        }

        return {
            "score": round(score, 4),
            "overall_correct": overall_correct,
            "total": total,
            "per_tier": per_tier,
        }
