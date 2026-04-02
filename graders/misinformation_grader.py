"""
misinformation_grader.py — Programmatic grader for Task 3: Misinformation Detection.

Grading uses an F1-balanced score that weighs precision and recall equally,
preventing agents from gaming the score by escalating everything.

Correct actions for MISINFORMATION: ESCALATE_TO_HUMAN or REMOVE
Correct actions for non-MISINFORMATION: ALLOW or WARN_USER

Final score:
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    score     = F1 = 2 * precision * recall / (precision + recall)
"""

from __future__ import annotations

from env.observation_model import Post, PostCategory
from env.action_model import ModerationAction


class MisinformationGrader:
    """
    Grader for the Misinformation Detection task.

    Uses F1-score to balance precision (avoid over-flagging) and recall
    (avoid missing real misinformation). This is harder to game than accuracy.
    """

    # Actions that count as "flagging" a post as misinformation
    FLAGGING_ACTIONS: frozenset[ModerationAction] = frozenset(
        {ModerationAction.ESCALATE_TO_HUMAN, ModerationAction.REMOVE}
    )

    def grade(
        self,
        actions: list[ModerationAction],
        posts: list[Post],
    ) -> float:
        """
        Grade actions using F1-score for misinformation detection.

        Args:
            actions: Ordered list of ModerationAction choices.
            posts:   Corresponding Post objects.

        Returns:
            float in [0.0, 1.0] — F1-balanced score.
        """
        if len(actions) != len(posts):
            raise ValueError(
                f"Length mismatch: {len(actions)} actions vs {len(posts)} posts."
            )

        if not posts:
            return 0.0

        tp = fp = fn = 0
        for action, post in zip(actions, posts):
            is_misinfo = post.category == PostCategory.MISINFORMATION
            flagged = action in self.FLAGGING_ACTIONS

            if is_misinfo and flagged:
                tp += 1
            elif not is_misinfo and flagged:
                fp += 1
            elif is_misinfo and not flagged:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return round(f1, 4)

    def detailed_report(
        self,
        actions: list[ModerationAction],
        posts: list[Post],
    ) -> dict:
        """
        Return a detailed breakdown with precision, recall, F1, and action distribution.

        Returns:
            Dict with all evaluation metrics and per-action counts.
        """
        tp = fp = fn = tn = 0
        action_counts: dict[str, int] = {a.value: 0 for a in ModerationAction}

        for action, post in zip(actions, posts):
            is_misinfo = post.category == PostCategory.MISINFORMATION
            flagged = action in self.FLAGGING_ACTIONS
            action_counts[action.value] += 1

            if is_misinfo and flagged:
                tp += 1
            elif not is_misinfo and flagged:
                fp += 1
            elif is_misinfo and not flagged:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / len(posts) if posts else 0.0

        return {
            "score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "action_distribution": action_counts,
            "total_posts": len(posts),
            "misinformation_posts": tp + fn,
        }
