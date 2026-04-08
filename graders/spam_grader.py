"""
spam_grader.py — Programmatic grader for Task 1: Spam Detection.

Grading metric:
    score = (correctly_removed_spam + correctly_allowed_benign) / total_posts

A "correct" decision is:
  - REMOVE for SPAM posts
  - ALLOW (or WARN_USER) for BENIGN posts
"""

from __future__ import annotations

from env.observation_model import Post, PostCategory
from env.action_model import ModerationAction


class SpamGrader:
    """
    Grader for the Spam Detection task.

    Evaluates whether the agent correctly identifies spam (REMOVE) and
    passes through benign content (ALLOW or WARN_USER).
    """

    def grade(
        self,
        actions: list[ModerationAction],
        posts: list[Post],
    ) -> float:
        """
        Grade the agent's actions.

        Args:
            actions: Ordered list of ModerationAction choices (one per post).
            posts:   Ordered list of Post objects corresponding to the actions.

        Returns:
            float in [0.0, 1.0] — fraction of correctly handled posts.

        Raises:
            ValueError: If actions and posts have different lengths.
        """
        if len(actions) != len(posts):
            raise ValueError(
                f"Length mismatch: {len(actions)} actions vs {len(posts)} posts."
            )

        if not posts:
            return 0.01

        correct = 0
        for action, post in zip(actions, posts):
            if post.category == PostCategory.SPAM:
                # Correct: REMOVE spam
                if action == ModerationAction.REMOVE:
                    correct += 1
            else:
                # Correct: do NOT remove benign content
                if action in (ModerationAction.ALLOW, ModerationAction.WARN_USER):
                    correct += 1

        raw = correct / len(posts)
        return max(0.01, min(0.99, raw))

    def detailed_report(
        self,
        actions: list[ModerationAction],
        posts: list[Post],
    ) -> dict:
        """
        Return a detailed breakdown of grading results.

        Returns:
            Dict with overall score, true positives, false positives, false negatives,
            and precision / recall for spam detection.
        """
        tp = fp = fn = tn = 0
        for action, post in zip(actions, posts):
            is_spam = post.category == PostCategory.SPAM
            predicted_spam = action == ModerationAction.REMOVE

            if is_spam and predicted_spam:
                tp += 1
            elif not is_spam and predicted_spam:
                fp += 1
            elif is_spam and not predicted_spam:
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
            "score": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }
