"""
baseline_agent.py — Rule-based baseline agent for the Content Moderation RL environment.

The baseline agent makes decisions purely based on observable signal thresholds:
  - High toxicity (> 0.8)          → REMOVE
  - High spam signal (> 0.75)      → REMOVE
  - High misinfo signal (> 0.70)   → ESCALATE_TO_HUMAN
  - Medium toxicity (> 0.5)        → WARN_USER
  - Medium spam or misinfo         → WARN_USER
  - High report count (> 30)       → ESCALATE_TO_HUMAN
  - Low user reputation (< 0.2)    → WARN_USER
  - Otherwise                      → ALLOW

This agent serves as a performance lower-bound. A well-trained RL agent
should significantly outperform it, especially on Task 3 (Misinformation).
"""

from __future__ import annotations

from env.observation_model import Observation
from env.action_model import Action, ModerationAction


class BaselineAgent:
    """
    Rule-based baseline content moderation agent.

    Uses threshold heuristics on observable signals to choose a moderation
    action. No learning occurs — the same rules apply to every post.

    This is useful as a performance benchmark and sanity check for the
    environment's reward function.
    """

    # Thresholds (tuned to match the reward function's intended behaviour)
    HIGH_TOXICITY: float = 0.8
    MEDIUM_TOXICITY: float = 0.5
    HIGH_SPAM: float = 0.75
    MEDIUM_SPAM: float = 0.45
    HIGH_MISINFO: float = 0.70
    MEDIUM_MISINFO: float = 0.40
    HIGH_REPORT_COUNT: int = 30
    LOW_REPUTATION: float = 0.2

    def select_action(self, obs: Observation) -> Action:
        """
        Choose a moderation action based on observable signals.

        Args:
            obs: The current Observation from the environment.

        Returns:
            Action: A Pydantic Action model wrapping the chosen ModerationAction.
        """
        action = self._decide(obs)
        return Action(action=action, confidence=self._confidence(obs, action))

    def _decide(self, obs: Observation) -> ModerationAction:
        """Core decision logic (separated for readability and easy override)."""

        # --- Immediate removal signals ---
        if obs.toxicity_score > self.HIGH_TOXICITY:
            return ModerationAction.REMOVE

        if obs.spam_score > self.HIGH_SPAM:
            return ModerationAction.REMOVE

        # --- Escalation signals (misinformation is hard to judge definitively) ---
        if obs.misinfo_score > self.HIGH_MISINFO:
            return ModerationAction.ESCALATE_TO_HUMAN

        if obs.report_count > self.HIGH_REPORT_COUNT and obs.misinfo_score > self.MEDIUM_MISINFO:
            return ModerationAction.ESCALATE_TO_HUMAN

        # --- Warning signals ---
        if obs.toxicity_score > self.MEDIUM_TOXICITY:
            return ModerationAction.WARN_USER

        if obs.spam_score > self.MEDIUM_SPAM:
            return ModerationAction.WARN_USER

        if obs.misinfo_score > self.MEDIUM_MISINFO:
            return ModerationAction.WARN_USER

        if obs.user_reputation < self.LOW_REPUTATION and obs.report_count > 5:
            return ModerationAction.WARN_USER

        if obs.image_flag and obs.toxicity_score > 0.35:
            return ModerationAction.WARN_USER

        # --- Default: allow the post ---
        return ModerationAction.ALLOW

    def _confidence(self, obs: Observation, action: ModerationAction) -> float:
        """
        Estimate agent confidence (0.0–1.0) based on signal strength.

        Higher confidence when signals are clearly above/below thresholds.
        This does not affect the reward but is useful for analysis.
        """
        if action == ModerationAction.REMOVE:
            return min(1.0, max(obs.toxicity_score, obs.spam_score))
        elif action == ModerationAction.ESCALATE_TO_HUMAN:
            return min(1.0, obs.misinfo_score)
        elif action == ModerationAction.WARN_USER:
            return min(1.0, max(obs.toxicity_score, obs.spam_score, obs.misinfo_score))
        else:  # ALLOW
            # Confidence in ALLOW is inverse of the maximum harmful signal
            max_signal = max(obs.toxicity_score, obs.spam_score, obs.misinfo_score)
            return 1.0 - max_signal

    def __repr__(self) -> str:
        return (
            f"BaselineAgent("
            f"high_tox={self.HIGH_TOXICITY}, "
            f"high_spam={self.HIGH_SPAM}, "
            f"high_misinfo={self.HIGH_MISINFO})"
        )
