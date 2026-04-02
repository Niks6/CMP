"""
reward_function.py — Reward computation for the Content Moderation RL environment.

The reward function encodes the human moderation policy:
  - High-toxicity content should be REMOVED
  - Medium-toxicity content should trigger a WARN_USER
  - Spam should be REMOVED
  - Misinformation should be ESCALATED for human review
  - Benign content should be ALLOWED

Incorrect actions carry negative rewards to teach the agent to avoid:
  - Letting harmful content through (false negatives)
  - Unnecessary removals of benign content (false positives)
  - Excessive / unjustified escalation
"""

from __future__ import annotations

from env.observation_model import Post, PostCategory, RewardInfo
from env.action_model import ModerationAction


# ---------------------------------------------------------------------------
# Reward table — (correct_action, reward), keyed by scenario
# ---------------------------------------------------------------------------

# Positive rewards for correct decisions
REWARD_HIGH_TOXICITY_REMOVE: float = 10.0    # toxicity > 0.8 → REMOVE
REWARD_MED_TOXICITY_WARN: float = 5.0        # 0.5 < toxicity ≤ 0.8 → WARN_USER
REWARD_LOW_TOXICITY_ALLOW: float = 3.0       # toxicity < 0.2 → ALLOW
REWARD_SPAM_REMOVE: float = 8.0              # spam → REMOVE
REWARD_MISINFO_ESCALATE: float = 7.0         # misinformation → ESCALATE_TO_HUMAN
REWARD_HATE_REMOVE: float = 9.0              # hate speech (high tox) → REMOVE
REWARD_HATE_WARN: float = 4.0               # hate speech (med tox) → WARN_USER

# Negative rewards for incorrect decisions
PENALTY_IGNORE_HARMFUL: float = -10.0        # allowing high-toxicity / spam / misinfo
PENALTY_UNNECESSARY_REMOVE: float = -5.0     # removing benign content
PENALTY_EXCESSIVE_ESCALATE: float = -2.0     # escalating clearly benign / easy cases
PENALTY_MISSED_WARN: float = -4.0            # allowing medium-toxicity without action
PENALTY_WRONG_ACTION: float = -3.0           # any other incorrect action


def compute_reward(post: Post, action: ModerationAction) -> RewardInfo:
    """
    Compute the reward for taking `action` on `post`.

    Args:
        post:   The full Post object (including ground-truth category).
        action: The moderation action chosen by the agent.

    Returns:
        RewardInfo with scalar value, explanation, and correctness flag.
    """

    category = post.category
    tox = post.toxicity_score
    spam = post.spam_score
    misinfo = post.misinfo_score

    # -----------------------------------------------------------------------
    # SPAM posts
    # -----------------------------------------------------------------------
    if category == PostCategory.SPAM:
        if action == ModerationAction.REMOVE:
            return RewardInfo(
                value=REWARD_SPAM_REMOVE,
                reason="Correctly removed spam post.",
                correct=True,
                post_category=category.value,
                action_taken=action.value,
            )
        elif action == ModerationAction.ESCALATE_TO_HUMAN:
            # Acceptable but suboptimal — spam is clear-cut
            return RewardInfo(
                value=PENALTY_EXCESSIVE_ESCALATE,
                reason="Escalated obvious spam instead of removing directly.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )
        elif action == ModerationAction.WARN_USER:
            return RewardInfo(
                value=PENALTY_WRONG_ACTION,
                reason="Only warned user for spam; should have removed.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )
        else:  # ALLOW
            return RewardInfo(
                value=PENALTY_IGNORE_HARMFUL,
                reason="Allowed spam to remain — missed detection.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )

    # -----------------------------------------------------------------------
    # HATE SPEECH posts
    # -----------------------------------------------------------------------
    if category == PostCategory.HATE_SPEECH:
        if tox > 0.8:
            # Severe hate speech → must REMOVE
            if action == ModerationAction.REMOVE:
                return RewardInfo(
                    value=REWARD_HATE_REMOVE,
                    reason="Correctly removed severe hate speech (toxicity > 0.8).",
                    correct=True,
                    post_category=category.value,
                    action_taken=action.value,
                )
            elif action == ModerationAction.WARN_USER:
                return RewardInfo(
                    value=PENALTY_MISSED_WARN,
                    reason="Only warned for severe hate speech; removal was required.",
                    correct=False,
                    post_category=category.value,
                    action_taken=action.value,
                )
            elif action == ModerationAction.ESCALATE_TO_HUMAN:
                return RewardInfo(
                    value=PENALTY_EXCESSIVE_ESCALATE,
                    reason="Escalated clear-cut severe hate speech unnecessarily.",
                    correct=False,
                    post_category=category.value,
                    action_taken=action.value,
                )
            else:  # ALLOW
                return RewardInfo(
                    value=PENALTY_IGNORE_HARMFUL,
                    reason="Allowed severe hate speech — critical miss.",
                    correct=False,
                    post_category=category.value,
                    action_taken=action.value,
                )
        else:
            # Moderate hate speech → WARN_USER
            if action == ModerationAction.WARN_USER:
                return RewardInfo(
                    value=REWARD_HATE_WARN,
                    reason="Correctly warned user for moderate hate speech.",
                    correct=True,
                    post_category=category.value,
                    action_taken=action.value,
                )
            elif action == ModerationAction.REMOVE:
                # Slightly too aggressive but not completely wrong
                return RewardInfo(
                    value=1.0,
                    reason="Removed moderate hate speech — arguably acceptable.",
                    correct=True,
                    post_category=category.value,
                    action_taken=action.value,
                )
            elif action == ModerationAction.ESCALATE_TO_HUMAN:
                return RewardInfo(
                    value=PENALTY_EXCESSIVE_ESCALATE,
                    reason="Escalated moderate hate speech; a warning would suffice.",
                    correct=False,
                    post_category=category.value,
                    action_taken=action.value,
                )
            else:  # ALLOW
                return RewardInfo(
                    value=PENALTY_IGNORE_HARMFUL,
                    reason="Allowed moderate hate speech without action.",
                    correct=False,
                    post_category=category.value,
                    action_taken=action.value,
                )

    # -----------------------------------------------------------------------
    # MISINFORMATION posts
    # -----------------------------------------------------------------------
    if category == PostCategory.MISINFORMATION:
        if action == ModerationAction.ESCALATE_TO_HUMAN:
            return RewardInfo(
                value=REWARD_MISINFO_ESCALATE,
                reason="Correctly escalated misinformation for human review.",
                correct=True,
                post_category=category.value,
                action_taken=action.value,
            )
        elif action == ModerationAction.REMOVE:
            # Removing misinformation is acceptable (though escalation is preferred)
            return RewardInfo(
                value=3.0,
                reason="Removed misinformation — acceptable, though human review preferred.",
                correct=True,
                post_category=category.value,
                action_taken=action.value,
            )
        elif action == ModerationAction.WARN_USER:
            return RewardInfo(
                value=PENALTY_WRONG_ACTION,
                reason="Warning insufficient for misinformation; removal or escalation needed.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )
        else:  # ALLOW
            return RewardInfo(
                value=PENALTY_IGNORE_HARMFUL,
                reason="Allowed misinformation to spread — dangerous miss.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )

    # -----------------------------------------------------------------------
    # BENIGN posts
    # -----------------------------------------------------------------------
    # category == PostCategory.BENIGN
    if tox < 0.2:
        if action == ModerationAction.ALLOW:
            return RewardInfo(
                value=REWARD_LOW_TOXICITY_ALLOW,
                reason="Correctly allowed low-toxicity benign content.",
                correct=True,
                post_category=category.value,
                action_taken=action.value,
            )
        elif action == ModerationAction.REMOVE:
            return RewardInfo(
                value=PENALTY_UNNECESSARY_REMOVE,
                reason="Unnecessarily removed benign low-toxicity content.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )
        elif action == ModerationAction.ESCALATE_TO_HUMAN:
            return RewardInfo(
                value=PENALTY_EXCESSIVE_ESCALATE,
                reason="Escalated clearly benign content — wasted human reviewer time.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )
        else:  # WARN_USER
            return RewardInfo(
                value=PENALTY_WRONG_ACTION,
                reason="Warned user for benign content — false positive.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )
    elif 0.2 <= tox <= 0.5:
        # Medium-low benign: ALLOW is still best; mild warning is tolerable
        if action == ModerationAction.ALLOW:
            return RewardInfo(
                value=2.0,
                reason="Allowed borderline-benign content — acceptable.",
                correct=True,
                post_category=category.value,
                action_taken=action.value,
            )
        elif action == ModerationAction.WARN_USER:
            return RewardInfo(
                value=1.0,
                reason="Warned for borderline benign content — cautious but acceptable.",
                correct=True,
                post_category=category.value,
                action_taken=action.value,
            )
        elif action == ModerationAction.REMOVE:
            return RewardInfo(
                value=PENALTY_UNNECESSARY_REMOVE,
                reason="Removed borderline-benign content — over-moderation.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )
        else:  # ESCALATE_TO_HUMAN
            return RewardInfo(
                value=PENALTY_EXCESSIVE_ESCALATE,
                reason="Escalated borderline-benign content unnecessarily.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )
    else:
        # High-toxicity benign (rare edge case) → should at least WARN
        if action in (ModerationAction.WARN_USER, ModerationAction.REMOVE):
            return RewardInfo(
                value=REWARD_MED_TOXICITY_WARN,
                reason="Correctly acted on high-toxicity signal in benign post.",
                correct=True,
                post_category=category.value,
                action_taken=action.value,
            )
        else:
            return RewardInfo(
                value=PENALTY_WRONG_ACTION,
                reason="Did not act on high-toxicity signal in post.",
                correct=False,
                post_category=category.value,
                action_taken=action.value,
            )
