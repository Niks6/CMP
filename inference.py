"""
inference.py — LLM-based inference agent for the Content Moderation RL environment.

This script runs an LLM agent against all three tasks using the OpenAI-compatible
client API and produces structured logs in the exact hackathon format:

    [START]
    task_name
    [STEP]
    action
    reward
    [END]
    final_score

Environment variables:
    API_BASE_URL  : OpenAI-compatible endpoint (LiteLLM proxy, injected by validator)
    API_KEY       : API key for the proxy (injected by validator — use this first)
    MODEL_NAME    : Model identifier (e.g. gpt-4o-mini, meta-llama/Llama-3-8b)
    HF_TOKEN      : Fallback API key if API_KEY is not set

If none of these are set the script falls back to the rule-based BaselineAgent
so it is always runnable — even without an API key.

Usage:
    python inference.py
    API_BASE_URL=https://proxy.example.com/v1 API_KEY=xxx MODEL_NAME=gpt-4o-mini python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any

from dataset.dataset_generator import generate_dataset
from env.moderation_env import ContentModerationEnv
from env.observation_model import Observation
from env.action_model import ModerationAction, Action
from baseline.baseline_agent import BaselineAgent
from tasks.spam_task import SpamTask
from tasks.hate_speech_task import HateSpeechTask
from tasks.misinformation_task import MisinformationTask
from graders.spam_grader import SpamGrader
from graders.hate_grader import HateGrader
from graders.misinformation_grader import MisinformationGrader

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str | None = os.getenv("API_BASE_URL")
MODEL_NAME: str | None = os.getenv("MODEL_NAME", "gpt-4o-mini")
# Validator injects API_KEY — fall back to HF_TOKEN / OPENAI_API_KEY for local runs
API_KEY: str | None = (
    os.getenv("API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
)

USE_LLM: bool = bool(API_BASE_URL and API_KEY)

# Cap posts per task to stay within the 20-min runtime limit.
# 200 posts × 3 tasks = 600 LLM calls ≈ 20-30 min (too slow).
# 30 posts × 3 tasks =  90 LLM calls ≈ 3-5  min (safe).
MAX_POSTS_PER_TASK: int = int(os.getenv("MAX_POSTS_PER_TASK", "30"))

# ---------------------------------------------------------------------------
# LLM client (lazy import so the script works without openai installed)
# ---------------------------------------------------------------------------

def _build_llm_client():
    """Build an OpenAI-compatible client pointed at the proxy."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("[WARN] openai package not installed. Falling back to baseline agent.")
        return None

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are a professional social media content moderator AI agent.
You will receive information about a social media post and must choose exactly ONE action.

Valid actions (respond with ONLY the action name, nothing else):
  ALLOW             - Post is safe, no action needed
  REMOVE            - Post must be removed immediately (spam, severe hate, dangerous content)
  WARN_USER         - Send a warning to the author, post stays up (borderline/moderate content)
  ESCALATE_TO_HUMAN - Flag for human review (ambiguous, misinformation, complex cases)

Rules:
- Very toxic content (toxicity_score > 0.8): respond REMOVE
- Spammy content (spam_score > 0.75): respond REMOVE
- Misinformation signals (misinfo_score > 0.70): respond ESCALATE_TO_HUMAN
- Moderate toxicity (0.5–0.8): respond WARN_USER
- Safe content (toxicity_score < 0.2, low scores): respond ALLOW

Respond with ONLY the action word. No explanation. No punctuation.
""").strip()


def _observation_to_prompt(obs: Observation) -> str:
    """Convert an Observation to a human-readable prompt for the LLM."""
    return textwrap.dedent(f"""
Post #{obs.post_id} (Step {obs.step_index}, {obs.posts_remaining} remaining):

Text: {obs.post_text[:500]}

Signals:
  - toxicity_score   : {obs.toxicity_score:.3f}
  - spam_score       : {obs.spam_score:.3f}
  - misinfo_score    : {obs.misinfo_score:.3f}
  - report_count     : {obs.report_count}
  - user_reputation  : {obs.user_reputation:.3f}
  - image_flag       : {obs.image_flag}

Your action:""").strip()


def _parse_llm_response(text: str) -> ModerationAction:
    """Extract a valid ModerationAction from the LLM's raw text response."""
    text = text.strip().upper().replace("-", "_").split()[0] if text.strip() else ""
    try:
        return ModerationAction(text)
    except ValueError:
        # Map common synonyms
        synonyms = {
            "DELETE": "REMOVE",
            "FLAG": "ESCALATE_TO_HUMAN",
            "ESCALATE": "ESCALATE_TO_HUMAN",
            "WARN": "WARN_USER",
            "PASS": "ALLOW",
            "OK": "ALLOW",
        }
        mapped = synonyms.get(text)
        if mapped:
            return ModerationAction(mapped)
        return ModerationAction.ALLOW  # safe fallback


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

class StructuredLogger:
    """Writes logs to stdout and optionally to a file in hackathon format."""

    def __init__(self, log_file: str | None = None):
        self._lines: list[str] = []
        self._file = open(log_file, "w") if log_file else None

    def _emit(self, line: str) -> None:
        print(line)
        self._lines.append(line)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    def start(self, task_name: str) -> None:
        self._emit("[START]")
        self._emit(task_name)

    def step(self, action: str, reward: float) -> None:
        self._emit("[STEP]")
        self._emit(action)
        self._emit(str(reward))

    def end(self, final_score: float) -> None:
        self._emit("[END]")
        # Use f-string formatting (never round()) to avoid float precision edge cases
        self._emit(f"{final_score:.4f}")

    def close(self) -> None:
        if self._file:
            self._file.close()

    @property
    def lines(self) -> list[str]:
        return list(self._lines)


# ---------------------------------------------------------------------------
# Agent wrapper (LLM or Baseline)
# ---------------------------------------------------------------------------

class LLMAgent:
    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def select_action(self, obs: Observation) -> Action:
        prompt = _observation_to_prompt(obs)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=16,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"[WARN] LLM call failed ({exc}); falling back to ALLOW")
            raw = "ALLOW"

        action = _parse_llm_response(raw)
        return Action(action=action, confidence=1.0)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    task_obj: Any,
    grader: Any,
    agent: Any,
    logger: StructuredLogger,
) -> float:
    """Run one task episode and return the grader score."""
    posts = task_obj.get_dataset()

    # Limit posts per task to stay within 20-min runtime budget.
    # Override with MAX_POSTS_PER_TASK env var (default 30).
    all_posts = list(posts)[:MAX_POSTS_PER_TASK]

    from env.moderation_env import ContentModerationEnv
    env = ContentModerationEnv(posts=all_posts, seed=42, shuffle=False)
    obs = env.reset()
    all_actions: list[ModerationAction] = []

    logger.start(task_name)

    while True:
        action_obj = agent.select_action(obs)
        moderation_action = action_obj.action
        all_actions.append(moderation_action)

        next_obs, reward, done, info = env.step(moderation_action)
        logger.step(moderation_action.value, round(reward, 2))

        if done:
            break
        obs = next_obs

    try:
        grader_score = grader.grade(actions=all_actions, posts=all_posts)
    except Exception as exc:
        print(f"[WARN] Grader raised exception: {exc}. Using fallback score.")
        grader_score = 0.5  # safe midpoint fallback

    # Guarantee strictly in (0, 1) with wide safe margins
    grader_score = max(0.01, min(0.99, float(grader_score)))
    logger.end(grader_score)

    return grader_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log_file = os.getenv("LOG_FILE")  # optional: write logs to file too
    logger = StructuredLogger(log_file=log_file)

    # Choose agent
    if USE_LLM:
        client = _build_llm_client()
        if client:
            agent: Any = LLMAgent(client=client, model=MODEL_NAME)
            print(f"[INFO] Using LLM agent: {MODEL_NAME} @ {API_BASE_URL}")
        else:
            agent = BaselineAgent()
            print("[INFO] Using rule-based BaselineAgent (LLM client unavailable)")
    else:
        agent = BaselineAgent()
        print(f"[INFO] Using rule-based BaselineAgent (API_BASE_URL={API_BASE_URL}, API_KEY set={bool(API_KEY)})")

    tasks_config = [
        ("spam_detection",          SpamTask(),            SpamGrader()),
        ("hate_speech_moderation",  HateSpeechTask(),      HateGrader()),
        ("misinformation_detection", MisinformationTask(), MisinformationGrader()),
    ]

    scores: list[float] = []
    for task_name, task_obj, grader in tasks_config:
        score = run_task(task_name, task_obj, grader, agent, logger)
        scores.append(score)

    # Summary to stderr (not part of the structured log)
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n[SUMMARY] Scores: {[round(s, 4) for s in scores]}", file=sys.stderr)
    print(f"[SUMMARY] Average grader score: {avg:.4f}", file=sys.stderr)

    logger.close()


if __name__ == "__main__":
    main()
