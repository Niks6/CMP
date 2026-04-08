"""
run_baseline.py — Demo entry-point for the Content Moderation RL Environment.

This script:
  1. Generates a synthetic dataset of social media posts
  2. Runs the rule-based BaselineAgent on all three tasks
  3. Prints per-step rewards (configurable verbosity) and final grader scores

Usage:
    python run_baseline.py
    python run_baseline.py --verbose     # print every step
    python run_baseline.py --quiet       # print only final scores
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

# --- Rich for pretty CLI output ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from dataset.dataset_generator import generate_dataset
from env.moderation_env import ContentModerationEnv
from env.action_model import ModerationAction
from baseline.baseline_agent import BaselineAgent
from tasks.spam_task import SpamTask
from tasks.hate_speech_task import HateSpeechTask
from tasks.misinformation_task import MisinformationTask
from graders.spam_grader import SpamGrader
from graders.hate_grader import HateGrader
from graders.misinformation_grader import MisinformationGrader


def _print(msg: str, style: str = "") -> None:
    """Unified print — uses Rich if available."""
    if HAS_RICH:
        Console().print(msg, style=style)
    else:
        print(msg)


def run_task(
    task_name: str,
    task_obj: Any,
    grader: Any,
    agent: BaselineAgent,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run one task end-to-end with the baseline agent.

    Args:
        task_name: Human-readable task label.
        task_obj:  Task dataclass (SpamTask, HateSpeechTask, or MisinformationTask).
        grader:    Corresponding grader instance.
        agent:     BaselineAgent instance.
        verbose:   If True, print every step's action and reward.

    Returns:
        Dict with task results including score, reward, and accuracy.
    """
    posts = task_obj.get_dataset()
    env = task_obj.make_env(shuffle=False)  # fixed order for reproducibility

    obs = env.reset()

    all_actions: list[ModerationAction] = []
    all_posts = list(posts)  # preserve original order (env doesn't shuffle with shuffle=False)
    total_reward = 0.0
    step = 0

    if verbose and HAS_RICH:
        console = Console()
        console.print(f"\n[bold cyan]── {task_name} Steps ──[/bold cyan]")

    while True:
        action_obj = agent.select_action(obs)
        moderation_action = action_obj.action
        all_actions.append(moderation_action)

        next_obs, reward, done, info = env.step(moderation_action)
        total_reward += reward

        if verbose:
            step_info = info["reward_info"]
            if HAS_RICH:
                correct_icon = "✓" if step_info["correct"] else "✗"
                style = "green" if step_info["correct"] else "red"
                Console().print(
                    f"  [{style}]{correct_icon}[/{style}] "
                    f"Step {step:>3} | "
                    f"Action: {moderation_action.value:<20} | "
                    f"Reward: {reward:>+6.1f} | "
                    f"Category: {step_info['post_category']}"
                )
            else:
                correct_str = "OK" if step_info["correct"] else "WRONG"
                print(
                    f"  [{correct_str}] Step {step:>3} | "
                    f"Action: {moderation_action.value:<20} | "
                    f"Reward: {reward:>+6.1f} | "
                    f"Category: {step_info['post_category']}"
                )

        step += 1
        if done:
            break
        obs = next_obs

    # Compute grader score
    grader_score = grader.grade(actions=all_actions, posts=all_posts)
    # Validator requires each task score to be strictly between 0 and 1.
    # Also avoid rounding down to 0.0 or up to 1.0.
    grader_score = max(1e-4, min(1 - 1e-4, float(grader_score)))
    summary = env.episode_summary()

    return {
        "task_name": task_name,
        "difficulty": task_obj.difficulty,
        "total_posts": len(all_posts),
        "total_reward": round(total_reward, 2),
        "accuracy": round(summary["accuracy"], 4),
        "grader_score": round(grader_score, 4),
        "correct_actions": summary["correct_actions"],
    }


def main(verbose: bool = False, quiet: bool = False) -> None:
    console = Console() if HAS_RICH else None

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    if HAS_RICH:
        console.print(
            Panel.fit(
                "[bold white]Content Moderation RL Environment[/bold white]\n"
                "[dim]Baseline Agent Demo — OpenEnv Specification[/dim]",
                border_style="bright_blue",
            )
        )
    else:
        print("=" * 60)
        print("  Content Moderation RL Environment — Baseline Agent Demo")
        print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Generate global dataset (shown for completeness)
    # -----------------------------------------------------------------------
    if not quiet:
        _print("\n[bold]Step 1:[/bold] Generating synthetic post dataset (500 posts)…" if HAS_RICH
               else "\nStep 1: Generating synthetic post dataset (500 posts)...")

    global_dataset = generate_dataset(n=500, seed=42)

    if not quiet:
        cat_counts: dict[str, int] = {}
        for p in global_dataset:
            cat_counts[p.category.value] = cat_counts.get(p.category.value, 0) + 1

        if HAS_RICH:
            tbl = Table(title="Dataset Summary", box=box.SIMPLE)
            tbl.add_column("Category", style="cyan")
            tbl.add_column("Count", style="white", justify="right")
            tbl.add_column("Fraction", style="dim", justify="right")
            for cat, cnt in sorted(cat_counts.items()):
                tbl.add_row(cat, str(cnt), f"{cnt/len(global_dataset):.1%}")
            console.print(tbl)
        else:
            print(f"  Dataset: {cat_counts}")

    # -----------------------------------------------------------------------
    # Step 2: Initialise agent and tasks
    # -----------------------------------------------------------------------
    if not quiet:
        _print("\n[bold]Step 2:[/bold] Initialising BaselineAgent and tasks…" if HAS_RICH
               else "\nStep 2: Initialising BaselineAgent and tasks...")

    agent = BaselineAgent()
    tasks_config = [
        ("Task 1 — Spam Detection (Easy)", SpamTask(), SpamGrader()),
        ("Task 2 — Hate Speech Moderation (Medium)", HateSpeechTask(), HateGrader()),
        ("Task 3 — Misinformation Detection (Hard)", MisinformationTask(), MisinformationGrader()),
    ]

    # -----------------------------------------------------------------------
    # Step 3: Run baseline agent on all tasks
    # -----------------------------------------------------------------------
    if not quiet:
        _print("\n[bold]Step 3:[/bold] Running baseline agent on all tasks…" if HAS_RICH
               else "\nStep 3: Running baseline agent on all tasks...")

    results: list[dict[str, Any]] = []
    for task_label, task_obj, grader in tasks_config:
        if not quiet:
            _print(f"\n▶ {task_label}" if HAS_RICH else f"\n> {task_label}")
        t0 = time.perf_counter()
        result = run_task(
            task_name=task_label,
            task_obj=task_obj,
            grader=grader,
            agent=agent,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        result["elapsed_s"] = round(elapsed, 3)
        results.append(result)

    # -----------------------------------------------------------------------
    # Step 4: Print final scores
    # -----------------------------------------------------------------------
    _print("\n[bold]Step 4:[/bold] Final Results" if HAS_RICH else "\nStep 4: Final Results")

    if HAS_RICH:
        results_table = Table(title="Baseline Agent — Final Scores", box=box.ROUNDED)
        results_table.add_column("Task", style="cyan", no_wrap=True)
        results_table.add_column("Difficulty", style="yellow", justify="center")
        results_table.add_column("Posts", justify="right")
        results_table.add_column("Correct", justify="right")
        results_table.add_column("Accuracy", justify="right")
        results_table.add_column("Grader Score", justify="right", style="bold")
        results_table.add_column("Total Reward", justify="right")
        results_table.add_column("Time (s)", justify="right", style="dim")

        for r in results:
            score = r["grader_score"]
            score_style = "green" if score >= 0.7 else ("yellow" if score >= 0.5 else "red")
            results_table.add_row(
                r["task_name"],
                r["difficulty"],
                str(r["total_posts"]),
                str(r["correct_actions"]),
                f"{r['accuracy']:.1%}",
                f"[{score_style}]{score:.4f}[/{score_style}]",
                f"{r['total_reward']:+.1f}",
                str(r["elapsed_s"]),
            )
        console.print(results_table)

        avg_score = sum(r["grader_score"] for r in results) / len(results)
        console.print(
            f"\n[bold]Overall average grader score:[/bold] "
            f"[{'green' if avg_score >= 0.65 else 'yellow'}]{avg_score:.4f}[/]"
        )
    else:
        print("\n" + "-" * 80)
        print(f"{'Task':<45} {'Difficulty':<10} {'Score':>8} {'Accuracy':>10} {'Reward':>10}")
        print("-" * 80)
        for r in results:
            print(
                f"{r['task_name']:<45} {r['difficulty']:<10} "
                f"{r['grader_score']:>8.4f} {r['accuracy']:>10.1%} "
                f"{r['total_reward']:>+10.1f}"
            )
        print("-" * 80)
        avg_score = sum(r["grader_score"] for r in results) / len(results)
        print(f"\nOverall average grader score: {avg_score:.4f}")

    # Return exit code 0 if all tasks pass their target score threshold
    # (Task 1: 0.80, Task 2: 0.70, Task 3: 0.60)
    targets = [0.80, 0.70, 0.60]
    all_passed = all(r["grader_score"] >= t for r, t in zip(results, targets))
    if not quiet:
        if all_passed:
            _print("\n[bold green]✓ All tasks met their target scores![/bold green]" if HAS_RICH
                   else "\n[PASS] All tasks met their target scores!")
        else:
            failed = [
                r["task_name"] for r, t in zip(results, targets)
                if r["grader_score"] < t
            ]
            _print(
                f"\n[bold yellow]⚠ Some tasks did not meet targets: {failed}[/bold yellow]"
                if HAS_RICH
                else f"\n[WARN] Some tasks did not meet targets: {failed}"
            )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Content Moderation RL baseline agent on all tasks."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print every step's action and reward.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Print only the final score table.",
    )
    args = parser.parse_args()
    main(verbose=args.verbose, quiet=args.quiet)
