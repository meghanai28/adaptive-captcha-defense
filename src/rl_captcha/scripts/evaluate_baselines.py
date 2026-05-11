"""Rule-based baseline policies and XGBoost comparison.

Baselines evaluated through the same EventEnv as the RL agents:
  random             -- uniform random valid action at every step
  always_block       -- continue on non-terminal; block on terminal
  always_allow       -- continue on non-terminal; allow on terminal
  always_easy_puzzle -- continue on non-terminal; easy CAPTCHA on terminal
  always_hard_puzzle -- continue on non-terminal; hard CAPTCHA on terminal
  honeypot_block     -- deploy up to 2 honeypots, then always block
  honeypot_decide    -- deploy honeypots; block if any triggered, else allow

XGBoost (session-level, no environment interaction):
  requires --xgboost-models classifier/models/xgb_v1 [xgb_v2 ...]

Usage:
    python -m rl_captcha.scripts.evaluate_baselines \\
        --data-dir data/ \\
        --reward-preset v2 \\
        --episodes 500 \\
        --eval-seeds 42 123 456 789 1024

    # With XGBoost comparison:
    python -m rl_captcha.scripts.evaluate_baselines \\
        --data-dir data/ --reward-preset v2 --episodes 500 \\
        --eval-seeds 42 123 456 789 1024 \\
        --xgboost-models classifier/models/xgb_v1 classifier/models/xgb_v2
"""

from __future__ import annotations

import argparse
import random as _random
import sys
import time
from dataclasses import replace

import numpy as np

from rl_captcha.config import REWARD_PRESETS
from rl_captcha.data.loader import (
    load_from_directory,
    split_sessions,
    split_sessions_by_family,
)
from rl_captcha.environment.event_env import EventEnv
from rl_captcha.scripts.evaluate_ppo import (
    _compute_metrics,
    _print_results,
    _print_per_family_results,
    _print_per_family_results_multiseed,
    _print_results_multiseed,
    _print_comparison,
    _print_comparison_multiseed,
)

# ---------------------------------------------------------------------------
# Baseline agent classes
# ---------------------------------------------------------------------------


class _Baseline:
    """Abstract baseline compatible with the rl_captcha evaluation harness."""

    device = "cpu"
    name: str = "baseline"

    def reset_hidden(self) -> None:
        pass

    def update_after_step(self, step_info: dict) -> None:
        """Called after each env.step(); lets stateful baselines see outcomes."""

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple:
        raise NotImplementedError

    @staticmethod
    def _valid(mask):
        if mask is None:
            return list(range(7))
        return [i for i, v in enumerate(mask) if v > 0]


class RandomPolicy(_Baseline):
    name = "random"

    def select_action(self, obs, action_mask=None, deterministic=False):
        return _random.choice(self._valid(action_mask)), None, None


class AlwaysBlock(_Baseline):
    name = "always_block"

    def select_action(self, obs, action_mask=None, deterministic=True):
        valid = self._valid(action_mask)
        return (6 if 6 in valid else 0), None, None


class AlwaysAllow(_Baseline):
    name = "always_allow"

    def select_action(self, obs, action_mask=None, deterministic=True):
        valid = self._valid(action_mask)
        return (5 if 5 in valid else 0), None, None


class AlwaysEasyPuzzle(_Baseline):
    name = "always_easy_puzzle"

    def select_action(self, obs, action_mask=None, deterministic=True):
        valid = self._valid(action_mask)
        return (2 if 2 in valid else 0), None, None


class AlwaysHardPuzzle(_Baseline):
    name = "always_hard_puzzle"

    def select_action(self, obs, action_mask=None, deterministic=True):
        valid = self._valid(action_mask)
        return (4 if 4 in valid else 0), None, None


class HoneypotThenBlock(_Baseline):
    """Deploy up to max_hp honeypots then unconditionally block."""

    name = "honeypot_block"

    def __init__(self, max_hp: int = 2):
        self._max_hp = max_hp
        self._hp_deployed = 0

    def reset_hidden(self):
        self._hp_deployed = 0

    def select_action(self, obs, action_mask=None, deterministic=True):
        valid = self._valid(action_mask)
        if 6 in valid:
            return 6, None, None
        if 1 in valid and self._hp_deployed < self._max_hp:
            self._hp_deployed += 1
            return 1, None, None
        return 0, None, None


class HoneypotThenDecide(_Baseline):
    """Deploy honeypots; block if any triggered, otherwise allow.

    Uses update_after_step() to receive trigger feedback from the environment.
    """

    name = "honeypot_decide"

    def __init__(self, max_hp: int = 2):
        self._max_hp = max_hp
        self._hp_deployed = 0
        self._any_triggered = False

    def reset_hidden(self):
        self._hp_deployed = 0
        self._any_triggered = False

    def update_after_step(self, step_info: dict):
        if step_info.get("outcome") in (
            "honeypot_bot_triggered",
            "honeypot_human_triggered",
        ):
            self._any_triggered = True

    def select_action(self, obs, action_mask=None, deterministic=True):
        valid = self._valid(action_mask)
        if 5 in valid or 6 in valid:
            return (6 if self._any_triggered else 5), None, None
        if 1 in valid and self._hp_deployed < self._max_hp:
            self._hp_deployed += 1
            return 1, None, None
        return 0, None, None


# ---------------------------------------------------------------------------
# Evaluation harness — same as evaluate_ppo but calls update_after_step
# ---------------------------------------------------------------------------


def _run_baseline_eval(
    env: EventEnv,
    agent: _Baseline,
    num_episodes: int,
    agent_name: str = "",
    eval_seed: int = 42,
) -> dict:
    _random.seed(eval_seed)
    episode_data = []
    t_start = time.time()

    for ep in range(num_episodes):
        if (ep + 1) % 10 == 0 or ep == 0:
            elapsed = time.time() - t_start
            eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else 0
            eta = (num_episodes - ep - 1) / eps_per_sec if eps_per_sec > 0 else 0
            sys.stdout.write(
                f"\r  [{agent_name}] Episode {ep+1}/{num_episodes} "
                f"({eps_per_sec:.1f} ep/s, ETA {eta:.0f}s)"
            )
            sys.stdout.flush()

        obs, info = env.reset()
        agent.reset_hidden()
        while info.get("too_short"):
            obs, info = env.reset()
            agent.reset_hidden()

        true_label = info["true_label"]
        bot_type = info.get("bot_type")
        total_reward = 0.0
        steps = 0
        actions_taken = []
        honeypots_deployed = 0
        action_mask = info.get("action_mask")
        done = False

        while not done:
            action, _, _ = agent.select_action(
                obs, action_mask=action_mask, deterministic=True
            )
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            agent.update_after_step(step_info)

            total_reward += reward
            steps += 1
            actions_taken.append(action)
            if step_info.get("outcome") in (
                "honeypot_bot_triggered",
                "honeypot_human_triggered",
                "honeypot_no_trigger",
            ):
                honeypots_deployed += 1
            action_mask = step_info.get("action_mask")
            info = step_info

        episode_data.append(
            {
                "true_label": true_label,
                "bot_type": bot_type,
                "outcome": info.get("outcome", "unknown"),
                "reward": total_reward,
                "steps": steps,
                "actions": actions_taken,
                "honeypots_deployed": honeypots_deployed,
                "final_action": actions_taken[-1] if actions_taken else -1,
            }
        )

    elapsed = time.time() - t_start
    sys.stdout.write(
        f"\r  [{agent_name}] Done: {num_episodes} episodes in {elapsed:.1f}s "
        f"({num_episodes/elapsed:.1f} ep/s)      \n"
    )
    sys.stdout.flush()
    return {"episodes": episode_data}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate rule-based baselines")
    p.add_argument("--data-dir", default="data/")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Eval RNG seeds — multiple seeds give mean +/- std",
    )
    p.add_argument(
        "--reward-preset",
        default="v2",
        choices=list(REWARD_PRESETS.keys()),
        help="Reward environment for baseline evaluation",
    )
    p.add_argument(
        "--held-out-families",
        type=str,
        nargs="*",
        default=None,
    )
    p.add_argument(
        "--held-out-tiers",
        type=int,
        nargs="*",
        default=None,
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading sessions from {args.data_dir}...")
    sessions = load_from_directory(args.data_dir, include_augmented=False)
    print(f"  Loaded {len(sessions)} sessions")

    if args.held_out_families or args.held_out_tiers:
        _, _, test_s = split_sessions_by_family(
            sessions,
            held_out_families=args.held_out_families,
            held_out_tiers=args.held_out_tiers,
            train=0.70,
            val=0.15,
            test=0.15,
            seed=args.split_seed,
        )
    else:
        _, _, test_s = split_sessions(
            sessions, train=0.70, val=0.15, test=0.15, seed=args.split_seed
        )

    if args.split == "all":
        eval_sessions = sessions
    else:
        splits_map = {"test": test_s}
        eval_sessions = splits_map.get(args.split, test_s)

    h = sum(1 for s in eval_sessions if s.label == 1)
    b = sum(1 for s in eval_sessions if s.label == 0)
    print(
        f"  Eval split ({args.split}): {len(eval_sessions)} sessions ({h} human, {b} bot)"
    )

    preset_cfg = REWARD_PRESETS[args.reward_preset]
    eval_cfg = replace(preset_cfg, augment=False)
    print(f"  Reward preset: {args.reward_preset}")

    env = EventEnv(eval_sessions, config=eval_cfg)

    eval_seeds = args.eval_seeds
    multi_seed = len(eval_seeds) > 1

    baselines = [
        RandomPolicy(),
        AlwaysBlock(),
        AlwaysAllow(),
        AlwaysEasyPuzzle(),
        AlwaysHardPuzzle(),
        HoneypotThenBlock(),
        HoneypotThenDecide(),
    ]

    print(f"\n  Baselines: {[b.name for b in baselines]}")
    print(f"  Episodes: {args.episodes} per seed, {len(eval_seeds)} seed(s)")
    print()

    all_results = {}
    all_multi_seed_metrics = {}

    for agent in baselines:
        print(f"{'=' * 60}")
        print(f"  Baseline: {agent.name}")
        print()

        if multi_seed:
            seed_metrics = []
            seed_episodes = []
            for seed in eval_seeds:
                results = _run_baseline_eval(
                    env,
                    agent,
                    args.episodes,
                    agent_name=f"{agent.name}/seed={seed}",
                    eval_seed=seed,
                )
                seed_episodes.append(results["episodes"])
                seed_metrics.append(_compute_metrics(results["episodes"]))

            combined = [e for eps in seed_episodes for e in eps]
            all_results[agent.name] = {
                "episodes": combined,
                "seed_episodes": seed_episodes,
            }
            all_multi_seed_metrics[agent.name] = seed_metrics

            _print_results(
                {"episodes": combined}, agent_name=agent.name, split_name=args.split
            )
            _print_per_family_results({"episodes": combined}, agent_name=agent.name)
            _print_results_multiseed(
                seed_metrics,
                eval_seeds,
                agent_name=agent.name,
                split_name=args.split,
                episodes_per_seed=args.episodes,
            )
            _print_per_family_results_multiseed(
                seed_episodes, eval_seeds, agent_name=agent.name
            )
        else:
            results = _run_baseline_eval(
                env,
                agent,
                args.episodes,
                agent_name=agent.name,
                eval_seed=eval_seeds[0],
            )
            all_results[agent.name] = results
            _print_results(results, agent_name=agent.name, split_name=args.split)
            _print_per_family_results(results, agent_name=agent.name)

    # Comparison table
    if multi_seed and all_multi_seed_metrics:
        _print_comparison_multiseed(
            all_multi_seed_metrics, eval_seeds, split_name=args.split
        )
    else:
        _print_comparison(all_results, split_name=args.split)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
