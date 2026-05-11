"""Evaluate one or more trained PPO/DG/Soft-PPO+LSTM agents on the event-level CAPTCHA environment.

Usage (single agent):
    python -m rl_captcha.scripts.evaluate_ppo \
        --agent rl_captcha/agent/checkpoints/ppo_noaug \
        --data-dir data/ \
        --episodes 500

Usage (all six — with and without adversarial augmentation):
    python -m rl_captcha.scripts.evaluate_ppo \
        --agent ppo_noaug=rl_captcha/agent/checkpoints/ppo_noaug \
               ppo_advaug=rl_captcha/agent/checkpoints/ppo_advaug \
               dg_noaug=rl_captcha/agent/checkpoints/dg_noaug \
               dg_advaug=rl_captcha/agent/checkpoints/dg_advaug \
               soft_ppo_noaug=rl_captcha/agent/checkpoints/soft_ppo_noaug \
               soft_ppo_advaug=rl_captcha/agent/checkpoints/soft_ppo_advaug \
        --episodes 500

By default, evaluation uses the held-out TEST split (15% of data) with
the same seed used during training, ensuring no overlap with training data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from rl_captcha.config import Config, REWARD_PRESETS
from rl_captcha.data.loader import (
    load_from_directory,
    split_sessions,
    split_sessions_by_family,
    split_sessions_by_person,
    bot_type_to_tier,
    TIER_NAMES,
)
from rl_captcha.environment.event_env import EventEnv, ACTION_NAMES
from rl_captcha.agent.ppo_lstm import PPOLSTM
from rl_captcha.agent.dg_lstm import DGLSTM, DGConfig
from rl_captcha.agent.soft_ppo_lstm import SoftPPOLSTM, SoftPPOConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO/DG/Soft-PPO+LSTM agents")
    p.add_argument(
        "--agent",
        type=str,
        nargs="+",
        required=True,
        help="Agent checkpoint(s). Either plain paths or name=path pairs "
        "(e.g. ppo=checkpoints/ppo_run1 dg=checkpoints/dg_run1)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Path to data directory with human/ and bot/ subdirs",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes to evaluate per agent",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train", "all"],
        help="Which data split to evaluate on (default: test)",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for split (must match training)",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run evaluation with multiple RNG seeds and report "
        "mean +/- std (e.g. --eval-seeds 42 123 456 789 1024)",
    )
    p.add_argument(
        "--include-augmented",
        action="store_true",
        help="Include adversarially augmented bot sessions in evaluation",
    )
    p.add_argument(
        "--challenge-as-fp",
        action="store_true",
        help="Strict mode: count human_passed_puzzle as false positive. "
        "Any unnecessary challenge to a human is a UX failure.",
    )
    p.add_argument(
        "--held-out-families",
        type=str,
        nargs="*",
        default=None,
        help="Bot families to hold out from train/val (test-only)",
    )
    p.add_argument(
        "--held-out-tiers",
        type=int,
        nargs="*",
        default=None,
        help="Bot tiers to hold out from train/val (test-only)",
    )
    p.add_argument(
        "--person-a-dir",
        type=str,
        default=None,
        help="Directory whose filenames identify Person A's sessions in data/human/",
    )
    p.add_argument(
        "--person-b-dir",
        type=str,
        default=None,
        help="Directory whose filenames identify Person B's sessions in data/human/",
    )
    p.add_argument(
        "--held-out-person",
        type=str,
        choices=["A", "B"],
        default=None,
        help="Evaluate only on this person's sessions (must match training's held-out person)",
    )
    p.add_argument(
        "--reward-preset",
        type=str,
        default="v2",
        choices=list(REWARD_PRESETS.keys()),
        help="Reward environment preset to evaluate in: v1 or v2 (default: v2)",
    )
    # Architecture overrides — must match the training config for ablation models
    p.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=None,
        help="Override LSTM hidden size when loading ablation checkpoints",
    )
    p.add_argument(
        "--lstm-num-layers",
        type=int,
        default=None,
        help="Override LSTM number of layers when loading ablation checkpoints",
    )
    # Environment overrides for ablation eval (must match training env)
    p.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Override EventEnvConfig.max_windows (e.g. 1 for single_view ablation)",
    )
    p.add_argument(
        "--random-window-subsample",
        action="store_true",
        default=False,
        help="Override EventEnvConfig.random_window_subsample=True (for single_view ablation)",
    )
    p.add_argument(
        "--log-per-session",
        type=str,
        default=None,
        metavar="PATH",
        help="Write per-episode CSV (agent, true_label, outcome, challenged, steps, bot_type) to PATH",
    )
    return p.parse_args()


def _parse_agent_specs(agent_args: list[str]) -> list[tuple[str, str]]:
    """Parse agent arguments into (name, path) pairs.

    Supports:
        --agent path/to/checkpoint              → name inferred from dir name
        --agent ppo=path/to/ppo dg=path/to/dg   → explicit names
    """
    specs = []
    for arg in agent_args:
        if "=" in arg:
            name, path = arg.split("=", 1)
            specs.append((name.strip(), path.strip()))
        else:
            # Infer name from last directory component
            from pathlib import Path

            name = Path(arg).name
            specs.append((name, arg))
    return specs


def _create_agent(name: str, cfg: Config, device: str) -> PPOLSTM:
    """Instantiate the correct agent class based on the agent name.

    Handles old-style names (ppo, dg, soft_ppo), augmentation-aware names
    (ppo_noaug, ppo_advaug, ...), and ablation names
    (ppo_advaug_v2_ablation_small_lstm_seed42, ...).
    """
    kwargs = dict(obs_dim=cfg.event_env.event_dim, action_dim=7, device=device)
    n = name.lower()
    # Detect algorithm by prefix to handle all naming conventions
    if n.startswith("dg"):
        return DGLSTM(
            config=DGConfig(
                **{
                    k: getattr(cfg.ppo, k)
                    for k in DGConfig.__dataclass_fields__
                    if k in cfg.ppo.__dataclass_fields__
                }
            ),
            **kwargs,
        )
    elif n.startswith("soft_ppo"):
        return SoftPPOLSTM(
            config=SoftPPOConfig(
                **{
                    k: getattr(cfg.ppo, k)
                    for k in SoftPPOConfig.__dataclass_fields__
                    if k in cfg.ppo.__dataclass_fields__
                }
            ),
            **kwargs,
        )
    else:
        return PPOLSTM(config=cfg.ppo, **kwargs)


_CHALLENGE_AS_FP: bool = (
    False  # set from args in main(), read by _compute_metrics callers
)


def main():
    global _CHALLENGE_AS_FP
    args = parse_args()
    _CHALLENGE_AS_FP = bool(args.challenge_as_fp)
    if _CHALLENGE_AS_FP:
        print("  [strict mode] human_passed_puzzle counted as false positive")
    cfg = Config()

    # Load data
    print(f"Loading sessions from {args.data_dir}...")
    sessions = load_from_directory(
        args.data_dir, include_augmented=args.include_augmented
    )
    human_count = sum(1 for s in sessions if s.label == 1)
    bot_count = sum(1 for s in sessions if s.label == 0)
    print(f"  Loaded {len(sessions)} sessions ({human_count} human, {bot_count} bot)")

    if not sessions:
        print("ERROR: No sessions found.")
        return

    # Select split
    if args.split == "all":
        eval_sessions = sessions
        print(f"  Evaluating on ALL {len(eval_sessions)} sessions")
    else:
        if args.held_out_person:
            person_dir = (
                args.person_a_dir if args.held_out_person == "A" else args.person_b_dir
            )
            if not person_dir:
                side = "a" if args.held_out_person == "A" else "b"
                print(
                    f"ERROR: --held-out-person {args.held_out_person} requires --person-{side}-dir"
                )
                return
            from pathlib import Path as _Path

            held_filenames = {p.name for p in _Path(person_dir).glob("*.json")}
            print(
                f"  Held-out person {args.held_out_person}: {len(held_filenames)} session files"
            )
            train_s, val_s, test_s = split_sessions_by_person(
                sessions,
                held_out_filenames=held_filenames,
                train=0.70,
                val=0.15,
                test=0.15,
                seed=args.split_seed,
            )
            # For person eval, restrict test to ONLY the held-out person's human sessions
            if args.split == "test":
                from pathlib import Path as _Path2

                test_s = [
                    s
                    for s in test_s
                    if s.label == 1
                    and _Path2(s.metadata.get("source_file", "")).name in held_filenames
                ]
        elif args.held_out_families or args.held_out_tiers:
            print(f"  Held-out families: {args.held_out_families or '(none)'}")
            print(f"  Held-out tiers:    {args.held_out_tiers or '(none)'}")
            train_s, val_s, test_s = split_sessions_by_family(
                sessions,
                held_out_families=args.held_out_families,
                held_out_tiers=args.held_out_tiers,
                train=0.70,
                val=0.15,
                test=0.15,
                seed=args.split_seed,
            )
        else:
            train_s, val_s, test_s = split_sessions(
                sessions,
                train=0.70,
                val=0.15,
                test=0.15,
                seed=args.split_seed,
            )
        splits = {"train": train_s, "val": val_s, "test": test_s}
        eval_sessions = splits[args.split]
        h = sum(1 for s in eval_sessions if s.label == 1)
        b = sum(1 for s in eval_sessions if s.label == 0)
        print(
            f"  Evaluating on {args.split.upper()} split: "
            f"{len(eval_sessions)} sessions ({h} human, {b} bot)"
        )

    # Create environment using the requested reward preset
    from dataclasses import replace

    preset_cfg = REWARD_PRESETS[args.reward_preset]
    # Apply eval-time env overrides (must match training config for ablations)
    if args.max_windows is not None:
        preset_cfg = replace(preset_cfg, max_windows=args.max_windows)
        print(f"  Override: max_windows={args.max_windows}")
    if args.random_window_subsample:
        preset_cfg = replace(preset_cfg, random_window_subsample=True)
        print("  Override: random_window_subsample=True")
    eval_cfg = replace(preset_cfg, augment=False)
    print(f"  Reward preset: {args.reward_preset}")
    env = EventEnv(eval_sessions, config=eval_cfg)

    # Apply LSTM architecture overrides (must match training config for ablations)
    if args.lstm_hidden_size is not None:
        cfg.ppo.lstm_hidden_size = args.lstm_hidden_size
        print(f"  Override: lstm_hidden_size={args.lstm_hidden_size}")
    if args.lstm_num_layers is not None:
        cfg.ppo.lstm_num_layers = args.lstm_num_layers
        print(f"  Override: lstm_num_layers={args.lstm_num_layers}")

    # Parse agent specs
    agent_specs = _parse_agent_specs(args.agent)
    eval_seeds = args.eval_seeds or [42]
    multi_seed = len(eval_seeds) > 1

    print(
        f"\n  Evaluating {len(agent_specs)} agent(s): "
        f"{', '.join(name for name, _ in agent_specs)}"
    )
    print(f"  Episodes per agent: {args.episodes}")
    if multi_seed:
        print(f"  Eval seeds: {eval_seeds} ({len(eval_seeds)} runs per agent)")
    print()

    # Evaluate each agent
    all_results = {}
    all_multi_seed_metrics = {}  # name -> list of metric dicts (one per seed)
    for name, path in agent_specs:
        print(f"{'=' * 60}")
        print(f"  Loading agent: {name} ({path})")
        agent = _create_agent(name, cfg, args.device)
        agent.load(path)
        print(f"  Device: {agent.device}")
        print()

        if multi_seed:
            seed_metrics = []
            seed_episodes = []
            for seed in eval_seeds:
                results = _run_evaluation(
                    env,
                    agent,
                    args.episodes,
                    agent_name=f"{name}/seed={seed}",
                    eval_seed=seed,
                )
                seed_episodes.append(results["episodes"])
                seed_metrics.append(
                    _compute_metrics(results["episodes"], _CHALLENGE_AS_FP)
                )

            combined_episodes = [e for eps in seed_episodes for e in eps]
            all_results[name] = {
                "episodes": combined_episodes,
                "seed_episodes": seed_episodes,
            }
            all_multi_seed_metrics[name] = seed_metrics

            # Pooled aggregate (all episodes as one)
            _print_results(
                {"episodes": combined_episodes}, agent_name=name, split_name=args.split
            )
            _print_per_family_results({"episodes": combined_episodes}, agent_name=name)

            # Mean +/- std across seeds
            _print_results_multiseed(
                seed_metrics,
                eval_seeds,
                agent_name=name,
                split_name=args.split,
                episodes_per_seed=args.episodes,
            )
            _print_per_family_results_multiseed(
                seed_episodes, eval_seeds, agent_name=name
            )
        else:
            results = _run_evaluation(
                env, agent, args.episodes, agent_name=name, eval_seed=eval_seeds[0]
            )
            all_results[name] = results
            _print_results(results, agent_name=name, split_name=args.split)
            _print_per_family_results(results, agent_name=name)

    # Print comparison table if multiple agents
    if len(all_results) > 1:
        if multi_seed and all_multi_seed_metrics:
            _print_comparison_multiseed(
                all_multi_seed_metrics, eval_seeds, split_name=args.split
            )
        else:
            _print_comparison(all_results, split_name=args.split)

    if args.log_per_session:
        import csv as _csv
        from pathlib import Path as _Path

        _CHALLENGE_OUTCOMES = {
            "false_positive_block",
            "fp_puzzle",
            "human_passed_puzzle",
        }
        _out = _Path(args.log_per_session)
        _out.parent.mkdir(parents=True, exist_ok=True)
        with open(_out, "w", newline="") as _f:
            _w = _csv.DictWriter(
                _f,
                fieldnames=[
                    "agent",
                    "true_label",
                    "outcome",
                    "challenged",
                    "steps",
                    "bot_type",
                ],
            )
            _w.writeheader()
            for _name in all_results:
                for _e in all_results[_name]["episodes"]:
                    _w.writerow(
                        {
                            "agent": _name,
                            "true_label": _e["true_label"],
                            "outcome": _e["outcome"],
                            "challenged": _e["outcome"] in _CHALLENGE_OUTCOMES,
                            "steps": _e["steps"],
                            "bot_type": _e.get("bot_type") or "",
                        }
                    )
        print(f"  Per-session log -> {_out}")

    print("\nEvaluation complete.")


def _run_evaluation(
    env: EventEnv,
    agent: PPOLSTM,
    num_episodes: int,
    agent_name: str = "",
    eval_seed: int = 42,
) -> dict:
    """Run deterministic evaluation episodes.

    Seeds the RNG before each run so every agent sees the exact same
    sequence of sessions, making results reproducible and comparable.
    """
    import random as _random
    import sys
    import time

    _random.seed(eval_seed)
    episode_data = []
    t_start = time.time()

    for ep in range(num_episodes):
        if (ep + 1) % 10 == 0 or ep == 0:
            elapsed = time.time() - t_start
            eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else 0
            eta = (num_episodes - ep - 1) / eps_per_sec if eps_per_sec > 0 else 0
            sys.stdout.write(
                f"\r  [{agent_name}] Episode {ep + 1}/{num_episodes} "
                f"({eps_per_sec:.1f} ep/s, ETA {eta:.0f}s)"
            )
            sys.stdout.flush()

        obs, info = env.reset()
        agent.reset_hidden()

        # Skip too-short sessions
        while info.get("too_short"):
            obs, info = env.reset()
            agent.reset_hidden()

        true_label = info["true_label"]
        bot_type = info.get("bot_type")
        total_reward = 0.0
        steps = 0
        actions_taken = []
        honeypots_deployed = 0  # actual deployments (not maxed-out attempts)

        action_mask = info.get("action_mask")
        done = False
        while not done:
            action, _, _ = agent.select_action(
                obs, action_mask=action_mask, deterministic=True
            )
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            actions_taken.append(action)
            step_outcome = step_info.get("outcome", "")
            if step_outcome in ("honeypot_triggered", "honeypot_no_trigger"):
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
        f"({num_episodes / elapsed:.1f} ep/s)      \n"
    )
    sys.stdout.flush()
    return {"episodes": episode_data}


def _compute_metrics(episodes: list[dict], challenge_as_fp: bool = False) -> dict:
    """Compute evaluation metrics from episode data.

    challenge_as_fp: if True, human_passed_puzzle counts as FP (strict mode).
    A human who was challenged unnecessarily is an incorrect outcome even if
    they ultimately passed the puzzle.
    """
    n = len(episodes)
    rewards = [e["reward"] for e in episodes]
    lengths = [e["steps"] for e in episodes]

    tp = sum(
        1
        for e in episodes
        if e["true_label"] == 0
        and e["outcome"] in ("correct_block", "bot_blocked_puzzle")
    )
    _tn_outcomes = (
        ("correct_allow",)
        if challenge_as_fp
        else ("correct_allow", "human_passed_puzzle")
    )
    tn = sum(
        1 for e in episodes if e["true_label"] == 1 and e["outcome"] in _tn_outcomes
    )
    _fp_outcomes = (
        ("false_positive_block", "fp_puzzle", "human_passed_puzzle")
        if challenge_as_fp
        else ("false_positive_block", "fp_puzzle")
    )
    fp = sum(
        1 for e in episodes if e["true_label"] == 1 and e["outcome"] in _fp_outcomes
    )
    fn = sum(
        1
        for e in episodes
        if e["true_label"] == 0
        and e["outcome"] in ("false_negative", "bot_passed_puzzle")
    )
    truncated = sum(1 for e in episodes if e["outcome"] == "truncated")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / n if n > 0 else 0.0

    # Honeypot usage — count actual deployments, not attempted actions
    honeypot_counts = [
        e.get("honeypots_deployed", sum(1 for a in e["actions"] if a == 1))
        for e in episodes
    ]
    episodes_with_honeypot = sum(1 for c in honeypot_counts if c > 0)
    avg_honeypots = float(np.mean(honeypot_counts)) if honeypot_counts else 0.0

    # Action distribution across ALL steps (not just final)
    all_actions = [a for e in episodes for a in e["actions"]]
    action_dist = defaultdict(int)
    for a in all_actions:
        action_dist[a] += 1
    total_actions = len(all_actions) if all_actions else 1

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_length": float(np.mean(lengths)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "truncated": truncated,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "honeypot_rate": episodes_with_honeypot / n if n > 0 else 0.0,
        "avg_honeypots_per_ep": avg_honeypots,
        "total_honeypots": sum(honeypot_counts),
        "action_dist": {a: c / total_actions for a, c in action_dist.items()},
    }


def _print_results(results: dict, agent_name: str = "agent", split_name: str = "test"):
    """Print evaluation summary for one agent."""
    episodes = results["episodes"]
    n = len(episodes)
    m = _compute_metrics(episodes, _CHALLENGE_AS_FP)

    print(f"=== {agent_name.upper()} - {split_name.upper()} split ({n} episodes) ===")
    print(f"  Avg reward:  {m['avg_reward']:.3f} +/- {m['std_reward']:.3f}")
    print(f"  Avg length:  {m['avg_length']:.1f}")
    print()

    # Confusion matrix
    print("--- Confusion Matrix ---")
    print(f"  True Positives  (bot blocked):   {m['tp']:4d} ({100*m['tp']/n:.1f}%)")
    print(f"  True Negatives  (human allowed): {m['tn']:4d} ({100*m['tn']/n:.1f}%)")
    print(f"  False Positives (human blocked): {m['fp']:4d} ({100*m['fp']/n:.1f}%)")
    print(f"  False Negatives (bot allowed):   {m['fn']:4d} ({100*m['fn']/n:.1f}%)")
    print(
        f"  Truncated (indecisive):          {m['truncated']:4d} ({100*m['truncated']/n:.1f}%)"
    )
    other = n - m["tp"] - m["tn"] - m["fp"] - m["fn"] - m["truncated"]
    if other > 0:
        print(f"  Other:                           {other:4d} ({100*other/n:.1f}%)")
    print()

    print(f"  Accuracy:  {m['accuracy']:.3f}")
    print(f"  Precision: {m['precision']:.3f}")
    print(f"  Recall:    {m['recall']:.3f}")
    print(f"  F1:        {m['f1']:.3f}")
    print()

    # Outcome distribution
    outcome_counts = defaultdict(int)
    for e in episodes:
        outcome_counts[e["outcome"]] += 1

    print("--- Outcome Distribution ---")
    for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {outcome:30s} {count:4d} ({100*count/n:.1f}%)")
    print()

    # Action distribution (final actions)
    action_counts = defaultdict(int)
    for e in episodes:
        fa = e["final_action"]
        if 0 <= fa < len(ACTION_NAMES):
            action_counts[ACTION_NAMES[fa]] += 1
        else:
            action_counts[f"action_{fa}"] += 1

    print("--- Final Action Distribution ---")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action:20s} {count:4d} ({100*count/n:.1f}%)")
    print()

    # Honeypot usage
    honeypot_counts = [
        e.get("honeypots_deployed", sum(1 for a in e["actions"] if a == 1))
        for e in episodes
    ]
    eps_with_hp = sum(1 for c in honeypot_counts if c > 0)
    avg_hp = float(np.mean(honeypot_counts)) if honeypot_counts else 0.0
    total_hp = sum(honeypot_counts)

    print("--- Honeypot Usage ---")
    print(f"  Episodes using honeypot: {eps_with_hp}/{n} ({100*eps_with_hp/n:.1f}%)")
    print(f"  Avg honeypots per episode: {avg_hp:.2f}")
    print(f"  Total honeypot deployments: {total_hp}")

    # Honeypot usage by label
    human_eps = [e for e in episodes if e["true_label"] == 1]
    bot_eps = [e for e in episodes if e["true_label"] == 0]
    if human_eps:
        hp_human = [sum(1 for a in e["actions"] if a == 1) for e in human_eps]
        print(
            f"  Honeypot on humans: {sum(1 for c in hp_human if c > 0)}/{len(human_eps)} "
            f"({100*sum(1 for c in hp_human if c > 0)/len(human_eps):.1f}%), "
            f"avg {np.mean(hp_human):.2f}/ep"
        )
    if bot_eps:
        hp_bot = [sum(1 for a in e["actions"] if a == 1) for e in bot_eps]
        print(
            f"  Honeypot on bots:   {sum(1 for c in hp_bot if c > 0)}/{len(bot_eps)} "
            f"({100*sum(1 for c in hp_bot if c > 0)/len(bot_eps):.1f}%), "
            f"avg {np.mean(hp_bot):.2f}/ep"
        )
    print()

    # All-step action distribution (not just final)
    all_actions = [a for e in episodes for a in e["actions"]]
    action_step_counts = defaultdict(int)
    for a in all_actions:
        action_step_counts[a] += 1
    total_steps = len(all_actions) if all_actions else 1

    print("--- All-Step Action Distribution ---")
    for idx in sorted(action_step_counts.keys()):
        aname = ACTION_NAMES[idx] if 0 <= idx < len(ACTION_NAMES) else f"action_{idx}"
        count = action_step_counts[idx]
        print(f"  {aname:20s} {count:5d} ({100*count/total_steps:.1f}%)")
    print()

    # Decision timing by label
    if human_eps:
        avg_human_steps = np.mean([e["steps"] for e in human_eps])
        print(f"  Avg steps (human sessions): {avg_human_steps:.1f}")
    if bot_eps:
        avg_bot_steps = np.mean([e["steps"] for e in bot_eps])
        print(f"  Avg steps (bot sessions):   {avg_bot_steps:.1f}")
    print()


def _print_results_multiseed(
    seed_metrics: list[dict],
    seeds: list[int],
    agent_name: str = "agent",
    split_name: str = "test",
    episodes_per_seed: int = 500,
):
    """Print mean +/- std evaluation summary across multiple seeds."""
    n_seeds = len(seeds)
    total_eps = n_seeds * episodes_per_seed

    print(
        f"=== {agent_name.upper()} - {split_name.upper()} split "
        f"({total_eps} episodes, {n_seeds} seeds) ==="
    )
    print()

    rows = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Avg Reward", "avg_reward"),
        ("Avg Length", "avg_length"),
        ("Honeypot %", "honeypot_rate"),
        ("Avg HP/ep", "avg_honeypots_per_ep"),
    ]

    for label, key in rows:
        values = [m[key] for m in seed_metrics]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {label:<14s} {mean:.3f} +/- {std:.3f}")
    print()

    # Confusion matrix (mean +/- std)
    print("--- Confusion Matrix (mean +/- std across seeds) ---")
    for label, key in [
        ("True Positives  (bot blocked)", "tp"),
        ("True Negatives  (human allowed)", "tn"),
        ("False Positives (human blocked)", "fp"),
        ("False Negatives (bot allowed)", "fn"),
        ("Truncated (indecisive)", "truncated"),
    ]:
        values = [m[key] for m in seed_metrics]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {label}:  {mean:.1f} +/- {std:.1f}")
    print()

    # Per-seed breakdown
    print("--- Per-Seed Breakdown ---")
    print(f"  {'Seed':<10s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
    print(f"  {'-' * 42}")
    for i, seed in enumerate(seeds):
        m = seed_metrics[i]
        print(
            f"  {seed:<10d} {m['accuracy']:8.3f} {m['precision']:8.3f} "
            f"{m['recall']:8.3f} {m['f1']:8.3f}"
        )
    print()


def _print_per_family_results_multiseed(
    seed_episodes: list[list[dict]],
    seeds: list[int],
    agent_name: str = "agent",
):
    """Print per-family and per-tier detection rates as mean +/- std across seeds."""
    detected_outcomes = {"correct_block", "bot_blocked_puzzle"}

    # Compute per-family detection rate for each seed
    all_families: set[str] = set()
    seed_family_rates: list[dict[str, float]] = []
    seed_family_counts: list[dict[str, int]] = []
    for episodes in seed_episodes:
        bot_eps = [e for e in episodes if e["true_label"] == 0]
        by_family: dict[str, list[dict]] = defaultdict(list)
        for e in bot_eps:
            family = e.get("bot_type") or "unknown"
            by_family[family].append(e)
            all_families.add(family)
        rates = {}
        counts = {}
        for family, eps in by_family.items():
            n = len(eps)
            detected = sum(1 for e in eps if e["outcome"] in detected_outcomes)
            rates[family] = detected / n if n > 0 else 0.0
            counts[family] = n
        seed_family_rates.append(rates)
        seed_family_counts.append(counts)

    print(
        f"--- Per-Family Bot Detection ({agent_name}, mean +/- std, {len(seeds)} seeds) ---"
    )
    print(f"  {'Family':<18s} {'Tier':>4s} {'N/seed':>7s} {'Rate':>14s}")
    print(f"  {'-' * 45}")

    for family in sorted(all_families):
        rates = [r.get(family, 0.0) for r in seed_family_rates]
        counts = [c.get(family, 0) for c in seed_family_counts]
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        avg_n = int(np.mean(counts))
        tier = bot_type_to_tier(family if family != "unknown" else None)
        tier_str = str(tier) if tier > 0 else "?"
        print(
            f"  {family:<18s} {tier_str:>4s} {avg_n:>7d} "
            f"{mean_rate:>7.1%} +/- {std_rate:.1%}"
        )
    print()

    # Per-tier aggregation
    all_tiers: set[int] = set()
    seed_tier_rates: list[dict[int, float]] = []
    seed_tier_counts: list[dict[int, int]] = []
    for episodes in seed_episodes:
        bot_eps = [e for e in episodes if e["true_label"] == 0]
        by_tier: dict[int, list[dict]] = defaultdict(list)
        for e in bot_eps:
            tier = bot_type_to_tier(e.get("bot_type"))
            by_tier[tier].append(e)
            all_tiers.add(tier)
        rates = {}
        counts = {}
        for tier_num, eps in by_tier.items():
            n = len(eps)
            detected = sum(1 for e in eps if e["outcome"] in detected_outcomes)
            rates[tier_num] = detected / n if n > 0 else 0.0
            counts[tier_num] = n
        seed_tier_rates.append(rates)
        seed_tier_counts.append(counts)

    print(f"--- Per-Tier Summary ({agent_name}, mean +/- std, {len(seeds)} seeds) ---")
    for tier_num in sorted(all_tiers):
        rates = [r.get(tier_num, 0.0) for r in seed_tier_rates]
        counts = [c.get(tier_num, 0) for c in seed_tier_counts]
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        avg_n = int(np.mean(counts))
        tier_name = TIER_NAMES.get(tier_num, "Unknown")
        print(
            f"  Tier {tier_num} ({tier_name}): ~{avg_n} bots, "
            f"{mean_rate:.1%} +/- {std_rate:.1%}"
        )
    print()


def _print_per_family_results(results: dict, agent_name: str = "agent"):
    """Print per-bot-family and per-tier detection rate breakdowns."""
    episodes = results["episodes"]
    bot_eps = [e for e in episodes if e["true_label"] == 0]

    if not bot_eps:
        return

    # --- Per-family breakdown ---
    by_family: dict[str, list[dict]] = defaultdict(list)
    for e in bot_eps:
        family = e.get("bot_type") or "unknown"
        by_family[family].append(e)

    detected_outcomes = {"correct_block", "bot_blocked_puzzle"}

    print(f"--- Per-Family Bot Detection ({agent_name}) ---")
    print(
        f"  {'Family':<18s} {'Tier':>4s} {'N':>5s} {'Detect':>7s} {'Miss':>5s} {'Rate':>7s}"
    )
    print(f"  {'-' * 48}")

    for family in sorted(by_family.keys()):
        eps = by_family[family]
        n = len(eps)
        detected = sum(1 for e in eps if e["outcome"] in detected_outcomes)
        missed = n - detected
        rate = detected / n if n > 0 else 0.0
        tier = bot_type_to_tier(family if family != "unknown" else None)
        tier_str = str(tier) if tier > 0 else "?"
        print(
            f"  {family:<18s} {tier_str:>4s} {n:5d} {detected:7d} {missed:5d} {rate:7.1%}"
        )

    print()

    # --- Per-tier breakdown ---
    by_tier: dict[int, list[dict]] = defaultdict(list)
    for e in bot_eps:
        tier = bot_type_to_tier(e.get("bot_type"))
        by_tier[tier].append(e)

    print(f"--- Per-Tier Summary ({agent_name}) ---")
    for tier_num in sorted(by_tier.keys()):
        eps = by_tier[tier_num]
        n = len(eps)
        detected = sum(1 for e in eps if e["outcome"] in detected_outcomes)
        rate = detected / n if n > 0 else 0.0
        tier_name = TIER_NAMES.get(tier_num, "Unknown")
        print(f"  Tier {tier_num} ({tier_name}): {n:4d} bots, {rate:.1%} detected")
    print()


def _print_comparison(all_results: dict[str, dict], split_name: str = "test"):
    """Print a side-by-side comparison table of all agents."""
    print()
    print("=" * 70)
    print(f"  COMPARISON TABLE - {split_name.upper()} split")
    print("=" * 70)

    metrics = {}
    for name, results in all_results.items():
        metrics[name] = _compute_metrics(results["episodes"], _CHALLENGE_AS_FP)

    # Header
    names = list(metrics.keys())
    col_w = max(12, max(len(n) for n in names) + 2)
    header = f"  {'Metric':<20s}" + "".join(f"{n:>{col_w}s}" for n in names)
    print(header)
    print("  " + "-" * (20 + col_w * len(names)))

    # Rows
    rows = [
        ("Accuracy", "accuracy", ".3f"),
        ("Precision", "precision", ".3f"),
        ("Recall", "recall", ".3f"),
        ("F1", "f1", ".3f"),
        ("Avg Reward", "avg_reward", ".3f"),
        ("Avg Length", "avg_length", ".1f"),
        ("Honeypot %", "honeypot_rate", ".1%"),
        ("Avg HP/ep", "avg_honeypots_per_ep", ".2f"),
        ("TP (bot blocked)", "tp", "d"),
        ("TN (human ok)", "tn", "d"),
        ("FP (human bad)", "fp", "d"),
        ("FN (bot missed)", "fn", "d"),
        ("Truncated", "truncated", "d"),
    ]

    for label, key, fmt in rows:
        row = f"  {label:<20s}"
        for name in names:
            val = metrics[name][key]
            row += f"{val:>{col_w}{fmt}}"
        print(row)

    print()


def _print_comparison_multiseed(
    all_metrics: dict[str, list[dict]],
    seeds: list[int],
    split_name: str = "test",
):
    """Print comparison table averaged over training seeds and eval seeds.

    Groups agent names by their base algorithm (strips trailing _seed{N}) so
    each column represents one model variant with mean +/- std computed over
    all training seeds (each training seed is itself already averaged over eval
    seeds). Never highlights a single best seed.
    """
    import re as _re

    names = list(all_metrics.keys())
    n_eval_seeds = len(seeds)

    rows = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Avg Reward", "avg_reward"),
        ("Honeypot %", "honeypot_rate"),
        ("Avg HP/ep", "avg_honeypots_per_ep"),
    ]

    # Group by base algorithm name (strip trailing _seed{N})
    groups: dict[str, list[str]] = {}
    for name in names:
        base = _re.sub(r"_seed\d+$", "", name)
        groups.setdefault(base, []).append(name)

    group_names = list(groups.keys())
    n_train_seeds = max(len(v) for v in groups.values())

    # For each group compute mean ± std over training seeds
    # Each training seed's value is already averaged over eval seeds
    group_stats: dict[str, dict[str, tuple[float, float]]] = {}
    for base, members in groups.items():
        group_stats[base] = {}
        for _, key in rows:
            per_seed = [
                float(np.mean([m[key] for m in all_metrics[n]])) for n in members
            ]
            group_stats[base][key] = (float(np.mean(per_seed)), float(np.std(per_seed)))

    print()
    print("=" * 80)
    print(
        f"  COMPARISON TABLE — {split_name.upper()} split "
        f"(mean +/- std, {n_train_seeds} training seeds × {n_eval_seeds} eval seeds)"
    )
    print("=" * 80)

    col_w = max(20, max(len(n) for n in group_names) + 4)
    header = f"  {'Metric':<16s}" + "".join(f"{n:>{col_w}s}" for n in group_names)
    print(header)
    print("  " + "-" * (16 + col_w * len(group_names)))

    for label, key in rows:
        row = f"  {label:<16s}"
        for base in group_names:
            mean, std = group_stats[base][key]
            row += f"{f'{mean:.3f} +/- {std:.3f}':>{col_w}s}"
        print(row)

    print()

    # Per-training-seed detail (for transparency, not for selection)
    print("--- Per-Training-Seed Detail ---")
    print(f"  {'Agent':<42s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
    print("  " + "-" * 68)
    for name in names:
        per_eval = all_metrics[name]
        acc = float(np.mean([m["accuracy"] for m in per_eval]))
        prec = float(np.mean([m["precision"] for m in per_eval]))
        rec = float(np.mean([m["recall"] for m in per_eval]))
        f1 = float(np.mean([m["f1"] for m in per_eval]))
        print(f"  {name:<42s} {acc:8.3f} {prec:8.3f} {rec:8.3f} {f1:8.3f}")
    print()


if __name__ == "__main__":
    main()
