"""Sensitivity analysis: evaluate trained agents under varied reward/env parameters.

Tests whether learned policies are robust to misspecified assumptions about:
  - Reward magnitudes (honeypot bonus, block rewards, penalties)
  - Challenge outcomes (puzzle pass rates for humans and bots)
  - Honeypot effectiveness by tier

Each sweep varies ONE parameter while holding the rest at their v2 defaults.

Usage:
    python -m rl_captcha.scripts.sensitivity_analysis \\
        --data-dir data/ \\
        --reward-preset v2 \\
        --sweep honeypot_info_bonus \\
        --agent ppo_advaug_v2_seed42=rl_captcha/agent/checkpoints/ppo_advaug_v2_seed42 \\
                dg_advaug_v2_seed42=rl_captcha/agent/checkpoints/dg_advaug_v2_seed42 \\
                soft_ppo_advaug_v2_seed42=rl_captcha/agent/checkpoints/soft_ppo_advaug_v2_seed42

Available sweeps:
  Reward:    honeypot_info_bonus, reward_direct_block_bot, penalty_block_human,
             penalty_bot_missed_allow
  Challenge: easy_puzzle_bot_pass, hard_puzzle_bot_pass, hard_puzzle_human_pass,
             tier5_honeypot_rate, all_honeypot_rates
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace

import numpy as np

from rl_captcha.config import Config, REWARD_PRESETS, EventEnvConfig
from rl_captcha.data.loader import (
    load_from_directory,
    split_sessions,
    TIER_NAMES,
    bot_type_to_tier,
)
from rl_captcha.environment.event_env import EventEnv
from rl_captcha.scripts.evaluate_ppo import (
    _parse_agent_specs,
    _create_agent,
    _run_evaluation,
    _compute_metrics,
)

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

SWEEPS: dict[str, dict] = {
    # --- Reward assumption sweeps ---
    "honeypot_info_bonus": {
        "label": "Honeypot Info Bonus",
        "group": "reward",
        "values": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        "default_v2": 0.5,
        "description": "Reward given when a deployed honeypot triggers on a bot",
    },
    "reward_direct_block_bot": {
        "label": "Direct Block Reward",
        "group": "reward",
        "values": [0.3, 0.5, 0.7, 1.0, 1.2],
        "default_v2": 0.7,
        "description": "Reward for directly blocking a bot (without puzzle)",
    },
    "penalty_block_human": {
        "label": "Human Block Penalty",
        "group": "reward",
        "values": [-3.0, -2.0, -1.5, -1.0, -0.5],
        "default_v2": -1.5,
        "description": "Penalty for incorrectly blocking a human",
    },
    "penalty_bot_missed_allow": {
        "label": "Missed Bot Penalty",
        "group": "reward",
        "values": [-2.0, -1.5, -1.0, -0.5, -0.1],
        "default_v2": -1.0,
        "description": "Penalty for allowing a bot through",
    },
    # --- Challenge-outcome assumption sweeps ---
    "easy_puzzle_bot_pass": {
        "label": "Easy Puzzle Bot Pass Rate",
        "group": "challenge",
        "values": [0.20, 0.30, 0.40, 0.50, 0.60],
        "default_v2": 0.40,
        "description": "Fraction of bots that pass the easy CAPTCHA",
    },
    "hard_puzzle_bot_pass": {
        "label": "Hard Puzzle Bot Pass Rate",
        "group": "challenge",
        "values": [0.01, 0.03, 0.05, 0.10, 0.15],
        "default_v2": 0.05,
        "description": "Fraction of bots that pass the hard CAPTCHA",
    },
    "hard_puzzle_human_pass": {
        "label": "Hard Puzzle Human Pass Rate",
        "group": "challenge",
        "values": [0.50, 0.60, 0.70, 0.80, 0.90],
        "default_v2": 0.70,
        "description": "Fraction of humans that pass the hard CAPTCHA",
    },
    "tier5_honeypot_rate": {
        "label": "Tier-5 (LLM) Honeypot Trigger Rate",
        "group": "challenge",
        "values": [0.01, 0.02, 0.05, 0.10, 0.20],
        "default_v2": 0.05,
        "description": "Fraction of LLM bots that trigger a honeypot",
    },
    "all_honeypot_rates": {
        "label": "All-Tier Honeypot Rate Scale Factor",
        "group": "challenge",
        "values": [0.25, 0.50, 1.0, 1.5, 2.0],
        "default_v2": 1.0,
        "description": "Multiplicative scale applied to all tier honeypot rates",
    },
}


def _make_modified_config(
    base_cfg: EventEnvConfig, sweep: str, value: float
) -> EventEnvConfig:
    """Return a modified EventEnvConfig with one parameter changed."""
    if sweep == "honeypot_info_bonus":
        return replace(base_cfg, honeypot_info_bonus=value)

    if sweep == "reward_direct_block_bot":
        return replace(base_cfg, reward_direct_block_bot=value)

    if sweep == "penalty_block_human":
        return replace(base_cfg, penalty_block_human=value)

    if sweep == "penalty_bot_missed_allow":
        return replace(base_cfg, penalty_bot_missed_allow=value)

    if sweep == "easy_puzzle_bot_pass":
        rates = dict(base_cfg.puzzle_pass_rates)
        human_pass, _ = rates[2]
        rates[2] = (human_pass, value)
        return replace(base_cfg, puzzle_pass_rates=rates)

    if sweep == "hard_puzzle_bot_pass":
        rates = dict(base_cfg.puzzle_pass_rates)
        human_pass, _ = rates[4]
        rates[4] = (human_pass, value)
        return replace(base_cfg, puzzle_pass_rates=rates)

    if sweep == "hard_puzzle_human_pass":
        rates = dict(base_cfg.puzzle_pass_rates)
        _, bot_pass = rates[4]
        rates[4] = (value, bot_pass)
        return replace(base_cfg, puzzle_pass_rates=rates)

    if sweep == "tier5_honeypot_rate":
        rates = dict(base_cfg.honeypot_trigger_rates_by_tier)
        rates[5] = value
        return replace(base_cfg, honeypot_trigger_rates_by_tier=rates)

    if sweep == "all_honeypot_rates":
        base_rates = dict(base_cfg.honeypot_trigger_rates_by_tier)
        scaled = {t: min(1.0, r * value) for t, r in base_rates.items()}
        return replace(base_cfg, honeypot_trigger_rates_by_tier=scaled)

    raise ValueError(f"Unknown sweep: {sweep}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_sweep_header(sweep: str, meta: dict, agent_names: list[str]):
    print()
    print("=" * 80)
    print(f"  SENSITIVITY SWEEP: {sweep}")
    print(f"  {meta['label']}")
    print(f"  {meta['description']}")
    print(f"  Default (v2): {meta['default_v2']}")
    print(f"  Values: {meta['values']}")
    print(f"  Agents: {', '.join(agent_names)}")
    print("=" * 80)


def _print_sweep_row(value, agent_metrics: dict[str, dict]):
    print(f"\n  --- param_value = {value} ---")
    for name, m in agent_metrics.items():
        print(
            f"  {name:<35s}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
            f"reward={m['avg_reward']:.3f}  hp/ep={m['avg_honeypots_per_ep']:.2f}"
        )


def _print_sweep_summary(sweep: str, meta: dict, results: list[dict]):
    """Print a parseable summary table for generate_paper_figures.py."""
    print()
    print(f"=== SENSITIVITY SUMMARY: {sweep} ===")
    print(f"  Label: {meta['label']}")
    print(f"  Default: {meta['default_v2']}")

    # Collect all agent names
    all_agents = list(results[0]["metrics"].keys()) if results else []
    col_w = max(12, max((len(n) for n in all_agents), default=0) + 2)

    header = f"  {'Value':>8s}"
    for agent in all_agents:
        header += f"  {agent[:col_w]:>{col_w}s}_acc  {agent[:col_w]:>{col_w}s}_f1"
    print(header)
    print("  " + "-" * (10 + (col_w * 2 + 8) * len(all_agents)))

    for row in results:
        v = row["value"]
        line = f"  {v:>8.4f}"
        for agent in all_agents:
            m = row["metrics"].get(agent, {})
            acc = m.get("accuracy", float("nan"))
            f1 = m.get("f1", float("nan"))
            line += f"  {acc:>{col_w}.4f}  {f1:>{col_w}.4f}"
        default_mark = " (*default*)" if abs(v - meta["default_v2"]) < 1e-9 else ""
        print(line + default_mark)

    print(f"=== END SENSITIVITY SUMMARY: {sweep} ===")


def _print_per_tier_sweep_summary(sweep: str, meta: dict, tier_results: list[dict]):
    """Print per-tier detection rates across sweep values."""
    print()
    print(f"=== SENSITIVITY PER-TIER: {sweep} ===")
    all_tiers = sorted({t for row in tier_results for t in row["tier_rates"]})
    all_agents = (
        list(tier_results[0]["tier_rates"][all_tiers[0]].keys())
        if tier_results and all_tiers
        else []
    )

    for tier in all_tiers:
        tier_name = TIER_NAMES.get(tier, "Unknown")
        print(f"\n  Tier {tier} ({tier_name}):")
        print(f"  {'Value':>8s}  " + "  ".join(f"{a:<16s}" for a in all_agents))
        for row in tier_results:
            v = row["value"]
            line = f"  {v:>8.4f}  "
            for agent in all_agents:
                rate = row["tier_rates"].get(tier, {}).get(agent, float("nan"))
                line += f"{rate:>16.3f}  "
            print(line)

    print(f"=== END SENSITIVITY PER-TIER: {sweep} ===")


# ---------------------------------------------------------------------------
# Core sweep runner
# ---------------------------------------------------------------------------


def _run_sweep(
    sweep: str,
    meta: dict,
    agent_specs: list[tuple[str, str]],
    eval_sessions,
    base_cfg: EventEnvConfig,
    cfg: Config,
    args,
) -> None:
    agent_names = [name for name, _ in agent_specs]
    _print_sweep_header(sweep, meta, agent_names)

    eval_seeds = args.eval_seeds or [42]
    multi_seed = len(eval_seeds) > 1

    summary_rows = []
    tier_rows = []

    for value in meta["values"]:
        mod_cfg = _make_modified_config(base_cfg, sweep, value)
        eval_cfg = replace(mod_cfg, augment=False)
        env = EventEnv(eval_sessions, config=eval_cfg)

        default_marker = (
            " (*default*)" if abs(value - meta["default_v2"]) < 1e-9 else ""
        )
        print(f"\n{'='*60}")
        print(f"  {meta['label']} = {value}{default_marker}")

        row_metrics = {}
        row_tier_rates: dict[int, dict[str, float]] = defaultdict(dict)

        for name, path in agent_specs:
            agent = _create_agent(name, cfg, args.device)
            agent.load(path)

            if multi_seed:
                seed_metrics_list = []
                seed_episodes_list = []
                for seed in eval_seeds:
                    r = _run_evaluation(
                        env,
                        agent,
                        args.episodes,
                        agent_name=f"{name}|{value:.3g}/seed={seed}",
                        eval_seed=seed,
                    )
                    seed_episodes_list.append(r["episodes"])
                    seed_metrics_list.append(_compute_metrics(r["episodes"]))

                # Average across seeds
                mean_metrics = {}
                for key in seed_metrics_list[0]:
                    vals = [
                        m[key]
                        for m in seed_metrics_list
                        if not isinstance(m[key], dict)
                    ]
                    if vals:
                        mean_metrics[key] = float(np.mean(vals))
                row_metrics[name] = mean_metrics

                # Per-tier (pooled across seeds)
                combined_eps = [e for eps in seed_episodes_list for e in eps]
                _collect_tier_rates(combined_eps, name, row_tier_rates)
            else:
                r = _run_evaluation(
                    env,
                    agent,
                    args.episodes,
                    agent_name=f"{name}|{value:.3g}",
                    eval_seed=eval_seeds[0],
                )
                row_metrics[name] = _compute_metrics(r["episodes"])
                _collect_tier_rates(r["episodes"], name, row_tier_rates)

        _print_sweep_row(value, row_metrics)
        summary_rows.append({"value": value, "metrics": row_metrics})
        tier_rows.append({"value": value, "tier_rates": dict(row_tier_rates)})

    _print_sweep_summary(sweep, meta, summary_rows)
    _print_per_tier_sweep_summary(sweep, meta, tier_rows)
    print(f"\nSensitivity complete: {sweep}")


def _collect_tier_rates(episodes, agent_name, tier_rates):
    detected_outcomes = {"correct_block", "bot_blocked_puzzle"}
    by_tier = defaultdict(list)
    for e in episodes:
        if e["true_label"] == 0:
            tier = bot_type_to_tier(e.get("bot_type"))
            by_tier[tier].append(e)
    for tier, eps in by_tier.items():
        n = len(eps)
        detected = sum(1 for e in eps if e["outcome"] in detected_outcomes)
        tier_rates[tier][agent_name] = detected / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Sensitivity analysis for reward/challenge parameters"
    )
    p.add_argument(
        "--sweep",
        type=str,
        required=True,
        choices=list(SWEEPS.keys()),
        help="Which parameter to sweep",
    )
    p.add_argument(
        "--agent",
        type=str,
        nargs="+",
        required=True,
        help="Agent checkpoints as name=path pairs",
    )
    p.add_argument("--data-dir", default="data/")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train", "all"],
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Eval RNG seeds (multiple → average across seeds)",
    )
    p.add_argument(
        "--reward-preset",
        default="v2",
        choices=list(REWARD_PRESETS.keys()),
        help="Base reward preset to modify (default: v2)",
    )
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading sessions from {args.data_dir}...")
    sessions = load_from_directory(args.data_dir, include_augmented=False)
    _, _, test_s = split_sessions(
        sessions, train=0.70, val=0.15, test=0.15, seed=args.split_seed
    )
    eval_sessions = test_s if args.split != "all" else sessions
    h = sum(1 for s in eval_sessions if s.label == 1)
    b = sum(1 for s in eval_sessions if s.label == 0)
    print(f"  Eval split: {len(eval_sessions)} ({h} human, {b} bot)")

    base_cfg = REWARD_PRESETS[args.reward_preset]
    cfg = Config()
    cfg.event_env = base_cfg

    sweep = args.sweep
    meta = SWEEPS[sweep]
    agent_specs = _parse_agent_specs(args.agent)

    print(f"\n  Sweep: {sweep}  ({meta['label']})")
    print(f"  Values: {meta['values']}")
    print(f"  Agents: {[n for n, _ in agent_specs]}")
    print(f"  Eval seeds: {args.eval_seeds}")
    print(f"  Episodes: {args.episodes}")

    _run_sweep(sweep, meta, agent_specs, eval_sessions, base_cfg, cfg, args)


if __name__ == "__main__":
    main()
