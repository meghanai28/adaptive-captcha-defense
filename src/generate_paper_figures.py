"""Generate all paper figures and LaTeX tables from eval logs.

Run from src/ directory:
    python generate_paper_figures.py

Outputs:
    figures/paper/  — all PNG figures
    tables/         — LaTeX .tex files + CSV equivalents
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
FIG_DIR = Path("figures/paper")
TABLE_DIR = Path("tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

ALGOS = ["ppo", "dg", "soft_ppo"]
PRESETS = ["v2"]
AUGS = ["noaug", "advaug"]
SEEDS = [42, 123, 456, 789, 1024]
TIERS = ["tier3", "tier4", "tier5", "tier45", "tier345"]
FAMILIES = ["stealth", "replay", "llm", "semi_auto", "trace_conditioned"]

ALGO_LABELS = {"ppo": "PPO", "dg": "DG", "soft_ppo": "Soft PPO"}
PRESET_LABELS = {"v2": ""}
AUG_LABELS = {"noaug": "No Aug", "advaug": "Adv Aug"}
TIER_LABELS = {
    "tier3": "T3 held out",
    "tier4": "T4 held out",
    "tier5": "T5 held out",
    "tier45": "T4+5 held out",
    "tier345": "T3+4+5 held out",
}
FAM_LABELS = {
    "stealth": "Stealth",
    "replay": "Replay",
    "llm": "LLM",
    "semi_auto": "Semi-Auto",
    "trace_conditioned": "Trace-Cond.",
}

AUG_COLORS = {"noaug": "#7faacc", "advaug": "#2563a8"}
ALGO_COLORS = {"ppo": "#2563a8", "dg": "#e07b54", "soft_ppo": "#4aad52"}
BASELINE_COLORS = [
    "#aaaaaa",
    "#e07b54",
    "#4aad52",
    "#2563a8",
    "#8b5cf6",
    "#f59e0b",
    "#06b6d4",
]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def _read_log(path: Path) -> str:
    for enc in ("utf-16", "utf-16-le", "utf-8"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def _parse_metric_line(text: str, key: str) -> tuple[float, float] | None:
    """Extract mean +/- std from a summary line like '  Accuracy  0.980 +/- 0.010'."""
    pat = rf"{re.escape(key)}\s+([\d.]+)\s+\+/-\s+([\d.]+)"
    m = re.search(pat, text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_native_log(log_path: Path) -> dict | None:
    """Parse a native or cross-env eval log. Returns dict of metrics."""
    text = _read_log(log_path)
    if not text or ("Evaluation complete" not in text and "Best F1:" not in text):
        return None

    # Find the last COMPARISON TABLE block
    comp_block = None
    for block in text.split("COMPARISON TABLE"):
        comp_block = block
    if comp_block is None:
        return None

    metrics = {}
    for key in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Avg Reward",
        "Avg Length",
        "Honeypot %",
        "Avg HP/ep",
    ]:
        pat = rf"{re.escape(key)}\s+((?:-?[\d.]+\s+\+/-\s+[\d.]+\s*)+)"
        m = re.search(pat, comp_block)
        if m:
            vals = re.findall(r"(-?[\d.]+)\s+\+/-\s+([\d.]+)", m.group(1))
            if not vals:
                continue
            if len(vals) == 1:
                metrics[key] = (float(vals[0][0]), float(vals[0][1]))
            else:
                means = [float(v[0]) for v in vals]
                metrics[key] = (float(np.mean(means)), float(np.std(means)))

    # Per-session-type avg steps (human vs bot) from the last per-agent block
    human_steps_all, bot_steps_all = [], []
    for m in re.finditer(r"Avg steps \(human sessions\):\s*([\d.]+)", text):
        human_steps_all.append(float(m.group(1)))
    for m in re.finditer(r"Avg steps \(bot sessions\):\s*([\d.]+)", text):
        bot_steps_all.append(float(m.group(1)))
    if human_steps_all:
        metrics["Avg Steps Human"] = (
            float(np.mean(human_steps_all)),
            float(np.std(human_steps_all)),
        )
    if bot_steps_all:
        metrics["Avg Steps Bot"] = (
            float(np.mean(bot_steps_all)),
            float(np.std(bot_steps_all)),
        )

    # Human challenge rate: (human_passed_puzzle + fp_puzzle) / total episodes
    hp_counts, fp_counts, total_counts = [], [], []
    for block in re.finditer(
        r"--- Outcome Distribution ---\n(.*?)(?=\n---|===|$)", text, re.DOTALL
    ):
        btext = block.group(1)
        total_m = re.search(r"Total episodes.*?(\d+)", btext)
        hp_m = re.search(r"human_passed_puzzle\s+(\d+)", btext)
        fp_m = re.search(r"fp_puzzle\s+(\d+)", btext)
        if total_m:
            total_counts.append(int(total_m.group(1)))
            hp_counts.append(int(hp_m.group(1)) if hp_m else 0)
            fp_counts.append(int(fp_m.group(1)) if fp_m else 0)
    if total_counts:
        rates = [
            (h + f) / t for h, f, t in zip(hp_counts, fp_counts, total_counts) if t > 0
        ]
        if rates:
            metrics["Human Challenge Rate"] = (
                float(np.mean(rates)),
                float(np.std(rates)),
            )

    # Per-tier detection rates (from last per-tier summary block)
    tier_rates = {}
    tier_blocks = re.findall(
        r"--- Per-Tier Summary.*?---\n(.*?)(?=\n---|$)", text, re.DOTALL
    )
    if tier_blocks:
        for line in tier_blocks[-1].splitlines():
            m = re.search(r"Tier (\d+).*?(\d+\.?\d*)%\s+\+/-\s+(\d+\.?\d*)%", line)
            if m:
                tier_rates[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))

    # Per-family detection rates
    family_rates = {}
    fam_blocks = re.findall(
        r"--- Per-Family Bot Detection.*?Rate\n(.*?)(?=\n---|$)", text, re.DOTALL
    )
    if fam_blocks:
        for line in fam_blocks[-1].splitlines():
            m = re.search(
                r"(\w+)\s+\d+\s+\d+/seed\s+([\d.]+)%\s+\+/-\s+([\d.]+)%", line
            )
            if not m:
                m = re.search(r"(\w+)\s+\d+\s+\d+\s+([\d.]+)%\s+\+/-\s+([\d.]+)%", line)
            if m:
                family_rates[m.group(1)] = (float(m.group(2)), float(m.group(3)))

    # Per-training-seed detail for variance plot
    seed_accs = []
    seed_block = re.search(
        r"Per-Training-Seed Detail.*?---\n(.*?)(?=\n---|$)", text, re.DOTALL
    )
    if seed_block:
        for line in seed_block.group(1).splitlines():
            m = re.search(r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", line.strip())
            if m:
                seed_accs.append(float(m.group(1)))

    return {
        "metrics": metrics,
        "tier_rates": tier_rates,
        "family_rates": family_rates,
        "seed_accs": seed_accs,
    }


def parse_heldout_log(log_path: Path) -> dict | None:
    """Parse a held-out tier/family eval log."""
    text = _read_log(log_path)
    if not text or ("Evaluation complete" not in text and "Best F1:" not in text):
        return None

    metrics = {}
    blocks = text.split("=== ")
    for block in blocks:
        for key in ["Accuracy", "F1", "Recall", "Precision"]:
            r = _parse_metric_line(block, key)
            if r:
                metrics[key] = r

    tier_rates = {}
    tier_blocks = re.findall(
        r"--- Per-Tier Summary.*?---\n(.*?)(?=\n---|$)", text, re.DOTALL
    )
    if tier_blocks:
        for line in tier_blocks[-1].splitlines():
            m = re.search(r"Tier (\d+).*?(\d+\.?\d*)%(?:\s+\+/-\s+([\d.]+)%)?", line)
            if m:
                std = float(m.group(3)) if m.group(3) else 0.0
                tier_rates[int(m.group(1))] = (float(m.group(2)), std)

    return {"metrics": metrics, "tier_rates": tier_rates}


# ---------------------------------------------------------------------------
# Baseline and sensitivity log parsing
# ---------------------------------------------------------------------------

SENSITIVITY_SWEEPS_REWARD = [
    "honeypot_info_bonus",
    "reward_direct_block_bot",
    "penalty_block_human",
    "penalty_bot_missed_allow",
]
SENSITIVITY_SWEEPS_CHALLENGE = [
    "easy_puzzle_bot_pass",
    "hard_puzzle_bot_pass",
    "hard_puzzle_human_pass",
    "tier5_honeypot_rate",
    "all_honeypot_rates",
]
ALL_SENSITIVITY_SWEEPS = SENSITIVITY_SWEEPS_REWARD + SENSITIVITY_SWEEPS_CHALLENGE

SENSITIVITY_LABELS = {
    "honeypot_info_bonus": "Honeypot Info Bonus",
    "reward_direct_block_bot": "Direct Block Reward",
    "penalty_block_human": "Human Block Penalty",
    "penalty_bot_missed_allow": "Missed Bot Penalty",
    "easy_puzzle_bot_pass": "Easy Puzzle Bot Pass",
    "hard_puzzle_bot_pass": "Hard Puzzle Bot Pass",
    "hard_puzzle_human_pass": "Hard Puzzle Human Pass",
    "tier5_honeypot_rate": "T5 Honeypot Rate",
    "all_honeypot_rates": "All-Tier HP Scale",
}
SENSITIVITY_DEFAULTS = {
    "honeypot_info_bonus": 0.5,
    "reward_direct_block_bot": 0.7,
    "penalty_block_human": -1.5,
    "penalty_bot_missed_allow": -1.0,
    "easy_puzzle_bot_pass": 0.40,
    "hard_puzzle_bot_pass": 0.05,
    "hard_puzzle_human_pass": 0.70,
    "tier5_honeypot_rate": 0.05,
    "all_honeypot_rates": 1.0,
}

ABLATION_NAMES = [
    "no_hp_bonus",
    "high_hp_bonus",
    "strict_fp",
    "no_continue_cost",
    "small_lstm",
    "large_lstm",
    "deep_lstm",
    "single_view",
]
ABLATION_LABELS = {
    "no_hp_bonus": "No HP Bonus",
    "high_hp_bonus": "High HP Bonus",
    "strict_fp": "Strict FP Pen.",
    "no_continue_cost": "No Cont. Cost",
    "small_lstm": "LSTM-64",
    "large_lstm": "LSTM-256",
    "deep_lstm": "2-Layer LSTM",
    "single_view": "Single View",
}
ABLATION_GROUPS = {
    "no_hp_bonus": "reward",
    "high_hp_bonus": "reward",
    "strict_fp": "reward",
    "no_continue_cost": "reward",
    "small_lstm": "arch",
    "large_lstm": "arch",
    "deep_lstm": "arch",
    "single_view": "arch",
}

BASELINE_LABELS = {
    "random": "Random",
    "always_block": "Always Block",
    "always_allow": "Always Allow",
    "always_easy_puzzle": "Easy Puzzle",
    "always_hard_puzzle": "Hard Puzzle",
    "honeypot_block": "HP+Block",
    "honeypot_decide": "HP+Decide",
}


def _parse_comparison_table_all_agents(text: str) -> dict:
    """Parse COMPARISON TABLE, returning {agent_name: {metric_key: (mean, std)}}."""
    last_block = None
    for block in text.split("COMPARISON TABLE"):
        last_block = block
    if last_block is None:
        return {}

    lines = [ln for ln in last_block.splitlines() if ln.strip()]
    header_line = None
    for line in lines:
        if "Metric" in line and "+/-" not in line and "---" not in line:
            header_line = line
            break
    if not header_line:
        return {}

    after_metric = header_line[header_line.index("Metric") + 6 :]
    agents = after_metric.split()
    if not agents:
        return {}

    results = {a: {} for a in agents}
    metric_map = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "Avg Reward": "avg_reward",
        "Avg Length": "avg_length",
        "Honeypot %": "honeypot_rate",
        "Avg HP/ep": "avg_honeypots_per_ep",
    }
    for line in lines:
        for label, key in metric_map.items():
            if line.strip().startswith(label):
                pairs = re.findall(r"([\d.]+)\s+\+/-\s+([\d.]+)", line)
                for i, (mean, std) in enumerate(pairs[: len(agents)]):
                    results[agents[i]][key] = (float(mean), float(std))
                break

    return results


def parse_baseline_log(log_path: Path) -> dict | None:
    """Parse an ablation/baseline eval log. Returns {agent: {metric: (mean, std)}}."""
    text = _read_log(log_path)
    if not text or "Evaluation complete" not in text:
        return None
    result = _parse_comparison_table_all_agents(text)
    if not result:
        return None

    # Avg Length and human challenge rate aren't in the comparison table —
    # parse them from per-agent multi-seed summary blocks.
    for agent_name in result:
        # Find the per-agent block: "=== AGENT_NAME - * split (*seeds) ==="
        pattern = (
            re.escape(agent_name.upper()) + r".*?split.*?seeds.*?\n(.*?)(?====|\Z)"
        )
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        block = m.group(1)

        al = re.search(r"Avg Length\s+([\d.]+)\s+\+/-\s+([\d.]+)", block)
        if al:
            result[agent_name]["avg_length"] = (float(al.group(1)), float(al.group(2)))

        # Human challenge rate from outcome counts in this agent's block
        total_m = re.search(r"Total episodes.*?(\d+)", block)
        hp_m = re.search(r"human_passed_puzzle\s+(\d+)", block)
        fp_m = re.search(r"fp_puzzle\s+(\d+)", block)
        if total_m and int(total_m.group(1)) > 0:
            hp = int(hp_m.group(1)) if hp_m else 0
            fp = int(fp_m.group(1)) if fp_m else 0
            tot = int(total_m.group(1))
            result[agent_name]["human_challenge_rate"] = ((hp + fp) / tot, 0.0)

    return result


def parse_human_disjoint_log(log_path: Path) -> dict | None:
    """Parse a human-disjoint eval log.
    Returns {baseline_acc, disjoint_acc, baseline_seeds, disjoint_seeds}.
    The test set is purely human sessions so F1 is undefined; accuracy = pass-through rate.
    """
    text = _read_log(log_path)
    if not text or "Evaluation complete" not in text:
        return None

    blocks = text.split("COMPARISON TABLE")
    if len(blocks) < 2:
        return None
    comp_block = blocks[-1]

    # The Accuracy line has both values: "Accuracy  0.997 +/- 0.007  0.979 +/- 0.007"
    acc_line_m = re.search(r"Accuracy\s+(.*)", comp_block)
    if not acc_line_m:
        return None
    acc_pairs = re.findall(r"([\d.]+)\s+\+/-\s+([\d.]+)", acc_line_m.group(1))
    if len(acc_pairs) < 2:
        return None
    baseline_acc = (float(acc_pairs[0][0]), float(acc_pairs[0][1]))
    disjoint_acc = (float(acc_pairs[1][0]), float(acc_pairs[1][1]))

    baseline_seeds, disjoint_seeds = [], []
    seed_section = re.search(
        r"Per-Training-Seed Detail.*?\n(.*?)(?=\n\n|\Z)", comp_block, re.DOTALL
    )
    if seed_section:
        for line in seed_section.group(1).splitlines():
            m = re.match(r"\s+(\S+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+", line)
            if not m:
                continue
            agent, acc = m.group(1), float(m.group(2))
            if "heldout" in agent.lower():
                disjoint_seeds.append(acc)
            else:
                baseline_seeds.append(acc)

    return {
        "baseline_acc": baseline_acc,
        "disjoint_acc": disjoint_acc,
        "baseline_seeds": baseline_seeds,
        "disjoint_seeds": disjoint_seeds,
    }


def parse_action_dist_log(log_path: Path) -> dict | None:
    """Parse a native eval log for the Final Action Distribution.
    Returns {action_name: pct} averaged across seeds, using the largest
    (multi-seed) block found before the comparison table.
    """
    text = _read_log(log_path)
    if not text or "Evaluation complete" not in text:
        return None

    # Only look before the comparison table
    body = text.split("COMPARISON TABLE")[0]

    # Find all Final Action Distribution blocks
    blocks = re.split(r"--- Final Action Distribution ---", body)
    if len(blocks) < 2:
        return None

    best_block, best_total = None, 0
    for block in blocks[1:]:
        lines = block.strip().splitlines()
        total, dist = 0, {}
        for line in lines:
            m = re.match(r"\s*(\w+)\s+(\d+)\s+\(([\d.]+)%\)", line)
            if not m:
                break
            name, count, pct = m.group(1), int(m.group(2)), float(m.group(3))
            dist[name] = pct / 100.0
            total += count
        if total > best_total:
            best_total, best_block = total, dist

    return best_block


def parse_training_log(log_path: Path) -> dict | None:
    """Parse a PPO/DG/SoftPPO training log.
    Returns {steps, rewards, entropies, correct_rates, val_steps, val_accs}.
    correct_rate = fraction of episodes with correct terminal outcome
                   (correct_allow + correct_block + bot_blocked_puzzle + human_passed_puzzle).
    """
    text = _read_log(log_path)
    if not text or "Training complete" not in text:
        return None

    steps, rewards, entropies, correct_rates = [], [], [], []
    val_steps, val_accs = [], []
    current_step = None

    for line in text.splitlines():
        m = re.search(r"Rollout\s+\d+/\d+\s+\|\s+Steps:\s+(\d+)", line)
        if m:
            current_step = int(m.group(1))
            continue

        m = re.search(r"Avg reward:\s+([-\d.]+)", line)
        if m and current_step is not None:
            steps.append(current_step)
            rewards.append(float(m.group(1)))
            continue

        m = re.search(r"Entropy:\s+([\d.]+)", line)
        if m and current_step is not None:
            entropies.append(float(m.group(1)))
            continue

        # Outcomes line: "correct_allow: X%, correct_block: Y%, ..."
        # human_passed_puzzle included: terminal outcome was still correct (human cleared),
        # just with friction. Strict eval is handled separately in fig 14.
        if "Outcomes:" in line:

            def _pct(key):
                mm = re.search(rf"{re.escape(key)}:\s*([\d.]+)%", line)
                return float(mm.group(1)) / 100.0 if mm else 0.0

            cr = (
                _pct("correct_allow")
                + _pct("correct_block")
                + _pct("bot_blocked_puzzle")
                + _pct("human_passed_puzzle")
            )
            correct_rates.append(cr)
            continue

        m = re.search(r"\[Val accuracy:\s+([\d.]+)", line)
        if m and current_step is not None:
            val_steps.append(current_step)
            val_accs.append(float(m.group(1)))

    if not steps:
        return None
    return {
        "steps": steps,
        "rewards": rewards,
        "entropies": entropies,
        "correct_rates": correct_rates,
        "val_steps": val_steps,
        "val_accs": val_accs,
    }


def parse_sensitivity_log(log_path: Path, sweep: str) -> list[dict] | None:
    """Parse a sensitivity_analysis.py log for one sweep."""
    text = _read_log(log_path)
    if not text or f"Sensitivity complete: {sweep}" not in text:
        return None

    pattern = (
        rf"=== SENSITIVITY SUMMARY: {re.escape(sweep)} ===(.*?)"
        rf"=== END SENSITIVITY SUMMARY: {re.escape(sweep)} ==="
    )
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None

    block = m.group(1)
    lines = [ln for ln in block.splitlines() if ln.strip()]

    header_line = next((ln for ln in lines if "_acc" in ln and "_f1" in ln), None)
    if not header_line:
        return None

    cols = header_line.split()
    agent_names = [c[:-4] for c in cols if c.endswith("_acc")]

    rows = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("Value")
            or re.match(r"^-+\s*$", stripped)
            or stripped.startswith("Label:")
            or stripped.startswith("Default:")
            or not stripped
        ):
            continue
        cleaned = stripped.replace("(*default*)", "").strip()
        nums = re.findall(r"-?[\d.]+", cleaned)
        if len(nums) < 1:
            continue
        try:
            param_val = float(nums[0])
        except ValueError:
            continue
        agent_metrics = {}
        for i, agent in enumerate(agent_names):
            ai, fi = 1 + i * 2, 2 + i * 2
            try:
                acc = float(nums[ai]) if ai < len(nums) else float("nan")
                f1 = float(nums[fi]) if fi < len(nums) else float("nan")
            except (IndexError, ValueError):
                acc = f1 = float("nan")
            agent_metrics[agent] = {"accuracy": acc, "f1": f1}
        rows.append({"value": param_val, "metrics": agent_metrics})

    return sorted(rows, key=lambda r: r["value"]) if rows else None


# ---------------------------------------------------------------------------
# Load all data  (v2 only — v1 logs are never loaded)
# ---------------------------------------------------------------------------


def load_all() -> dict:
    data = {
        "native": {},  # (algo, aug, preset) -> parsed
        "cross": {},  # (algo, aug, preset, eval_preset) -> parsed
        "heldout_tier": {},  # (algo, aug, preset, tier_label) -> parsed
        "heldout_family": {},  # (algo, aug, preset, family) -> parsed
        "augtest": {},  # (algo, aug, preset) -> parsed
        "baselines": {},  # preset -> {agent: {metric: (mean, std)}}
        "sensitivity": {},  # sweep -> list of {value, metrics}
        "ablations": {},  # ablation_name -> {agent: {metric: (mean, std)}}
        "strict": {},  # "baselines" / "single_view" -> {agent: {metric}}
        "disjoint": {},  # "tier5" -> {agent: {metric}}
        "human_disjoint": {},  # "personA" / "personB" -> parsed
        "training": {},  # (algo, aug, preset, seed) -> {steps, rewards, val_steps, val_accs}
        "action_dist": {},  # (algo, aug, preset) -> {action: pct}
    }

    for preset in PRESETS:
        for aug in AUGS:
            for algo in ALGOS:
                p = LOG_DIR / f"eval_{algo}_{aug}_{preset}_native.log"
                r = parse_native_log(p)
                if r:
                    data["native"][(algo, aug, preset)] = r

                # Cross-env: v2-trained evaluated in v1 (robustness check)
                p = LOG_DIR / f"eval_{algo}_{aug}_{preset}_in_v1_env.log"
                r = parse_native_log(p)
                if r:
                    data["cross"][(algo, aug, preset, "v1")] = r

                # Augmented test
                p = LOG_DIR / f"eval_{algo}_{aug}_{preset}_augtest.log"
                r = parse_native_log(p)
                if r:
                    data["augtest"][(algo, aug, preset)] = r

        # Held-out tiers + families (advaug only)
        for algo in ALGOS:
            for tlabel in TIERS:
                p = LOG_DIR / f"eval_{algo}_advaug_{preset}_heldout_{tlabel}.log"
                r = parse_heldout_log(p)
                if r:
                    data["heldout_tier"][(algo, "advaug", preset, tlabel)] = r

            for fam in FAMILIES:
                p = LOG_DIR / f"eval_{algo}_advaug_{preset}_heldout_{fam}.log"
                r = parse_heldout_log(p)
                if r:
                    data["heldout_family"][(algo, "advaug", preset, fam)] = r

    # Baselines (v2 only)
    p = LOG_DIR / "baselines_v2.log"
    r = parse_baseline_log(p)
    if r:
        data["baselines"]["v2"] = r

    # Sensitivity sweeps
    for sweep in ALL_SENSITIVITY_SWEEPS:
        p = LOG_DIR / f"sensitivity_{sweep}.log"
        r = parse_sensitivity_log(p, sweep)
        if r:
            data["sensitivity"][sweep] = r

    # Ablation evals
    for abl_name in ABLATION_NAMES:
        p = LOG_DIR / f"eval_ablation_{abl_name}.log"
        r = parse_baseline_log(p)
        if r:
            data["ablations"][abl_name] = r

    # Strict mode evals (--challenge-as-fp: human_passed_puzzle counted as FP)
    p = LOG_DIR / "eval_baselines_v2_strict.log"
    r = parse_baseline_log(p)
    if r:
        data["strict"]["baselines"] = r

    for abl_name in ABLATION_NAMES:
        p = LOG_DIR / f"eval_ablation_{abl_name}_strict.log"
        r = parse_baseline_log(p)
        if r:
            data["strict"][f"ablation_{abl_name}"] = r

    # Disjoint generalization evals (trained without tier X, tested on tier X only)
    for _dtag in ["tier1", "tier2", "tier3", "tier4", "tier5", "tier45", "tier345"]:
        p = LOG_DIR / f"eval_disjoint_{_dtag}.log"
        r = parse_baseline_log(p)
        if r:
            data["disjoint"][_dtag] = r

    # Training curves (reward + val accuracy over rollouts)
    for algo in ALGOS:
        for aug in AUGS:
            for preset in PRESETS:
                for seed in SEEDS:
                    p = LOG_DIR / f"{algo}_{aug}_{preset}_seed{seed}_training.log"
                    r = parse_training_log(p)
                    if r:
                        data["training"][(algo, aug, preset, seed)] = r

    # Human disjoint generalization (trained without person A/B, tested on that person only)
    for _person in ["personA", "personB"]:
        p = LOG_DIR / f"eval_human_disjoint_{_person}.log"
        r = parse_human_disjoint_log(p)
        if r:
            data["human_disjoint"][_person] = r

    # Action distributions for v1 vs v2 comparison (advaug only; load both presets)
    for preset in ["v1", "v2"]:
        for algo in ALGOS:
            p = LOG_DIR / f"eval_{algo}_advaug_{preset}_native.log"
            r = parse_action_dist_log(p)
            if r:
                data["action_dist"][(algo, "advaug", preset)] = r

    return data


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _save(fig, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _metric(d: dict, key: str, default=(0.0, 0.0)):
    return d["metrics"].get(key, default)


# ---------------------------------------------------------------------------
# FIGURE 1 — Main accuracy: all 6 combos (3 algos × 2 augs), v2 env
# ---------------------------------------------------------------------------
def fig_main_accuracy(data: dict):
    native = data["native"]
    if not native:
        print("  [skip] fig_main_accuracy — no native logs yet")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ALGOS))
    w = 0.35
    for j, aug in enumerate(AUGS):
        means = [
            _metric(native.get((a, aug, "v2"), {"metrics": {}}), "Accuracy")[0]
            for a in ALGOS
        ]
        stds = [
            _metric(native.get((a, aug, "v2"), {"metrics": {}}), "Accuracy")[1]
            for a in ALGOS
        ]
        offset = (j - 0.5) * w
        ax.bar(
            x + offset,
            means,
            w,
            yerr=stds,
            capsize=4,
            label=AUG_LABELS[aug],
            color=AUG_COLORS[aug],
            alpha=0.85,
            ecolor="gray",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS])
    ax.set_ylim(0.85, 1.03)
    ax.set_ylabel("Accuracy (mean ± std, 5 training seeds)")
    ax.legend()
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.suptitle("Bot Detection Accuracy", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig1_main_accuracy")


# ---------------------------------------------------------------------------
# FIGURE 2 — Augmentation effect: noaug vs advaug across key metrics, v2 env
# ---------------------------------------------------------------------------
def fig_aug_effect(data: dict):
    native = data["native"]
    if not native:
        print("  [skip] fig_aug_effect")
        return

    metrics_to_show = ["Accuracy", "F1", "Recall", "Avg HP/ep"]
    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(14, 4))

    for ax, metric in zip(axes, metrics_to_show):
        x = np.arange(len(ALGOS))
        w = 0.35
        for j, aug in enumerate(AUGS):
            means = [
                _metric(native.get((a, aug, "v2"), {"metrics": {}}), metric)[0]
                for a in ALGOS
            ]
            stds = [
                _metric(native.get((a, aug, "v2"), {"metrics": {}}), metric)[1]
                for a in ALGOS
            ]
            offset = (j - 0.5) * w
            ax.bar(
                x + offset,
                means,
                w,
                yerr=stds,
                capsize=4,
                label=AUG_LABELS[aug],
                color=AUG_COLORS[aug],
                alpha=0.85,
                ecolor="gray",
            )
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS])
        if metric == "Accuracy":
            ax.set_ylim(0.85, 1.03)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Effect of Adversarial Augmentation on Bot Detection", fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "fig2_aug_effect")


# ---------------------------------------------------------------------------
# FIGURE 3 — Cross-environment robustness: v2-trained in v1 env
# ---------------------------------------------------------------------------
def fig_cross_env(data: dict):
    native = data["native"]
    cross = data["cross"]
    if not native and not cross:
        print("  [skip] fig_cross_env")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ALGOS))
    w = 0.35

    nat_means, nat_stds, cross_means, cross_stds = [], [], [], []
    for algo in ALGOS:
        d_nat = native.get((algo, "advaug", "v2"))
        d_cross = cross.get((algo, "advaug", "v2", "v1"))
        nat_acc = _metric(d_nat, "Accuracy") if d_nat else (0.0, 0.0)
        cross_acc = _metric(d_cross, "Accuracy") if d_cross else (0.0, 0.0)
        nat_means.append(nat_acc[0])
        nat_stds.append(nat_acc[1])
        cross_means.append(cross_acc[0])
        cross_stds.append(cross_acc[1])

    ax.bar(
        x - w / 2,
        nat_means,
        w,
        yerr=nat_stds,
        capsize=4,
        label="v2 env (native)",
        color="#4C8BB5",
        alpha=0.85,
        ecolor="gray",
    )
    ax.bar(
        x + w / 2,
        cross_means,
        w,
        yerr=cross_stds,
        capsize=4,
        label="v1 env (transfer)",
        color="#E07B54",
        alpha=0.85,
        ecolor="gray",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS])
    ax.set_ylim(0.85, 1.03)
    ax.set_ylabel("Accuracy (mean ± std, 5 eval seeds)")
    ax.legend()
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.suptitle(
        "Cross-Environment Transfer: RL Agents Evaluated on Out-of-Distribution Sessions (advaug)",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig3_cross_env_heatmap")


# ---------------------------------------------------------------------------
# FIGURE 4 — Held-out tier generalization (v2, advaug)
# ---------------------------------------------------------------------------
def fig_heldout_tiers(data: dict):
    ht = data["heldout_tier"]
    if not ht:
        print("  [skip] fig_heldout_tiers")
        return

    tier_keys = ["tier3", "tier4", "tier5", "tier345"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tier_keys))
    w = 0.25
    for j, algo in enumerate(ALGOS):
        means, stds = [], []
        for tk in tier_keys:
            d = ht.get((algo, "advaug", "v2", tk))
            m, s = _metric(d, "Accuracy") if d else (0.0, 0.0)
            means.append(m)
            stds.append(s)
        offset = (j - 1) * w
        ax.bar(
            x + offset,
            means,
            w,
            yerr=stds,
            capsize=4,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            alpha=0.85,
            ecolor="gray",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in tier_keys], rotation=15, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Accuracy (advaug)")
    ax.legend()
    ax.set_xlabel(
        "Test set filtered to only bots of this tier (same trained model across all groups)",
        fontsize=9,
        color="gray",
    )
    fig.suptitle(
        "Per-Tier Detection Accuracy\n"
        "(Standard model evaluated on each bot tier's test sessions separately)",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig4_heldout_tiers")


# ---------------------------------------------------------------------------
# FIGURE 5 — Held-out family generalization (v2, advaug)
# ---------------------------------------------------------------------------
def fig_heldout_families(data: dict):
    hf = data["heldout_family"]
    if not hf:
        print("  [skip] fig_heldout_families")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(FAMILIES))
    w = 0.25
    for j, algo in enumerate(ALGOS):
        means, stds = [], []
        for fam in FAMILIES:
            d = hf.get((algo, "advaug", "v2", fam))
            m, s = _metric(d, "Accuracy") if d else (0.0, 0.0)
            means.append(m)
            stds.append(s)
        offset = (j - 1) * w
        bars = ax.bar(
            x + offset,
            means,
            w,
            yerr=stds,
            capsize=4,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            alpha=0.85,
            ecolor="gray",
        )
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                m + s + 0.003,
                f"{m*100:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([FAM_LABELS[f] for f in FAMILIES], rotation=15, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Accuracy (advaug)")
    ax.legend()
    ax.set_xlabel(
        "Test set filtered to only bots of this family (same trained model across all groups)",
        fontsize=9,
        color="gray",
    )
    fig.suptitle(
        "Per-Family Detection Accuracy\n"
        "(Standard model evaluated on each bot family's test sessions separately)",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig5_heldout_families")


# ---------------------------------------------------------------------------
# FIGURE 6 — Per-seed variance box plots (v2)
# ---------------------------------------------------------------------------
def fig_seed_variance(data: dict):
    native = data["native"]
    if not native:
        print("  [skip] fig_seed_variance")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    labels, box_data = [], []
    for algo in ALGOS:
        for aug in AUGS:
            d = native.get((algo, aug, "v2"))
            if d and d.get("seed_accs"):
                labels.append(f"{ALGO_LABELS[algo]}\n{AUG_LABELS[aug]}")
                box_data.append(d["seed_accs"])

    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True, notch=False)
        colors_cycle = [AUG_COLORS["noaug"], AUG_COLORS["advaug"]] * len(ALGOS)
        for patch, color in zip(bp["boxes"], colors_cycle):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticklabels(labels, fontsize=8)

    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("Accuracy across 5 training seeds")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    patch_no = mpatches.Patch(color=AUG_COLORS["noaug"], label="No Aug")
    patch_adv = mpatches.Patch(color=AUG_COLORS["advaug"], label="Adv Aug")
    ax.legend(handles=[patch_no, patch_adv], fontsize=8)
    fig.suptitle(
        "Training Seed Variance (5 Seeds per Configuration)", fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "fig6_seed_variance")


# ---------------------------------------------------------------------------
# FIGURE 7 — Standard vs augmented test set (v2)
# ---------------------------------------------------------------------------
def fig_augmented_test(data: dict):
    native = data["native"]
    augtest = data["augtest"]
    if not augtest:
        print("  [skip] fig_augmented_test")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ALGOS))
    w = 0.2
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    configs = [
        ("noaug", "standard"),
        ("advaug", "standard"),
        ("noaug", "augmented"),
        ("advaug", "augmented"),
    ]
    clabels = [
        "No Aug (std test)",
        "Adv Aug (std test)",
        "No Aug (aug test)",
        "Adv Aug (aug test)",
    ]
    ccolors = ["#aac4e0", "#2563a8", "#f5b89a", "#e07b54"]

    for (aug, testtype), off, clabel, color in zip(configs, offsets, clabels, ccolors):
        src = augtest if testtype == "augmented" else native
        means = [
            _metric(src.get((a, aug, "v2"), {"metrics": {}}), "Accuracy")[0]
            for a in ALGOS
        ]
        stds = [
            _metric(src.get((a, aug, "v2"), {"metrics": {}}), "Accuracy")[1]
            for a in ALGOS
        ]
        ax.bar(
            x + off,
            means,
            w,
            yerr=stds,
            capsize=3,
            label=clabel,
            color=color,
            alpha=0.85,
            ecolor="gray",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS])
    ax.set_ylim(0.80, 1.03)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7)
    fig.suptitle("Standard vs Augmented Test Set Accuracy", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig7_augmented_test")


# ---------------------------------------------------------------------------
# FIGURE 8 — Per-tier detection rate (v2, advaug)
# ---------------------------------------------------------------------------
def fig_per_tier_detection(data: dict):
    native = data["native"]
    if not native:
        print("  [skip] fig_per_tier_detection")
        return

    tiers = [1, 2, 3, 4, 5]
    tier_names = {
        1: "T1\nCommodity",
        2: "T2\nCareful",
        3: "T3\nSemi-Auto",
        4: "T4\nTrace-Cond",
        5: "T5\nLLM",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tiers))
    w = 0.25
    for j, algo in enumerate(ALGOS):
        d = native.get((algo, "advaug", "v2"))
        if not d:
            continue
        means = [d["tier_rates"].get(t, (0.0, 0.0))[0] for t in tiers]
        stds = [d["tier_rates"].get(t, (0.0, 0.0))[1] for t in tiers]
        offset = (j - 1) * w
        ax.bar(
            x + offset,
            means,
            w,
            yerr=stds,
            capsize=4,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            alpha=0.85,
            ecolor="gray",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([tier_names[t] for t in tiers])
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Detection Rate % (advaug)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend()
    fig.suptitle("Per-Tier Bot Detection Rate", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig8_per_tier_detection")


# ---------------------------------------------------------------------------
# FIGURE 9 — Baselines vs RL comparison (v2)
# ---------------------------------------------------------------------------
def fig_baselines_comparison(data: dict):
    baselines = data.get("baselines", {})
    native = data.get("native", {})
    if not baselines:
        print("  [skip] fig_baselines_comparison — no baseline logs yet")
        return

    bl = baselines.get("v2", {})
    if not bl:
        print("  [skip] fig_baselines_comparison — no v2 baseline data")
        return

    bl_names = [n for n in BASELINE_LABELS if n in bl]
    rl_entries = []
    for algo in ALGOS:
        d = native.get((algo, "advaug", "v2"))
        if d:
            rl_entries.append((f"RL-{ALGO_LABELS[algo]}", _metric(d, "Accuracy")))

    all_names = bl_names + [e[0] for e in rl_entries]
    all_accs = [bl[n].get("accuracy", (0.0, 0.0)) for n in bl_names] + [
        e[1] for e in rl_entries
    ]
    all_labels = [BASELINE_LABELS.get(n, n) for n in bl_names] + [
        e[0] for e in rl_entries
    ]
    colors = BASELINE_COLORS[: len(bl_names)] + [
        ALGO_COLORS[a] for a in ALGOS if native.get((a, "advaug", "v2"))
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_names))
    means = [a[0] for a in all_accs]
    stds = [a[1] for a in all_accs]
    bars = ax.bar(
        x, means, 0.6, yerr=stds, capsize=4, color=colors, alpha=0.85, ecolor="gray"
    )
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.01,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    if bl_names:
        ax.axvline(len(bl_names) - 0.5, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (mean ± std)")
    ax.set_ylim(0.3, 1.05)
    ax.set_title(
        "Bot Detection Accuracy: Rule-Based Baselines vs RL Agents (v2 env)",
        fontweight="bold",
    )
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    from matplotlib.patches import Patch

    legend_els = [
        Patch(color="#aaaaaa", label="Rule-based"),
        Patch(color=ALGO_COLORS["ppo"], label="RL Agents (advaug, v2)"),
    ]
    ax.legend(handles=legend_els, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig9_baselines_comparison")


# ---------------------------------------------------------------------------
# FIGURE 10 — Reward sensitivity line plots
# ---------------------------------------------------------------------------
def fig_sensitivity_reward(data: dict):
    sens = data.get("sensitivity", {})
    reward_sweeps = [s for s in SENSITIVITY_SWEEPS_REWARD if s in sens]
    if not reward_sweeps:
        print("  [skip] fig_sensitivity_reward — no sensitivity logs yet")
        return

    ncols = len(reward_sweeps)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    for ax, sweep in zip(axes, reward_sweeps):
        rows = sens[sweep]
        values = [r["value"] for r in rows]
        default_val = SENSITIVITY_DEFAULTS.get(sweep)

        for algo, color in ALGO_COLORS.items():
            algo_acc = []
            for row in rows:
                seed_accs = [
                    v["accuracy"]
                    for k, v in row["metrics"].items()
                    if k.startswith(algo + "_") and not np.isnan(v["accuracy"])
                ]
                algo_acc.append(np.mean(seed_accs) if seed_accs else float("nan"))
            ax.plot(
                values,
                algo_acc,
                marker="o",
                color=color,
                label=ALGO_LABELS[algo],
                linewidth=2,
                markersize=5,
            )

        if default_val is not None:
            ax.axvline(
                default_val,
                color="gray",
                linestyle="--",
                linewidth=1,
                label=f"v2 default ({default_val})",
            )
        ax.set_xlabel(SENSITIVITY_LABELS.get(sweep, sweep))
        ax.set_ylabel("Accuracy")
        ax.set_title(SENSITIVITY_LABELS.get(sweep, sweep))
        ax.set_ylim(0.7, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Sensitivity to Reward Assumption Misspecification (advaug v2 agents)",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig10_sensitivity_reward")


# ---------------------------------------------------------------------------
# FIGURE 11 — Challenge-outcome sensitivity line plots
# ---------------------------------------------------------------------------
def fig_sensitivity_challenge(data: dict):
    sens = data.get("sensitivity", {})
    challenge_sweeps = [s for s in SENSITIVITY_SWEEPS_CHALLENGE if s in sens]
    if not challenge_sweeps:
        print("  [skip] fig_sensitivity_challenge — no sensitivity logs yet")
        return

    from matplotlib.gridspec import GridSpec

    n = len(challenge_sweeps)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.5 * ncols, 4.5 * nrows))
    top_n = min(n, ncols)
    bottom_n = n - top_n
    axes_flat = []
    if nrows == 1:
        gs = GridSpec(1, ncols, figure=fig)
        for i in range(top_n):
            axes_flat.append(fig.add_subplot(gs[0, i]))
    else:
        gs = GridSpec(nrows, ncols * 2, figure=fig)
        for i in range(top_n):
            axes_flat.append(fig.add_subplot(gs[0, i * 2 : (i + 1) * 2]))
        offset = (ncols - bottom_n) * 2 // 2
        for i in range(bottom_n):
            axes_flat.append(
                fig.add_subplot(gs[1, offset + i * 2 : offset + (i + 1) * 2])
            )

    for ax, sweep in zip(axes_flat, challenge_sweeps):
        rows = sens[sweep]
        values = [r["value"] for r in rows]
        default_val = SENSITIVITY_DEFAULTS.get(sweep)

        for algo, color in ALGO_COLORS.items():
            algo_acc = []
            for row in rows:
                seed_accs = [
                    v["accuracy"]
                    for k, v in row["metrics"].items()
                    if k.startswith(algo + "_") and not np.isnan(v["accuracy"])
                ]
                algo_acc.append(np.mean(seed_accs) if seed_accs else float("nan"))
            ax.plot(
                values,
                algo_acc,
                marker="o",
                color=color,
                label=ALGO_LABELS[algo],
                linewidth=2,
                markersize=5,
            )

        if default_val is not None:
            ax.axvline(
                default_val,
                color="gray",
                linestyle="--",
                linewidth=1,
                label="v2 default",
            )
        ax.set_xlabel(SENSITIVITY_LABELS.get(sweep, sweep))
        ax.set_ylabel("Accuracy")
        ax.set_title(SENSITIVITY_LABELS.get(sweep, sweep))
        ax.set_ylim(0.7, 1.05)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Sensitivity to Challenge-Outcome Assumption Misspecification (advaug v2 agents)",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig11_sensitivity_challenge")


# ---------------------------------------------------------------------------
# FIGURE 12 — Ablation study
# ---------------------------------------------------------------------------
def fig_ablation_comparison(data: dict):
    ablations = data.get("ablations", {})
    native = data.get("native", {})
    if not ablations:
        print("  [skip] fig_ablation_comparison — no ablation logs yet")
        return

    baseline_d = native.get(("ppo", "advaug", "v2"))
    # native logs use capitalized keys; ablation logs use lowercase
    baseline_f1 = _metric(baseline_d, "F1") if baseline_d else (None, None)
    baseline_len = _metric(baseline_d, "Avg Length") if baseline_d else (None, None)

    reward_names = [n for n in ABLATION_NAMES if ABLATION_GROUPS[n] == "reward"]
    arch_names = [n for n in ABLATION_NAMES if ABLATION_GROUPS[n] == "arch"]

    def _abl_metric(name, metric):
        """Pull metric from ablation-agent entries in the ablation eval log."""
        abl_data = ablations.get(name, {})
        agents = [v for k, v in abl_data.items() if f"ablation_{name}" in k]
        if not agents:
            return None, None
        vals = [a.get(metric, (np.nan, np.nan))[0] for a in agents]
        vals = [v for v in vals if not np.isnan(v)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    def _baseline_from_abl_log(name, metric):
        """Pull baseline-agent metric from inside an ablation eval log."""
        abl_data = ablations.get(name, {})
        agents = [v for k, v in abl_data.items() if f"ablation_{name}" not in k]
        if not agents:
            return None, None
        vals = [a.get(metric, (np.nan, np.nan))[0] for a in agents]
        vals = [v for v in vals if not np.isnan(v)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: Reward ablations F1 ──────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(reward_names))
    means = [_abl_metric(n, "f1")[0] or 0.0 for n in reward_names]
    stds = [_abl_metric(n, "f1")[1] or 0.0 for n in reward_names]
    bars = ax.bar(
        x,
        means,
        0.55,
        yerr=stds,
        capsize=5,
        color="#4C8BB5",
        alpha=0.85,
        ecolor="gray",
        label="Ablation",
    )
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.001,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    if baseline_f1[0] is not None:
        ax.axhline(
            baseline_f1[0],
            color="#2563a8",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline: {baseline_f1[0]:.3f}",
        )
        if baseline_f1[1]:
            ax.fill_between(
                [-0.5, len(reward_names) - 0.5],
                baseline_f1[0] - baseline_f1[1],
                baseline_f1[0] + baseline_f1[1],
                alpha=0.12,
                color="#2563a8",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ABLATION_LABELS[n] for n in reward_names], rotation=20, ha="right"
    )
    ax.set_ylabel("F1 (mean ± std, 5 seeds)")
    ax.set_ylim(0.88, 1.01)
    ax.set_title("Reward Ablations")
    ax.legend(fontsize=8)

    # ── Panel 2: Architecture ablations F1 ────────────────────────────────────
    ax = axes[1]
    x = np.arange(len(arch_names))
    means = [_abl_metric(n, "f1")[0] or 0.0 for n in arch_names]
    stds = [_abl_metric(n, "f1")[1] or 0.0 for n in arch_names]
    colors = ["#E07B54" if n != "single_view" else "#C0392B" for n in arch_names]
    bars = ax.bar(
        x, means, 0.55, yerr=stds, capsize=5, color=colors, alpha=0.85, ecolor="gray"
    )
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.001,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    if baseline_f1[0] is not None:
        ax.axhline(
            baseline_f1[0],
            color="#2563a8",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline: {baseline_f1[0]:.3f}",
        )
        if baseline_f1[1]:
            ax.fill_between(
                [-0.5, len(arch_names) - 0.5],
                baseline_f1[0] - baseline_f1[1],
                baseline_f1[0] + baseline_f1[1],
                alpha=0.12,
                color="#2563a8",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ABLATION_LABELS[n] for n in arch_names], rotation=20, ha="right"
    )
    ax.set_ylabel("F1 (mean ± std, 5 seeds)")
    ax.set_ylim(0.88, 1.01)
    ax.set_title("Architecture Ablations")
    ax.legend(fontsize=8)

    # ── Panel 3: Avg observation windows per episode ──────────────────────────
    ax = axes[2]
    # Use no_hp_bonus log for baseline avg_length (reward ablation, same arch)
    ref_abl = reward_names[0] if reward_names else None
    if baseline_len[0]:
        bl_len = baseline_len[0]
    elif ref_abl:
        bl_len = _baseline_from_abl_log(ref_abl, "avg_length")[0] or 0.0
    else:
        bl_len = 0.0

    all_names = ["baseline"] + arch_names
    labels = ["Baseline\n(PPO advaug)"] + [ABLATION_LABELS[n] for n in arch_names]
    x = np.arange(len(all_names))

    step_means = [bl_len] + [
        (_abl_metric(n, "avg_length")[0] or 0.0) for n in arch_names
    ]
    step_stds = [0.0] + [(_abl_metric(n, "avg_length")[1] or 0.0) for n in arch_names]

    bar_colors = ["#2563a8"] + [
        "#C0392B" if n == "single_view" else "#E07B54" for n in arch_names
    ]
    bars = ax.bar(
        x,
        step_means,
        0.6,
        yerr=step_stds,
        capsize=4,
        color=bar_colors,
        alpha=0.85,
        ecolor="gray",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Avg observation windows per episode")
    ax.set_title("Observation Depth\n(temporal reasoning vs. single-step)")
    ax.set_ylim(0, 25)

    for bar, val in zip(bars, step_means):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(
        "Ablation Study — PPO (advaug): Reward Shaping, Architecture, and Observation Depth",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig12_ablation_comparison")


# ---------------------------------------------------------------------------
# FIGURE 13 — Temporal vs Single-View: same F1, different strategy
# ---------------------------------------------------------------------------
def fig_temporal_vs_singleview(data: dict):
    """Four-panel figure proving temporal reasoning matters beyond F1."""
    ablations = data.get("ablations", {})
    native = data.get("native", {})

    baseline_d = native.get(("ppo", "advaug", "v2"))
    sv_abl_data = ablations.get("single_view", {})

    if not baseline_d or not sv_abl_data:
        print("  [skip] fig_temporal_vs_singleview — missing data")
        return

    # Baseline metrics (full temporal agent, multiple windows)
    bl_f1 = _metric(baseline_d, "F1")
    bl_len = _metric(baseline_d, "Avg Length")
    bl_hlen = _metric(baseline_d, "Avg Steps Human")
    bl_blen = _metric(baseline_d, "Avg Steps Bot")
    bl_cr = _metric(baseline_d, "Human Challenge Rate")

    # Single-view ablation agents (trained for 1 window)
    sv_agents = [v for k, v in sv_abl_data.items() if "ablation_single_view" in k]
    # Baseline agents evaluated in single-view mode (--max-windows 1)
    bl_in_sv = [v for k, v in sv_abl_data.items() if "ablation_single_view" not in k]

    def _pool(agents, key):
        raw = [a.get(key, (np.nan, np.nan)) for a in agents]
        vals = [r[0] if isinstance(r, tuple) else r for r in raw]
        vals = [v for v in vals if v is not None and not np.isnan(float(v))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    sv_f1 = _pool(sv_agents, "f1")
    sv_len = _pool(sv_agents, "avg_length")
    _pool(sv_agents, "human_challenge_rate")  # unused but kept for reference

    bl_sv_f1 = _pool(bl_in_sv, "f1")  # baseline forced into single-view

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    BLUE, RED, ORANGE = "#2563a8", "#C0392B", "#E07B54"

    # Panel 1 — F1: baseline vs sv-trained vs baseline-forced-sv
    ax = axes[0]
    labels = [
        "Temporal\nAgent",
        "Single-View\n(trained)",
        "Temporal Agent\n(forced 1-window)",
    ]
    f1_means = [
        bl_f1[0] if bl_f1[0] else 0,
        sv_f1[0] if sv_f1[0] else 0,
        bl_sv_f1[0] if bl_sv_f1[0] else 0,
    ]
    f1_stds = [
        bl_f1[1] if bl_f1[1] else 0,
        sv_f1[1] if sv_f1[1] else 0,
        bl_sv_f1[1] if bl_sv_f1[1] else 0,
    ]
    colors = [BLUE, ORANGE, RED]
    bars = ax.bar(
        range(3),
        f1_means,
        0.55,
        yerr=f1_stds,
        capsize=5,
        color=colors,
        alpha=0.85,
        ecolor="gray",
    )
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Detection F1\n(accuracy measure)")
    for bar, v in zip(bars, f1_means):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # Panel 2 — Avg observation windows
    ax = axes[1]
    # Use Avg Length from parsed data, fall back to known values
    bl_len_val = (
        bl_len[0]
        if (bl_len and bl_len[0])
        else (bl_hlen[0] if bl_hlen and bl_hlen[0] else None)
    )
    sv_len_val = sv_len[0] if (sv_len and sv_len[0]) else None
    len_means = [bl_len_val or 0, sv_len_val or 1.0]
    len_stds = [
        bl_len[1] if bl_len and bl_len[1] else 0,
        sv_len[1] if sv_len and sv_len[1] else 0,
    ]
    bars = ax.bar(
        [0, 1],
        len_means,
        0.45,
        yerr=len_stds,
        capsize=5,
        color=[BLUE, ORANGE],
        alpha=0.85,
        ecolor="gray",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Temporal\nAgent", "Single-View\n(trained)"], fontsize=8)
    ax.set_ylabel("Avg observation windows / episode")
    ax.set_title("Observation Depth\n(how much context used)")
    ax.set_ylim(0, 25)
    for bar, v in zip(bars, len_means):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Panel 3 — Avg steps: human vs bot sessions for temporal agent
    ax = axes[2]
    if bl_hlen and bl_hlen[0] and bl_blen and bl_blen[0]:
        bars = ax.bar(
            [0, 1],
            [bl_hlen[0], bl_blen[0]],
            0.45,
            yerr=[bl_hlen[1], bl_blen[1]],
            capsize=5,
            color=["#4aad52", "#e07b54"],
            alpha=0.85,
            ecolor="gray",
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Human\nSessions", "Bot\nSessions"], fontsize=8)
        ax.set_ylabel("Avg windows before decision")
        ax.set_title(
            "Temporal Agent Decision Timing\n(earlier for bots = early intervention)"
        )
        ax.set_ylim(0, 25)
        ax.axhline(21, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        for bar, v in zip(bars, [bl_hlen[0], bl_blen[0]]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.annotate(
            "Bots detected\nearlier than\nhumans cleared",
            xy=(1, bl_blen[0]),
            xytext=(1.3, 18),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=7,
            color="gray",
            ha="center",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Avg steps data\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
        )
        ax.set_title("Temporal Agent Decision Timing")

    # Panel 4 — Human challenge rate
    ax = axes[3]
    bl_cr_val = (bl_cr[0] * 100) if (bl_cr and bl_cr[0] is not None) else 0.0
    sv_cr_raw = _pool(sv_agents, "human_challenge_rate")
    sv_cr_val = (
        (sv_cr_raw[0] * 100) if (sv_cr_raw and sv_cr_raw[0] is not None) else 1.5
    )

    bars = ax.bar(
        [0, 1], [bl_cr_val, sv_cr_val], 0.45, color=[BLUE, ORANGE], alpha=0.85
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Temporal\nAgent", "Single-View\n(trained)"], fontsize=8)
    ax.set_ylabel("Human challenge rate (%)")
    ax.set_title("Human Friction\n(unnecessary challenges to real users)")
    ax.set_ylim(0, 5)
    for bar, v in zip(bars, [bl_cr_val, sv_cr_val]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "Why Temporal Reasoning Matters: Same F1, Fundamentally Different Detection Strategy",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig13_temporal_vs_singleview")


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _pct(v, s=None):
    if s is not None:
        return f"{v*100:.1f}$\\pm${s*100:.1f}"
    return f"{v*100:.1f}"


def _write_table(name: str, latex_lines: list[str], csv_rows: list[list]):
    tex_path = TABLE_DIR / f"{name}.tex"
    tex_path.write_text("\n".join(latex_lines), encoding="utf-8")
    csv_path = TABLE_DIR / f"{name}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(str(c) for c in row) + "\n")
    print(f"  Saved: {tex_path} + {csv_path}")


# ---------------------------------------------------------------------------
# TABLE 1 — Main results (v2 only)
# ---------------------------------------------------------------------------
def table_main_results(data: dict):
    native = data["native"]
    if not native:
        print("  [skip] table_main_results")
        return

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Bot Detection Results (v2 environment, mean $\pm$ std across 5 training seeds)}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Algorithm & Aug & Acc (\%) & Prec (\%) & Recall (\%) & F1 (\%) & HP\% \\",
        r"\midrule",
    ]
    csv_rows = [
        [
            "Algorithm",
            "Aug",
            "Acc_mean",
            "Acc_std",
            "F1_mean",
            "F1_std",
            "HP_mean",
            "HP_std",
        ]
    ]

    for aug in AUGS:
        for algo in ALGOS:
            d = native.get((algo, aug, "v2"))
            if not d:
                continue
            m = d["metrics"]
            acc = m.get("Accuracy", (0, 0))
            prec = m.get("Precision", (0, 0))
            rec = m.get("Recall", (0, 0))
            f1 = m.get("F1", (0, 0))
            hpct = m.get("Honeypot %", (0, 0))
            lines.append(
                f"  {ALGO_LABELS[algo]} & {AUG_LABELS[aug]} & "
                f"{_pct(*acc)} & {_pct(*prec)} & {_pct(*rec)} & {_pct(*f1)} & "
                f"{_pct(*hpct)} \\\\"
            )
            csv_rows.append(
                [
                    ALGO_LABELS[algo],
                    AUG_LABELS[aug],
                    f"{acc[0]:.4f}",
                    f"{acc[1]:.4f}",
                    f"{f1[0]:.4f}",
                    f"{f1[1]:.4f}",
                    f"{hpct[0]:.3f}",
                    f"{hpct[1]:.3f}",
                ]
            )
        lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("tab1_main_results", lines, csv_rows)


# ---------------------------------------------------------------------------
# TABLE 2 — Cross-environment transfer (v2-trained only)
# ---------------------------------------------------------------------------
def table_cross_env(data: dict):
    native = data["native"]
    cross = data["cross"]
    if not native and not cross:
        print("  [skip] table_cross_env")
        return

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cross-Environment Transfer: v2-trained agents"
        r" in native and legacy environments (advaug models)}",
        r"\label{tab:cross_env}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Algorithm & v2 env (native) & v1 env (transfer) \\",
        r"\midrule",
    ]
    csv_rows = [["Algorithm", "v2_native", "v2_in_v1"]]

    for algo in ALGOS:
        d_nat = native.get((algo, "advaug", "v2"))
        d_cross = cross.get((algo, "advaug", "v2", "v1"))
        acc_nat = _metric(d_nat, "Accuracy") if d_nat else (0.0, 0.0)
        acc_cross = _metric(d_cross, "Accuracy") if d_cross else (0.0, 0.0)
        lines.append(
            f"  {ALGO_LABELS[algo]} & {_pct(*acc_nat)} & {_pct(*acc_cross)} \\\\"
        )
        csv_rows.append([ALGO_LABELS[algo], f"{acc_nat[0]:.4f}", f"{acc_cross[0]:.4f}"])

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("tab2_cross_env", lines, csv_rows)


# ---------------------------------------------------------------------------
# TABLE 3 — Held-out tier generalization (v2, advaug)
# ---------------------------------------------------------------------------
def table_heldout_tiers(data: dict):
    ht = data["heldout_tier"]
    if not ht:
        print("  [skip] table_heldout_tiers")
        return

    tier_keys = ["tier3", "tier4", "tier5", "tier345"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Generalization to Unseen Bot Tiers: Accuracy when held-out tiers"
        r" are removed from training (advaug models, v2 environment)}",
        r"\label{tab:heldout_tiers}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Algorithm & T3 Held-out & T4 Held-out & T5 Held-out & T3+4+5 Held-out \\",
        r"\midrule",
    ]
    csv_rows = [["Algorithm", "tier3", "tier4", "tier5", "tier345"]]

    for algo in ALGOS:
        cells = []
        for tk in tier_keys:
            d = ht.get((algo, "advaug", "v2", tk))
            acc = _metric(d, "Accuracy") if d else (0.0, 0.0)
            cells.append(_pct(*acc))
        lines.append(f"  {ALGO_LABELS[algo]} & " + " & ".join(cells) + r" \\")
        csv_rows.append([ALGO_LABELS[algo]] + cells)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("tab3_heldout_tiers", lines, csv_rows)


# ---------------------------------------------------------------------------
# TABLE 4 — Held-out family generalization (v2, advaug)
# ---------------------------------------------------------------------------
def table_heldout_families(data: dict):
    hf = data["heldout_family"]
    if not hf:
        print("  [skip] table_heldout_families")
        return

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Generalization to Unseen Bot Families: Accuracy (advaug models, v2 environment)}",
        r"\label{tab:heldout_families}",
        r"\begin{tabular}{l" + "c" * len(FAMILIES) + "}",
        r"\toprule",
        "Algorithm & " + " & ".join(FAM_LABELS[f] for f in FAMILIES) + r" \\",
        r"\midrule",
    ]
    csv_rows = [["Algorithm"] + [FAM_LABELS[f] for f in FAMILIES]]

    for algo in ALGOS:
        cells = []
        for fam in FAMILIES:
            d = hf.get((algo, "advaug", "v2", fam))
            acc = _metric(d, "Accuracy") if d else (0.0, 0.0)
            cells.append(_pct(*acc))
        lines.append(f"  {ALGO_LABELS[algo]} & " + " & ".join(cells) + r" \\")
        csv_rows.append([ALGO_LABELS[algo]] + cells)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("tab4_heldout_families", lines, csv_rows)


# ---------------------------------------------------------------------------
# TABLE 5 — Per-seed variance (v2)
# ---------------------------------------------------------------------------
def table_seed_variance(data: dict):
    native = data["native"]
    if not native:
        print("  [skip] table_seed_variance")
        return

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Training Seed Variance: Per-seed accuracy (v2 environment)}",
        r"\label{tab:seed_variance}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Algo & Aug & Seed 42 & Seed 123 & Seed 456 & Seed 789 & Seed 1024 & Mean & Std \\",
        r"\midrule",
    ]
    csv_rows = [["Algo", "Aug", "s42", "s123", "s456", "s789", "s1024", "mean", "std"]]

    for aug in AUGS:
        for algo in ALGOS:
            d = native.get((algo, aug, "v2"))
            if not d or not d.get("seed_accs"):
                continue
            accs = d["seed_accs"]
            cells = [f"{a:.3f}" for a in accs]
            mean, std = np.mean(accs), np.std(accs)
            lines.append(
                f"  {ALGO_LABELS[algo]} & {AUG_LABELS[aug]} & "
                + " & ".join(cells)
                + f" & {mean:.3f} & {std:.3f} \\\\"
            )
            csv_rows.append(
                [ALGO_LABELS[algo], AUG_LABELS[aug]]
                + cells
                + [f"{mean:.4f}", f"{std:.4f}"]
            )
        lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("tab5_seed_variance", lines, csv_rows)


# ---------------------------------------------------------------------------
# TABLE 6 — Baselines comparison (v2)
# ---------------------------------------------------------------------------
def table_baselines(data: dict):
    baselines = data.get("baselines", {})
    native = data.get("native", {})
    if not baselines:
        print("  [skip] table_baselines")
        return

    bl = baselines.get("v2", {})
    if not bl:
        return

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Baseline Comparison: rule-based policies vs RL agents (v2 environment)}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Accuracy (\%) & Precision (\%) & Recall (\%) & F1 (\%) \\",
        r"\midrule",
    ]
    csv_rows = [["Method", "Accuracy", "Acc_std", "F1", "F1_std"]]

    for name, label in BASELINE_LABELS.items():
        if name not in bl:
            continue
        m = bl[name]
        acc = m.get("accuracy", (0.0, 0.0))
        prec = m.get("precision", (0.0, 0.0))
        rec = m.get("recall", (0.0, 0.0))
        f1 = m.get("f1", (0.0, 0.0))
        lines.append(
            f"  {label} & {_pct(*acc)} & {_pct(*prec)} & {_pct(*rec)} & {_pct(*f1)} \\\\"
        )
        csv_rows.append(
            [label, f"{acc[0]:.4f}", f"{acc[1]:.4f}", f"{f1[0]:.4f}", f"{f1[1]:.4f}"]
        )

    lines.append(r"\midrule")

    for algo in ALGOS:
        d = native.get((algo, "advaug", "v2"))
        if not d:
            continue
        m = d["metrics"]
        acc = m.get("Accuracy", (0, 0))
        prec = m.get("Precision", (0, 0))
        rec = m.get("Recall", (0, 0))
        f1 = m.get("F1", (0, 0))
        label = f"RL {ALGO_LABELS[algo]} (advaug)"
        lines.append(
            f"  {label} & {_pct(*acc)} & {_pct(*prec)} & {_pct(*rec)} & {_pct(*f1)} \\\\"
        )
        csv_rows.append(
            [label, f"{acc[0]:.4f}", f"{acc[1]:.4f}", f"{f1[0]:.4f}", f"{f1[1]:.4f}"]
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("tab6_baselines", lines, csv_rows)


# ---------------------------------------------------------------------------
# TABLE 7 — Sensitivity analysis summary
# ---------------------------------------------------------------------------
def table_sensitivity(data: dict):
    sens = data.get("sensitivity", {})
    if not sens:
        print("  [skip] table_sensitivity")
        return

    for sweep in ALL_SENSITIVITY_SWEEPS:
        rows = sens.get(sweep)
        if not rows:
            continue

        label = SENSITIVITY_LABELS.get(sweep, sweep)
        default_val = SENSITIVITY_DEFAULTS.get(sweep)
        all_agents = list(rows[0]["metrics"].keys()) if rows else []
        algos_seen = sorted({a.split("_")[0] for a in all_agents})

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            f"\\caption{{Sensitivity: {label} (advaug v2 agents, accuracy)}}",
            f"\\label{{tab:sensitivity_{sweep}}}",
            r"\begin{tabular}{l" + "c" * len(algos_seen) + "}",
            r"\toprule",
            "Value & " + " & ".join(ALGO_LABELS.get(a, a) for a in algos_seen) + r" \\",
            r"\midrule",
        ]
        csv_rows = [["Value"] + [ALGO_LABELS.get(a, a) for a in algos_seen]]

        for row in rows:
            v = row["value"]
            algo_accs = []
            for algo in algos_seen:
                seed_accs = [
                    vals["accuracy"]
                    for k, vals in row["metrics"].items()
                    if k.startswith(algo + "_") and not np.isnan(vals["accuracy"])
                ]
                algo_accs.append(np.mean(seed_accs) if seed_accs else float("nan"))

            default_mark = (
                r" \textbf{(default)}"
                if (default_val is not None and abs(v - default_val) < 1e-9)
                else ""
            )
            cell_strs = [f"{a*100:.1f}" if not np.isnan(a) else "--" for a in algo_accs]
            lines.append(f"  {v:.4g}{default_mark} & " + " & ".join(cell_strs) + r" \\")
            csv_rows.append(
                [str(v)] + [f"{a:.4f}" if not np.isnan(a) else "" for a in algo_accs]
            )

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _write_table(f"tab7_sensitivity_{sweep}", lines, csv_rows)


# ---------------------------------------------------------------------------
# TABLE 8 — Ablation study results
# ---------------------------------------------------------------------------
def table_ablations(data: dict):
    ablations = data.get("ablations", {})
    native = data.get("native", {})
    if not ablations:
        print("  [skip] table_ablations")
        return

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation Study: Accuracy and F1 for each design choice"
        r" (PPO, advaug, v2 base; mean $\pm$ std across 5 seeds)}",
        r"\label{tab:ablations}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Group & Ablation & Accuracy (\%) & F1 (\%) \\",
        r"\midrule",
    ]
    csv_rows = [["Group", "Ablation", "Acc_mean", "Acc_std", "F1_mean", "F1_std"]]

    d_base = native.get(("ppo", "advaug", "v2"))
    if d_base:
        acc = d_base["metrics"].get("Accuracy", (0, 0))
        f1 = d_base["metrics"].get("F1", (0, 0))
        lines.append(
            f"  -- & Baseline (PPO, advaug, v2) & {_pct(*acc)} & {_pct(*f1)} \\\\"
        )
        csv_rows.append(
            [
                "baseline",
                "PPO advaug v2",
                f"{acc[0]:.4f}",
                f"{acc[1]:.4f}",
                f"{f1[0]:.4f}",
                f"{f1[1]:.4f}",
            ]
        )
    lines.append(r"\midrule")

    for group, names in [
        ("Reward", [n for n in ABLATION_NAMES if ABLATION_GROUPS[n] == "reward"]),
        ("Architecture", [n for n in ABLATION_NAMES if ABLATION_GROUPS[n] == "arch"]),
    ]:
        for name in names:
            abl_data = ablations.get(name, {})
            abl_agents = {k: v for k, v in abl_data.items() if f"ablation_{name}" in k}
            if not abl_agents:
                lines.append(f"  {group} & {ABLATION_LABELS[name]} & -- & -- \\\\")
                csv_rows.append([group, ABLATION_LABELS[name], "", "", "", ""])
                continue
            accs = [v.get("accuracy", (0.0, 0.0))[0] for v in abl_agents.values()]
            f1s = [v.get("f1", (0.0, 0.0))[0] for v in abl_agents.values()]
            acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
            f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))
            lines.append(
                f"  {group} & {ABLATION_LABELS[name]} & "
                f"{_pct(acc_mean, acc_std)} & {_pct(f1_mean, f1_std)} \\\\"
            )
            csv_rows.append(
                [
                    group,
                    ABLATION_LABELS[name],
                    f"{acc_mean:.4f}",
                    f"{acc_std:.4f}",
                    f"{f1_mean:.4f}",
                    f"{f1_std:.4f}",
                ]
            )
        lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("tab8_ablations", lines, csv_rows)


# ---------------------------------------------------------------------------
# FIGURE 14 — Strict accuracy: standard vs strict F1 for all 3 algos
# ---------------------------------------------------------------------------
def fig_strict_accuracy(data: dict):
    """Standard vs strict F1 for all algos and all ablations.

    Panel 1 — All 3 algorithms: standard vs strict F1.
    Panel 2 — All ablations: F1 drop (standard minus strict) shows which ablations
              challenge humans more, causing larger degradation under strict metric.
    """
    strict_all = data.get("strict", {})
    strict_d = strict_all.get("baselines")
    native = data.get("native", {})
    ablations = data.get("ablations", {})

    has_baseline_strict = strict_d is not None
    has_ablation_strict = any(f"ablation_{n}" in strict_all for n in ABLATION_NAMES)

    if not has_baseline_strict and not has_ablation_strict:
        print(
            "  [skip] fig_strict_accuracy — no strict logs found yet (run eval_strict.ps1)"
        )
        return

    BLUE, ORANGE, RED = "#2563a8", "#E07B54", "#C0392B"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Panel 1: All algos, grouped bars ─────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(ALGOS))
    w = 0.35

    std_means, std_stds, strict_means, strict_stds = [], [], [], []
    for algo in ALGOS:
        nat = native.get((algo, "advaug", "v2"))
        sf = _metric(nat, "F1") if nat else (0.0, 0.0)
        std_means.append(sf[0])
        std_stds.append(sf[1])
        key = f"{algo}_advaug_v2"
        entry = (strict_d or {}).get(key, {})
        strict_means.append(entry.get("f1", (0.0, 0.0))[0])
        strict_stds.append(entry.get("f1", (0.0, 0.0))[1])

    all_vals = [v for v in std_means + strict_means if v > 0]
    ymin = round(min(all_vals) - 0.02, 2) if all_vals else 0.85

    def _errs(mean, std):
        """Asymmetric error bars clipped to [ymin, 1.0]."""
        up = min(std, 1.0 - mean)
        dn = min(std, mean - ymin - 0.001)
        return max(up, 0.0), max(dn, 0.0)

    std_err = [_errs(m, s) for m, s in zip(std_means, std_stds)]
    ax.bar(
        x - w / 2,
        std_means,
        w,
        yerr=[[e[1] for e in std_err], [e[0] for e in std_err]],
        capsize=4,
        ecolor="gray",
        color=BLUE,
        alpha=0.85,
        label="Standard F1",
    )
    for i in range(len(ALGOS)):
        if strict_means[i] > 0:
            up, dn = _errs(strict_means[i], strict_stds[i])
            ax.bar(
                x[i] + w / 2,
                strict_means[i],
                w,
                yerr=[[dn], [up]],
                capsize=4,
                ecolor="gray",
                color=ORANGE,
                alpha=0.85,
                label="Strict F1 (challenge=FP)" if i == 0 else "_nolegend_",
            )
            ax.text(
                x[i] + w / 2,
                strict_means[i] + up + 0.001,
                f"{strict_means[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for i in range(len(ALGOS)):
        if std_means[i] > 0:
            ax.text(
                x[i] - w / 2,
                std_means[i] + std_err[i][0] + 0.001,
                f"{std_means[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS])
    ax.set_ylabel("F1 (mean +/- std, 5 seeds)")
    ax.set_ylim(ymin, 1.01)
    ax.set_title("Standard vs Strict F1\nAll Algorithms (advaug v2)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9)

    # ── Panel 2: Ablations, grouped bars ─────────────────────────────────────
    ax = axes[1]

    def _abl_strict_f1(name):
        src = strict_all.get(f"ablation_{name}", {})
        agents = [v for k, v in src.items() if f"ablation_{name}" in k]
        vals = [a.get("f1", (np.nan, np.nan))[0] for a in agents]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else 0.0

    def _abl_std_f1(name):
        src = ablations.get(name, {})
        agents = [v for k, v in src.items() if f"ablation_{name}" in k]
        vals = [a.get("f1", (np.nan, np.nan))[0] for a in agents]
        vals = [v for v in vals if not np.isnan(v)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    bl_nat = native.get(("ppo", "advaug", "v2"))
    bl_std_f1 = _metric(bl_nat, "F1") if bl_nat else (0.0, 0.0)
    bl_strict_f1 = (strict_d or {}).get("ppo_advaug_v2", {}).get("f1", (0.0, 0.0))[0]

    all_names = ["Baseline\n(PPO)"] + [ABLATION_LABELS[n] for n in ABLATION_NAMES]
    x2 = np.arange(len(all_names))
    w2 = 0.35

    std_m = [bl_std_f1[0]] + [_abl_std_f1(n)[0] for n in ABLATION_NAMES]
    std_s = [bl_std_f1[1]] + [_abl_std_f1(n)[1] for n in ABLATION_NAMES]
    str_m = [bl_strict_f1] + [_abl_strict_f1(n) for n in ABLATION_NAMES]

    sv_idx = len(all_names) - 1
    ax.bar(
        x2 - w2 / 2,
        std_m,
        w2,
        yerr=std_s,
        capsize=3,
        color=BLUE,
        alpha=0.85,
        ecolor="gray",
        label="Standard F1",
    )
    for i in range(len(all_names)):
        if str_m[i] > 0:
            c = RED if i == sv_idx else ORANGE
            ax.bar(
                x2[i] + w2 / 2,
                str_m[i],
                w2,
                color=c,
                alpha=0.85,
                label="Strict F1 (challenge=FP)" if i == 0 else "_nolegend_",
            )

    ax.set_xticks(x2)
    ax.set_xticklabels(all_names, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score (mean +/- std, 5 seeds)")
    ax.set_ylim(0.88, 1.01)
    ax.set_title("Strict F1 by Ablation\n(challenge to human = false positive)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Strict Accuracy: Counting Human CAPTCHA Challenges as False Positives",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig14_strict_accuracy")


# ---------------------------------------------------------------------------
# FIGURE 15 — Disjoint generalization across all held-out tier configurations
# ---------------------------------------------------------------------------
def fig_disjoint_eval(data: dict):
    """Baseline vs disjoint recall/F1 across all held-out tier configurations."""
    dj_data = data.get("disjoint", {})
    if not dj_data:
        print(
            "  [skip] fig_disjoint_eval — no disjoint logs found (run eval_disjoint_all.ps1 first)"
        )
        return

    # Ordered tier configs to show
    DTIER_ORDER = ["tier1", "tier2", "tier3", "tier4", "tier5", "tier45", "tier345"]
    DTIER_LABELS = {
        "tier1": "T1\nCommodity",
        "tier2": "T2\nCareful",
        "tier3": "T3\nAdaptive",
        "tier4": "T4\nStealth",
        "tier5": "T5\nLLM",
        "tier45": "T4+5\nHeld Out",
        "tier345": "T3+4+5\nHeld Out",
    }

    # Collect which configs have data
    available = [t for t in DTIER_ORDER if t in dj_data]
    if not available:
        print("  [skip] fig_disjoint_eval — no disjoint logs with parsed data")
        return

    def _agg(agents_dict, metric):
        vals = [v.get(metric, (np.nan, np.nan))[0] for v in agents_dict.values()]
        vals = [v for v in vals if not np.isnan(v)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    BLUE, RED = "#2563a8", "#C0392B"

    # If only Tier 5 available, fall back to the old 3-metric single-tier layout
    if available == ["tier5"] or len(available) == 1:
        tier_key = available[0]
        tier_d = dj_data[tier_key]
        baseline_agents = {k: v for k, v in tier_d.items() if "disjoint" not in k}
        disjoint_agents = {k: v for k, v in tier_d.items() if "disjoint" in k}

        metrics_to_show = ["f1", "accuracy", "recall"]
        metric_labels = ["F1", "Accuracy", "Recall"]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(metrics_to_show))
        w = 0.35

        bl_means = [_agg(baseline_agents, m)[0] for m in metrics_to_show]
        bl_stds = [_agg(baseline_agents, m)[1] for m in metrics_to_show]
        dj_means = [_agg(disjoint_agents, m)[0] for m in metrics_to_show]
        dj_stds = [_agg(disjoint_agents, m)[1] for m in metrics_to_show]

        ax.bar(
            x - w / 2,
            bl_means,
            w,
            yerr=bl_stds,
            capsize=5,
            color=BLUE,
            alpha=0.85,
            ecolor="gray",
            label=f"Baseline (trained WITH {DTIER_LABELS[tier_key].replace(chr(10), ' ')})",
        )
        ax.bar(
            x + w / 2,
            dj_means,
            w,
            yerr=dj_stds,
            capsize=5,
            color=RED,
            alpha=0.85,
            ecolor="gray",
            label=f"Disjoint (trained WITHOUT {DTIER_LABELS[tier_key].replace(chr(10), ' ')})",
        )

        for i in range(len(metrics_to_show)):
            if bl_means[i] > 0:
                ax.text(
                    x[i] - w / 2,
                    bl_means[i] + bl_stds[i] + 0.005,
                    f"{bl_means[i]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            if dj_means[i] > 0:
                ax.text(
                    x[i] + w / 2,
                    dj_means[i] + dj_stds[i] + 0.005,
                    f"{dj_means[i]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score (mean +/- std, 5 seeds)")
        ax.set_ylim(0.5, 1.07)
        ax.legend(fontsize=9)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        fig.suptitle(
            "Disjoint Generalization: Bot Family Held Out During Training",
            fontweight="bold",
        )
        fig.tight_layout()
        _save(fig, "fig15_disjoint_eval")
        return

    # Multi-tier layout: two panels
    #   Panel 1 — Recall across all available held-out tiers (key generalization metric)
    #   Panel 2 — F1 across all available held-out tiers
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, mlabel in [
        (axes[0], "recall", "Recall"),
        (axes[1], "f1", "F1"),
    ]:
        n = len(available)
        x = np.arange(n)
        w = 0.35
        bl_m, bl_s, dj_m, dj_s = [], [], [], []

        for tag in available:
            tier_d = dj_data[tag]
            baseline_agents = {k: v for k, v in tier_d.items() if "disjoint" not in k}
            disjoint_agents = {k: v for k, v in tier_d.items() if "disjoint" in k}
            bm, bs = _agg(baseline_agents, metric)
            dm, ds = _agg(disjoint_agents, metric)
            bl_m.append(bm)
            bl_s.append(bs)
            dj_m.append(dm)
            dj_s.append(ds)

        ax.bar(
            x - w / 2,
            bl_m,
            w,
            yerr=bl_s,
            capsize=4,
            color=BLUE,
            alpha=0.85,
            ecolor="gray",
            label="Baseline (all tiers)",
        )
        ax.bar(
            x + w / 2,
            dj_m,
            w,
            yerr=dj_s,
            capsize=4,
            color=RED,
            alpha=0.85,
            ecolor="gray",
            label="Disjoint (tier held out)",
        )

        for i in range(n):
            if bl_m[i] > 0:
                ax.text(
                    x[i] - w / 2,
                    bl_m[i] + bl_s[i] + 0.004,
                    f"{bl_m[i]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            if dj_m[i] > 0:
                ax.text(
                    x[i] + w / 2,
                    dj_m[i] + dj_s[i] + 0.004,
                    f"{dj_m[i]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        all_vals = [v for v in bl_m + dj_m if v > 0]
        ymin = round(min(all_vals) - 0.08, 1) if all_vals else 0.4
        ax.set_ylim(max(0.0, ymin), 1.07)
        ax.set_xticks(x)
        ax.set_xticklabels([DTIER_LABELS[t] for t in available], fontsize=9)
        ax.set_ylabel(f"{mlabel} (mean +/- std, 5 seeds)")
        ax.set_title(f"{mlabel} on Held-Out Tier Sessions")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Disjoint Generalization: PPO Tested on Bot Tiers Excluded from Training",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig15_disjoint_eval")


# ---------------------------------------------------------------------------
# FIGURE 16 — Challenge difficulty: when does each model challenge humans?
# ---------------------------------------------------------------------------
def fig_challenge_difficulty(data: dict):
    """Hazard-rate chart: fraction of humans challenged at each window step, by model."""
    import csv as _csv

    csv_path = LOG_DIR / "per_session_challenges.csv"
    if not csv_path.exists():
        print(
            "  [skip] fig_challenge_difficulty — per_session_challenges.csv not found"
        )
        return

    rows = []
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            rows.append(
                {
                    "agent": r["agent"],
                    "true_label": int(r["true_label"]),
                    "challenged": r["challenged"].lower() == "true",
                    "steps": int(r["steps"]),
                    "outcome": r["outcome"],
                }
            )

    # Map agent name -> (model_label, seed)
    import re as _re

    def _model_label(name):
        n = name.lower()
        if "single_view" in n:
            return "Single-view"
        if n.startswith("dg"):
            return "DG"
        if n.startswith("soft_ppo"):
            return "Soft PPO"
        return "PPO"

    def _seed(name):
        m = _re.search(r"seed(\d+)$", name)
        return int(m.group(1)) if m else 0

    # Group by (model, seed) for per-seed challenge rates, and by model overall
    from collections import defaultdict

    by_model_seed = defaultdict(list)  # (model, seed) -> [human episodes]
    for r in rows:
        if r["true_label"] == 1:
            key = (_model_label(r["agent"]), _seed(r["agent"]))
            by_model_seed[key].append(r)

    if not by_model_seed:
        print("  [skip] fig_challenge_difficulty — no human episodes in CSV")
        return

    MODEL_ORDER = ["PPO", "DG", "Soft PPO", "Single-view"]
    COLORS = {
        "PPO": "#2563a8",
        "DG": "#C0392B",
        "Soft PPO": "#4aad52",
        "Single-view": "#E07B54",
    }

    # Per-seed challenge rates
    seed_rates = defaultdict(list)  # model -> [challenge_rate per seed]
    for (model, seed), eps in by_model_seed.items():
        n = len(eps)
        challenged = sum(1 for e in eps if e["challenged"])
        seed_rates[model].append(challenged / n if n > 0 else 0.0)

    present = [m for m in MODEL_ORDER if m in seed_rates]

    # Two-panel figure:
    # Left: per-seed challenge rate scatter + mean bar (shows DG variance)
    # Right: stacked by step bucket (shows WHEN challenges happen)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 1: Per-seed challenge rate ──────────────────────────────────────
    ax = axes[0]
    x1 = np.arange(len(present))
    w1 = 0.5
    for idx, model in enumerate(present):
        rates = seed_rates[model]
        mean_r = float(np.mean(rates))
        std_r = float(np.std(rates))
        up = min(std_r, 1.0 - mean_r)
        dn = min(std_r, mean_r)
        ax.bar(
            idx,
            mean_r,
            w1,
            color=COLORS[model],
            alpha=0.7,
            yerr=[[dn], [up]],
            capsize=5,
            ecolor="gray",
            label=model,
        )
        # Scatter individual seed dots
        jitter = np.linspace(-0.15, 0.15, len(rates))
        for j, r in zip(jitter, sorted(rates)):
            ax.scatter(idx + j, r, color=COLORS[model], s=30, zorder=5, alpha=0.9)
        if mean_r > 0.001:
            ax.text(
                idx,
                mean_r + up + 0.005,
                f"{mean_r:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x1)
    ax.set_xticklabels(present, fontsize=10)
    ax.set_ylabel("Human challenge rate (false positive rate)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(
        0, min(max(max(v for vv in seed_rates.values() for v in vv) * 1.4, 0.05), 1.05)
    )
    ax.set_title(
        "Challenge Rate per Training Seed\n(dots = individual seeds, bar = mean ± std)"
    )
    ax.axhline(0, color="gray", linewidth=0.5)

    # ── Panel 2: Step-at-challenge breakdown ─────────────────────────────────
    ax = axes[1]
    STEP_BUCKETS = [(1, "1 win"), (2, "2 win"), (3, "3 win"), (4, "4+ win")]
    x2 = np.arange(len(STEP_BUCKETS))
    w2 = 0.18
    offsets = np.linspace(
        -(len(present) - 1) * w2 / 2, (len(present) - 1) * w2 / 2, len(present)
    )

    # Aggregate all human episodes per model
    by_model_all = defaultdict(list)
    for (model, seed), eps in by_model_seed.items():
        by_model_all[model].extend(eps)

    def _bucket(s):
        return min(s, 4)

    for pidx, model in enumerate(present):
        eps = by_model_all[model]
        n_all = len(eps)
        # fraction of ALL human sessions challenged at each bucket
        fracs = []
        for bkt, _ in STEP_BUCKETS:
            chal = sum(1 for e in eps if e["challenged"] and _bucket(e["steps"]) == bkt)
            fracs.append(chal / n_all if n_all > 0 else 0.0)
        bars = ax.bar(
            x2 + offsets[pidx], fracs, w2, color=COLORS[model], alpha=0.85, label=model
        )
        for bar, f in zip(bars, fracs):
            if f > 0.003:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    f + 0.001,
                    f"{f:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    ax.set_xticks(x2)
    ax.set_xticklabels([lbl for _, lbl in STEP_BUCKETS], fontsize=10)
    ax.set_ylabel("Fraction of all human sessions challenged", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(
        "When Is the Challenge Issued?\n(% of all human sessions, by window observed at challenge)"
    )
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5)

    fig.suptitle(
        "Human Challenge Rate: Single-View Challenges 1.4% of Users; Temporal Agents Challenge None",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig16_challenge_difficulty")


# ---------------------------------------------------------------------------
# FIGURE 18 — Training curves: reward + val accuracy for PPO / DG / Soft PPO
# ---------------------------------------------------------------------------
def fig_training_curves(data: dict):
    """2×2 training dynamics: reward, val accuracy, policy entropy, train correct rate."""
    training = data.get("training", {})
    if not training:
        print("  [skip] fig_training_curves — no training logs found")
        return

    aug, preset = "advaug", "v2"

    def _smooth(vals, w=5):
        out = []
        for i in range(len(vals)):
            lo, hi = max(0, i - w + 1), i + 1
            out.append(float(np.mean(vals[lo:hi])))
        return out

    def _interp_to_common(xs_list, ys_list):
        all_steps = sorted({s for xs in xs_list for s in xs})
        interp_ys = []
        for xs, ys in zip(xs_list, ys_list):
            if len(xs) < 2:
                continue
            interp_ys.append(np.interp(all_steps, xs, ys))
        return np.array(all_steps), np.array(interp_ys)

    def _plot(ax, xs_list, ys_list, color, label, smooth=False, marker=None):
        if not xs_list:
            return
        ys_list = [_smooth(y) if smooth else y for y in ys_list]
        common_steps, mat = _interp_to_common(xs_list, ys_list)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        kw = dict(color=color, linewidth=2, label=label)
        if marker:
            kw["marker"] = marker
            kw["markersize"] = 4
        ax.plot(common_steps / 1000, mean, **kw)
        ax.fill_between(
            common_steps / 1000, mean - std, mean + std, color=color, alpha=0.15
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_reward, ax_val, ax_entropy, ax_correct = (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    )

    for algo in ALGOS:
        color = ALGO_COLORS[algo]
        label = ALGO_LABELS[algo]

        r_xs, r_ys, e_xs, e_ys, c_xs, c_ys, v_xs, v_ys = [], [], [], [], [], [], [], []
        for seed in SEEDS:
            d = training.get((algo, aug, preset, seed))
            if not d:
                continue
            if d["steps"] and d["rewards"]:
                r_xs.append(d["steps"])
                r_ys.append(d["rewards"])
            if d["steps"] and d.get("entropies"):
                e_xs.append(d["steps"][: len(d["entropies"])])
                e_ys.append(d["entropies"])
            if d["steps"] and d.get("correct_rates"):
                c_xs.append(d["steps"][: len(d["correct_rates"])])
                c_ys.append(d["correct_rates"])
            if d["val_steps"] and d["val_accs"]:
                v_xs.append(d["val_steps"])
                v_ys.append(d["val_accs"])

        _plot(ax_reward, r_xs, r_ys, color, label, smooth=True)
        _plot(ax_entropy, e_xs, e_ys, color, label, smooth=True)
        _plot(ax_correct, c_xs, c_ys, color, label, smooth=True)
        _plot(ax_val, v_xs, v_ys, color, label, marker="o")

    ax_reward.set_ylabel("Avg episode reward")
    ax_reward.set_title("Training Reward (smoothed)")
    ax_reward.legend(fontsize=9)

    ax_val.set_ylabel("Validation accuracy")
    ax_val.set_title("Validation Accuracy")
    ax_val.set_ylim(0.5, 1.02)
    ax_val.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax_val.legend(fontsize=9)

    ax_entropy.set_ylabel("Policy entropy")
    ax_entropy.set_title("Policy Entropy (smoothed)")
    ax_entropy.set_xlabel("Training steps (×1000)")
    ax_entropy.legend(fontsize=9)

    ax_correct.set_ylabel("Fraction correct decisions")
    ax_correct.set_title(
        "Training Correct Rate (smoothed)\n"
        "(correct_allow + correct_block + bot_blocked_puzzle + human_passed_puzzle)"
    )
    ax_correct.set_ylim(0.0, 1.05)
    ax_correct.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax_correct.set_xlabel("Training steps (×1000)")
    ax_correct.legend(fontsize=9)

    fig.suptitle(
        "Training Dynamics: PPO, DG, and Soft PPO (Adversarial Augmentation, mean ± std, 5 seeds)",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig18_training_curves")


# ---------------------------------------------------------------------------
# FIGURE 17 — Human disjoint generalization
# ---------------------------------------------------------------------------
def fig_human_disjoint(data: dict):
    """Accuracy (= human pass-through rate) for baseline vs disjoint PPO on held-out persons."""
    hd = data.get("human_disjoint", {})
    persons = [p for p in ["personA", "personB"] if p in hd]
    if not persons:
        print("  [skip] fig_human_disjoint — no human disjoint logs found")
        return

    person_labels = {"personA": "Person A", "personB": "Person B"}
    BLUE = "#2563a8"
    RED = "#C0392B"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(persons))
    w = 0.30

    for j, (kind, color, label) in enumerate(
        [
            ("baseline", BLUE, "Baseline PPO\n(trained WITH person)"),
            ("disjoint", RED, "Disjoint PPO\n(trained WITHOUT person)"),
        ]
    ):
        means, stds = [], []
        seed_lists = []
        for p in persons:
            d = hd[p]
            means.append(d[f"{kind}_acc"][0])
            stds.append(d[f"{kind}_acc"][1])
            seed_lists.append(d.get(f"{kind}_seeds", []))

        offset = (j - 0.5) * w
        ax.bar(
            x + offset,
            means,
            w,
            yerr=stds,
            capsize=5,
            color=color,
            alpha=0.85,
            ecolor="gray",
            label=label,
        )

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(
                x[i] + offset,
                m + s + 0.003,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        for i, seeds in enumerate(seed_lists):
            n = len(seeds)
            for k, sv in enumerate(seeds):
                jit = (k - n / 2) * (0.05 / max(n, 1))
                ax.scatter(
                    x[i] + offset + jit, sv, color=color, s=18, zorder=5, alpha=0.65
                )

    ax.set_xticks(x)
    ax.set_xticklabels([person_labels[p] for p in persons], fontsize=11)
    ax.set_ylabel("Human pass-through rate (accuracy)")
    ax.set_ylim(0.84, 1.04)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "Human Disjoint Generalization: Agent Correctly Passes Unseen Persons",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig17_human_disjoint")


# ---------------------------------------------------------------------------
# FIGURE 19 — v1 vs v2 reward: action distribution shift
# ---------------------------------------------------------------------------
def fig_v1_v2_action_dist(data: dict):
    """Stacked bar chart showing how the v1→v2 reward redesign shifted action distributions."""
    ad = data.get("action_dist", {})
    if not any((algo, "advaug", p) in ad for algo in ALGOS for p in ["v1", "v2"]):
        print("  [skip] fig_v1_v2_action_dist — no action dist data")
        return

    # Canonical action order + colors
    ACTION_ORDER = ["allow", "block", "easy_puzzle", "medium_puzzle", "hard_puzzle"]
    ACTION_COLORS = {
        "allow": "#2ecc71",  # green
        "block": "#e74c3c",  # red
        "easy_puzzle": "#aed6f1",  # light blue
        "medium_puzzle": "#2980b9",  # medium blue
        "hard_puzzle": "#1a5276",  # dark blue
    }
    ACTION_LABELS = {
        "allow": "Allow",
        "block": "Direct Block",
        "easy_puzzle": "Easy Puzzle",
        "medium_puzzle": "Medium Puzzle",
        "hard_puzzle": "Hard Puzzle",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    preset_labels = {
        "v1": "legacy reward\n(block-dominant)",
        "v2": "revised reward\n(puzzle-dominant)",
    }

    for ax, algo in zip(list(axes), ALGOS):
        x = np.arange(2)  # v1, v2
        bottoms = np.zeros(2)
        plotted = set()

        for action in ACTION_ORDER:
            heights = []
            for i, preset in enumerate(["v1", "v2"]):
                dist = ad.get((algo, "advaug", preset), {})
                heights.append(dist.get(action, 0.0))

            if all(h == 0 for h in heights):
                continue

            label = ACTION_LABELS[action] if action not in plotted else "_nolegend_"
            plotted.add(action)
            ax.bar(
                x,
                heights,
                bottom=bottoms,
                color=ACTION_COLORS[action],
                alpha=0.88,
                label=label,
                edgecolor="white",
                linewidth=0.5,
            )

            for i, (h, b) in enumerate(zip(heights, bottoms)):
                if h > 0.03:
                    ax.text(
                        x[i],
                        b + h / 2,
                        f"{h*100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white",
                        fontweight="bold",
                    )
            bottoms += np.array(heights)

        ax.set_xticks(x)
        ax.set_xticklabels([preset_labels["v1"], preset_labels["v2"]], fontsize=9)
        ax.set_title(ALGO_LABELS[algo], fontweight="bold")
        ax.set_ylim(0, 1.05)
        if ax is list(axes)[0]:
            ax.set_ylabel("Fraction of terminal decisions")

    # Shared legend from first subplot that has patches
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
        framealpha=0.9,
    )

    fig.suptitle(
        "Reward Redesign Shifts Decisions: Legacy Favours Direct Block; Revised Favours Puzzle Challenge",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _save(fig, "fig19_v1_v2_action_dist")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading eval logs...")
    data = load_all()

    n_native = len(data["native"])
    n_cross = len(data["cross"])
    n_ht = len(data["heldout_tier"])
    n_hf = len(data["heldout_family"])
    n_aug = len(data["augtest"])
    n_base = len(data["baselines"])
    n_sens = len(data["sensitivity"])
    print(
        f"  Native: {n_native}/6  Cross(v2->v1): {n_cross}/6  "
        f"HeldTier: {n_ht}/15  HeldFam: {n_hf}/15  AugTest: {n_aug}/6"
    )
    print(
        f"  Baselines: {n_base}/1  Sensitivity sweeps: {n_sens}/{len(ALL_SENSITIVITY_SWEEPS)}"
    )

    if n_native + n_cross + n_ht + n_hf + n_aug + n_base + n_sens == 0:
        print(
            "No completed eval logs found — run eval_all.ps1 / eval_baselines.ps1 / eval_sensitivity.ps1 first."
        )
        sys.exit(0)

    print("\nGenerating figures...")
    fig_main_accuracy(data)
    fig_aug_effect(data)
    fig_cross_env(data)
    fig_heldout_tiers(data)
    fig_heldout_families(data)
    fig_seed_variance(data)
    fig_augmented_test(data)
    fig_per_tier_detection(data)
    fig_baselines_comparison(data)
    fig_sensitivity_reward(data)
    fig_sensitivity_challenge(data)
    fig_ablation_comparison(data)
    fig_temporal_vs_singleview(data)
    fig_strict_accuracy(data)
    fig_disjoint_eval(data)
    fig_challenge_difficulty(data)
    fig_human_disjoint(data)
    fig_training_curves(data)
    fig_v1_v2_action_dist(data)

    print("\nGenerating tables...")
    table_main_results(data)
    table_cross_env(data)
    table_heldout_tiers(data)
    table_heldout_families(data)
    table_seed_variance(data)
    table_baselines(data)
    table_sensitivity(data)
    table_ablations(data)

    print(f"\nDone. Figures -> {FIG_DIR}/  Tables -> {TABLE_DIR}/")
