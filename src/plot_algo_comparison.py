"""Compare XGBoost vs RL algorithms (PPO, Soft PPO) on the standard test split.

Produces two figures in src/figures/:

1. ``fig_algo_accuracy.png`` — overall accuracy on the standard 70/30 test
   split, bar chart with 3 bars (XGBoost, PPO+advaug, Soft PPO+advaug).
   Error bars are std across training seeds for RL; XGBoost is a single
   trained model so it has no seed std.

2. ``fig_algo_per_family.png`` — per-family bot detection rate, grouped
   bars by family with one bar per algorithm. Shows where each method
   is strong/weak across the 10 bot families.

RL numbers come from the native eval logs (``eval_{algo}_advaug_v2_native.log``)
parsed across all 5 training seeds. XGBoost numbers come from the
``per_family_summary.json`` (xgb_v2 default+aug) and the train-test
evaluation header (n=192, acc, f1).

Usage (from repo root or src/)::

    python src/plot_algo_comparison.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FAMILY_DISPLAY = {
    "linear": "Linear",
    "tabber": "Tabber",
    "speedrun": "Speedrun",
    "scripted": "Scripted",
    "stealth": "Stealth",
    "slow": "Slow",
    "erratic": "Erratic",
    "semi_auto": "Semi-Auto",
    "trace_conditioned": "Trace-Cond.",
    "llm": "LLM",
}
FAMILY_ORDER = [
    "linear",
    "tabber",
    "speedrun",
    "scripted",
    "stealth",
    "slow",
    "erratic",
    "semi_auto",
    "trace_conditioned",
    "llm",
]


def _decode_log(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def _parse_rl_accuracy(log_path: Path) -> tuple[float, float] | None:
    """Pull the aggregate accuracy from the COMPARISON TABLE (single algo log)."""
    if not log_path.exists():
        return None
    text = _decode_log(log_path)
    # In native logs the single agent's aggregate row is just "Accuracy   X +/- Y"
    matches = re.findall(r"Accuracy\s+([0-9]+\.[0-9]+)\s*\+/-\s*([0-9]+\.[0-9]+)", text)
    if not matches:
        return None
    # First match is the first agent's section. For 5 training seeds, every
    # agent reports an aggregate (mean across eval seeds). We average them.
    means = np.array([float(a) for a, _ in matches])
    return float(means.mean()), float(means.std(ddof=0))


def _parse_rl_per_family(log_path: Path) -> dict[str, tuple[float, float]]:
    """Average per-family detection rate across all training seeds in the log."""
    if not log_path.exists():
        return {}
    text = _decode_log(log_path)
    # Per-family blocks live under "Per-Family Bot Detection (AGENT) ---"
    # then a header row and per-family lines.
    per_family: dict[str, list[float]] = {f: [] for f in FAMILY_ORDER}
    # Match each Per-Family block per agent.
    blocks = re.findall(
        r"Per-Family Bot Detection \([^)]+\)[^\n]*\n[^\n]*\n[^\n]*\n((?:\s+\S+[^\n]*\n)+)",
        text,
    )
    for block in blocks:
        for line in block.splitlines():
            # e.g.: "  erratic               2    91      88     3   96.7%"
            m = re.match(r"\s+(\S+)\s+\d+\s+\d+\s+\d+\s+\d+\s+([0-9]+\.[0-9]+)%", line)
            if not m:
                continue
            fam = m.group(1).strip()
            rate = float(m.group(2)) / 100.0
            if fam in per_family:
                per_family[fam].append(rate)
    return {
        f: (float(np.mean(v)), float(np.std(v, ddof=0)))
        for f, v in per_family.items()
        if v
    }


def _parse_xgb_per_family(per_family_json: Path, model_label: str) -> dict[str, float]:
    """Return {family: detection_rate} for the requested XGBoost model."""
    with open(per_family_json) as f:
        payload = json.load(f)
    out: dict[str, float] = {}
    for row in payload["rows"]:
        if row["model"] != model_label:
            continue
        out[row["family"]] = row["detection_rate"]
    return out


def _xgb_standard_accuracy() -> tuple[float, float]:
    """Hard-code the standard-split numbers read from the train-test evaluation
    plot headers (n=192). xgb_v2 default+aug: acc=0.995, n=192."""
    return 0.995, 0.0


def _add_bar_labels(ax, bars, values, fmt: str = "{:.3f}", fontsize: float = 8.5):
    for rect, val in zip(bars, values):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.012,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )


def plot_accuracy_comparison(ppo_acc, soft_acc, xgb_acc, out_path: Path) -> None:
    """Bar chart: XGBoost vs PPO vs Soft PPO accuracy."""
    labels = ["XGBoost\n(v2 default+aug)", "RL PPO\n(advaug)", "RL Soft PPO\n(advaug)"]
    means = np.array([xgb_acc[0], ppo_acc[0], soft_acc[0]])
    stds = np.array([xgb_acc[1], ppo_acc[1], soft_acc[1]])
    colors = ["tomato", "#4c72b0", "#55a868"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    _add_bar_labels(ax, bars, means, fmt="{:.3f}", fontsize=10)

    ax.set_ylim(0.92, 1.02)
    ax.set_ylabel("Accuracy (standard 70/30 test split)")
    ax.set_title("Algorithm Comparison: Overall Accuracy on Standard Test Split")
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


def plot_per_family_comparison(
    ppo_per_fam, soft_per_fam, xgb_per_fam, out_path: Path
) -> None:
    """Per-family detection rate grouped by family, one bar per algorithm."""
    families = [f for f in FAMILY_ORDER if f in xgb_per_fam or f in ppo_per_fam]
    labels = [FAMILY_DISPLAY.get(f, f) for f in families]

    xgb = np.array([xgb_per_fam.get(f, np.nan) for f in families])
    ppo_m = np.array([ppo_per_fam.get(f, (np.nan, 0.0))[0] for f in families])
    ppo_s = np.array([ppo_per_fam.get(f, (np.nan, 0.0))[1] for f in families])
    soft_m = np.array([soft_per_fam.get(f, (np.nan, 0.0))[0] for f in families])
    soft_s = np.array([soft_per_fam.get(f, (np.nan, 0.0))[1] for f in families])

    x = np.arange(len(families))
    w = 0.27

    fig, ax = plt.subplots(figsize=(13, 5.5))
    b1 = ax.bar(
        x - w,
        xgb,
        w,
        label="XGBoost (v2 default+aug)",
        color="tomato",
        edgecolor="black",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x,
        ppo_m,
        w,
        yerr=ppo_s,
        capsize=2,
        label="RL PPO+advaug",
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.5,
    )
    b3 = ax.bar(
        x + w,
        soft_m,
        w,
        yerr=soft_s,
        capsize=2,
        label="RL Soft PPO+advaug",
        color="#55a868",
        edgecolor="black",
        linewidth=0.5,
    )
    _add_bar_labels(ax, b1, xgb, fmt="{:.2f}", fontsize=6.5)
    _add_bar_labels(ax, b2, ppo_m, fmt="{:.2f}", fontsize=6.5)
    _add_bar_labels(ax, b3, soft_m, fmt="{:.2f}", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Bot Detection Rate")
    ax.set_title(
        "Per-Family Bot Detection: XGBoost vs RL PPO vs RL Soft PPO "
        "(standard test split, advaug)"
    )
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=str, default="src/logs")
    p.add_argument(
        "--xgb-per-family",
        type=str,
        default="src/classifier/family_disjoint/per_family/per_family_summary.json",
    )
    p.add_argument("--output-dir", type=str, default="src/figures")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo_acc = _parse_rl_accuracy(log_dir / "eval_ppo_advaug_v2_native.log")
    soft_acc = _parse_rl_accuracy(log_dir / "eval_soft_ppo_advaug_v2_native.log")
    xgb_acc = _xgb_standard_accuracy()
    print(f"  PPO acc:  {ppo_acc}")
    print(f"  Soft acc: {soft_acc}")
    print(f"  XGB acc:  {xgb_acc}")

    if not (ppo_acc and soft_acc):
        print("ERROR: could not parse RL accuracy from logs")
        raise SystemExit(1)

    ppo_pf = _parse_rl_per_family(log_dir / "eval_ppo_advaug_v2_native.log")
    soft_pf = _parse_rl_per_family(log_dir / "eval_soft_ppo_advaug_v2_native.log")
    xgb_pf = _parse_xgb_per_family(Path(args.xgb_per_family), "xgb_v2 (default+aug)")
    print(f"  PPO families parsed:  {sorted(ppo_pf.keys())}")
    print(f"  Soft families parsed: {sorted(soft_pf.keys())}")
    print(f"  XGB families parsed:  {sorted(xgb_pf.keys())}")

    plot_accuracy_comparison(
        ppo_acc, soft_acc, xgb_acc, out_dir / "fig_algo_accuracy.png"
    )
    plot_per_family_comparison(
        ppo_pf, soft_pf, xgb_pf, out_dir / "fig_algo_per_family.png"
    )


if __name__ == "__main__":
    main()
