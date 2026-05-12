"""Generate the per-family / family-disjoint detection plots for the
XGBoost classifier.

Produces two figures (matching the layout of the RL per-family figure
``fig:per_family_detection`` in Section 4.1.5):

1. ``family_disjoint_bars.png`` — primary figure. Detection accuracy on
   held-out family bots, grouped by classifier configuration (noaug vs.
   advaug). Error bars are std over training seeds. The retrained model
   has never seen the held-out family during training.

2. ``per_family_test_bars.png`` — companion figure. Per-family slice of the
   standard saved models (xgb_v1, xgb_v1_noaug, xgb_v2, xgb_v2_noaug)
   evaluated on the standard 70/30 test split.

Usage (from repo root)::

    python src/classifier/scripts/plot_family_disjoint.py
"""

from __future__ import annotations

import argparse
import json
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

# Match the bot-tier ordering used in the paper.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot family-disjoint results")
    p.add_argument(
        "--disjoint-dir",
        type=str,
        default="src/classifier/family_disjoint",
        help="Directory containing noaug/ and advaug/ subdirs",
    )
    p.add_argument(
        "--per-family-dir",
        type=str,
        default="src/classifier/family_disjoint/per_family",
        help="Directory containing per_family_summary.json",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="src/classifier/family_disjoint/plots",
    )
    return p.parse_args()


def _load_disjoint_summary(path: Path) -> dict[str, dict]:
    """Load family_disjoint_summary -> {family: {mean, std, n}}."""
    with open(path) as f:
        payload = json.load(f)
    summary = payload["summary"]
    return {
        row["family"]: {
            "mean": row["heldout_detection_rate_mean"],
            "std": row["heldout_detection_rate_std"],
            "n": row["n_heldout_orig"],
        }
        for row in summary
    }


def plot_family_disjoint(noaug_path: Path, advaug_path: Path, out_path: Path) -> None:
    noaug = _load_disjoint_summary(noaug_path)
    advaug = _load_disjoint_summary(advaug_path)

    families = [f for f in FAMILY_ORDER if f in noaug or f in advaug]
    labels = [FAMILY_DISPLAY.get(f, f) for f in families]

    noaug_mean = np.array([noaug.get(f, {}).get("mean", np.nan) for f in families])
    noaug_std = np.array([noaug.get(f, {}).get("std", 0.0) for f in families])
    advaug_mean = np.array([advaug.get(f, {}).get("mean", np.nan) for f in families])
    advaug_std = np.array([advaug.get(f, {}).get("std", 0.0) for f in families])

    x = np.arange(len(families))
    width = 0.4

    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(
        x - width / 2,
        noaug_mean,
        width,
        yerr=noaug_std,
        label="XGBoost (noaug)",
        color="steelblue",
        capsize=3,
        edgecolor="black",
        linewidth=0.6,
    )
    b2 = ax.bar(
        x + width / 2,
        advaug_mean,
        width,
        yerr=advaug_std,
        label="XGBoost (advaug)",
        color="tomato",
        capsize=3,
        edgecolor="black",
        linewidth=0.6,
    )

    for bars, means in [(b1, noaug_mean), (b2, advaug_mean)]:
        for rect, val in zip(bars, means):
            if np.isnan(val):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.012,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Detection Accuracy on Held-Out Family")
    ax.set_title(
        "XGBoost Family-Disjoint Detection (held-out family unseen during training)"
    )
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


def plot_per_family_test(per_family_path: Path, out_path: Path) -> None:
    """Per-family detection on the standard saved models (no retraining)."""
    with open(per_family_path) as f:
        payload = json.load(f)
    rows = payload["rows"]
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    families = [f for f in FAMILY_ORDER if any(r["family"] == f for r in rows)]
    labels = [FAMILY_DISPLAY.get(f, f) for f in families]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(families))
    n_models = len(models)
    width = 0.8 / n_models
    cmap = plt.get_cmap("tab10")

    for mi, model in enumerate(models):
        means = []
        for fam in families:
            cell = next(
                (r for r in rows if r["model"] == model and r["family"] == fam),
                None,
            )
            means.append(cell["detection_rate"] if cell else np.nan)
        offset = (mi - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=model,
            color=cmap(mi),
            edgecolor="black",
            linewidth=0.5,
        )
        for rect, val in zip(bars, means):
            if np.isnan(val):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.012,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Detection Accuracy on Test Split")
    ax.set_title("XGBoost Per-Family Detection (standard test split, no retraining)")
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noaug_json = Path(args.disjoint_dir) / "noaug" / "family_disjoint_per_seed.json"
    advaug_json = Path(args.disjoint_dir) / "advaug" / "family_disjoint_per_seed.json"
    if not noaug_json.exists() or not advaug_json.exists():
        print(
            f"ERROR: missing disjoint results. Looked for:\n  {noaug_json}\n  {advaug_json}"
        )
        raise SystemExit(1)

    plot_family_disjoint(noaug_json, advaug_json, out_dir / "family_disjoint_bars.png")

    per_family_json = Path(args.per_family_dir) / "per_family_summary.json"
    if per_family_json.exists():
        plot_per_family_test(per_family_json, out_dir / "per_family_test_bars.png")
    else:
        print(
            f"[plot] Skipping per-family test plot (no {per_family_json}); "
            "run evaluate_per_family.py first."
        )


if __name__ == "__main__":
    main()
