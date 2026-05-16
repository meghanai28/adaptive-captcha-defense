"""Generate the XGBoost tier-disjoint detection plot.

``tier_disjoint_bars.png`` — XGBoost held-out-tier detection (noaug vs
advaug), mirroring family_disjoint_bars but partitioned by adversarial
tier (1-5) instead of bot family.

Usage (from repo root)::

    python src/classifier/scripts/plot_tier_disjoint.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TIER_NAMES = {
    1: "T1\nCommodity",
    2: "T2\nCareful",
    3: "T3\nSemi-Auto",
    4: "T4\nTrace-Cond",
    5: "T5\nLLM",
}
TIER_ORDER = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot XGBoost tier-disjoint results")
    p.add_argument("--disjoint-dir", type=str, default="src/classifier/tier_disjoint")
    p.add_argument(
        "--output-dir", type=str, default="src/classifier/tier_disjoint/plots"
    )
    return p.parse_args()


def _load_xgb_summary(path: Path) -> dict[int, dict]:
    with open(path) as f:
        payload = json.load(f)
    return {
        int(row["tier"]): {
            "mean": row["heldout_detection_rate_mean"],
            "std": row["heldout_detection_rate_std"],
            "n": row["n_heldout_orig"],
        }
        for row in payload["summary"]
    }


def _add_bar_labels(ax, bars, values):
    for rect, val in zip(bars, values):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.012,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )


def plot_xgb_tier_disjoint(noaug_path: Path, advaug_path: Path, out_path: Path) -> None:
    noaug = _load_xgb_summary(noaug_path)
    advaug = _load_xgb_summary(advaug_path)
    tiers = [t for t in TIER_ORDER if t in noaug or t in advaug]
    labels = [TIER_NAMES[t] for t in tiers]

    nm = np.array([noaug.get(t, {}).get("mean", np.nan) for t in tiers])
    ns = np.array([noaug.get(t, {}).get("std", 0.0) for t in tiers])
    am = np.array([advaug.get(t, {}).get("mean", np.nan) for t in tiers])
    asd = np.array([advaug.get(t, {}).get("std", 0.0) for t in tiers])

    x = np.arange(len(tiers))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(
        x - w / 2,
        nm,
        w,
        yerr=ns,
        label="XGBoost (noaug)",
        color="steelblue",
        capsize=3,
        edgecolor="black",
        linewidth=0.6,
    )
    b2 = ax.bar(
        x + w / 2,
        am,
        w,
        yerr=asd,
        label="XGBoost (advaug)",
        color="tomato",
        capsize=3,
        edgecolor="black",
        linewidth=0.6,
    )
    _add_bar_labels(ax, b1, nm)
    _add_bar_labels(ax, b2, am)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Detection Accuracy on Held-Out Tier")
    ax.set_title(
        "XGBoost Tier-Disjoint Detection (held-out tier unseen during training)"
    )
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noaug_json = Path(args.disjoint_dir) / "noaug" / "tier_disjoint_per_seed.json"
    advaug_json = Path(args.disjoint_dir) / "advaug" / "tier_disjoint_per_seed.json"
    if not noaug_json.exists() or not advaug_json.exists():
        print(f"ERROR: missing disjoint results: {noaug_json}, {advaug_json}")
        raise SystemExit(1)

    plot_xgb_tier_disjoint(noaug_json, advaug_json, out_dir / "tier_disjoint_bars.png")


if __name__ == "__main__":
    main()
