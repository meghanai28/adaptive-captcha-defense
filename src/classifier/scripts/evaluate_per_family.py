"""Per-family detection accuracy for saved XGBoost classifiers.

Mirrors the protocol used for the RL agents in Section 4.1.5 / Figure
``fig:per_family_detection``: load a model that was trained on the standard
70/30 split, slice the held-out test split by ``bot_type``, and report
detection accuracy on each family.

Unlike ``evaluate_family_disjoint.py``, this script does NOT retrain — it
just slices the existing test set per family, so we get a fair like-for-like
comparison with the figure that already exists for the RL agents.

Usage (from repo root)::

    python src/classifier/scripts/evaluate_per_family.py \\
        --data-dir src/data/ \\
        --models src/classifier/models/xgb_v1 src/classifier/models/xgb_v2 \\
        --labels "xgb_v1 (tuned+aug)" "xgb_v2 (default+aug)" \\
        --output-dir src/classifier/family_disjoint/per_family
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from classifier.data_loader import is_augmented, load_from_directory
from classifier.features import SessionFeatureExtractor
from classifier.model import HumanLikelihoodClassifier


FAMILY_DISPLAY = {
    "linear": "Linear",
    "tabber": "Tabber",
    "speedrun": "Speedrun",
    "scripted": "Scripted",
    "stealth": "Stealth",
    "slow": "Slow",
    "erratic": "Erratic",
    "semi_auto": "Semi-Auto",
    "trace_conditioned": "Trace-Conditioned",
    "llm": "LLM",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-family detection accuracy on saved classifiers"
    )
    p.add_argument("--data-dir", type=str, default="src/data/")
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="One or more saved-model directories",
    )
    p.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional display labels, one per --models entry",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="src/classifier/family_disjoint/per_family",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction held out (must match training; default 0.3)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=13,
        help="Random seed (must match training; default 13)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold (default 0.5)",
    )
    return p.parse_args()


def _reproduce_test_split(
    data_dir: str, test_size: float, random_state: int
) -> list:
    """Rebuild the held-out test split that train_classifier.py uses."""
    from sklearn.model_selection import train_test_split

    sessions = load_from_directory(data_dir, include_augmented=False)
    labeled = [s for s in sessions if s.label is not None]
    originals = [s for s in labeled if not is_augmented(s)]
    y_orig = np.array([s.label for s in originals], dtype=int)
    _, test_idx = train_test_split(
        np.arange(len(originals)),
        test_size=test_size,
        stratify=y_orig if len(np.unique(y_orig)) > 1 else None,
        random_state=random_state,
    )
    return [originals[i] for i in test_idx]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [Path(m).name for m in args.models]
    if len(labels) != len(args.models):
        print("ERROR: --labels length must match --models length")
        sys.exit(1)

    test_sessions = _reproduce_test_split(
        args.data_dir, args.test_size, args.random_state
    )
    test_bots = [s for s in test_sessions if s.label == 0]
    test_humans = [s for s in test_sessions if s.label == 1]
    print(
        f"[per_family] Test split = {len(test_sessions)} "
        f"({len(test_humans)}H / {len(test_bots)}B)"
    )

    families = sorted({str(s.metadata.get("bot_type", "unknown")) for s in test_bots})
    fam_to_sessions = {
        fam: [s for s in test_bots if s.metadata.get("bot_type") == fam]
        for fam in families
    }
    print("  Families in test split:")
    for fam in families:
        print(f"    {fam}: {len(fam_to_sessions[fam])}")

    extractor = SessionFeatureExtractor()

    all_rows: list[dict] = []
    for model_dir, label in zip(args.models, labels):
        print(f"\n=== {label}  ({model_dir}) ===")
        clf = HumanLikelihoodClassifier.load(model_dir)

        # Overall detection rate / human pass-through, for context.
        X_b = extractor.extract_many(test_bots)
        bot_scores = clf.human_score(X_b)
        bot_preds = (bot_scores >= args.threshold).astype(int)
        overall_detect = float((bot_preds == 0).mean())

        X_h = extractor.extract_many(test_humans)
        human_scores = clf.human_score(X_h)
        human_preds = (human_scores >= args.threshold).astype(int)
        overall_pass = float((human_preds == 1).mean())
        print(
            f"  Overall: detection_rate={overall_detect:.4f}  "
            f"human_pass_through={overall_pass:.4f}"
        )

        for fam in families:
            fam_sessions = fam_to_sessions[fam]
            if not fam_sessions:
                continue
            X = extractor.extract_many(fam_sessions)
            scores = clf.human_score(X)
            preds = (scores >= args.threshold).astype(int)
            detection_rate = float((preds == 0).mean())
            row = {
                "model": label,
                "model_dir": str(model_dir),
                "family": fam,
                "display_name": FAMILY_DISPLAY.get(fam, fam),
                "n": len(fam_sessions),
                "detection_rate": detection_rate,
                "mean_human_score": float(scores.mean()),
                "overall_detect_rate": overall_detect,
                "overall_human_pass_through": overall_pass,
            }
            all_rows.append(row)
            print(
                f"  {row['display_name']:<22s} "
                f"n={row['n']:>3d}  detect={detection_rate:.4f}"
            )

    csv_path = out_dir / "per_family_summary.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[per_family] Wrote summary CSV -> {csv_path}")

    json_path = out_dir / "per_family_summary.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "args": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()},
                "rows": all_rows,
            },
            fh,
            indent=2,
        )
    print(f"[per_family] Wrote JSON       -> {json_path}")


if __name__ == "__main__":
    main()
