"""Tier-disjoint evaluation for the XGBoost classifier.

For each adversarial tier T (1-5):
    1. Hold out every bot session whose bot_type maps to tier T.
    2. Train a fresh classifier on (humans + all bots from other tiers).
       With ``--adversarial-augment``, also include augmented bots from
       other tiers in the training split.
    3. Evaluate on all held-out tier T bot sessions and on a stratified
       held-out sanity test split drawn from (humans + non-T bots).

Mirrors evaluate_family_disjoint.py but partitions by tier instead of
family, to match the RL per-tier generalization figure.

Usage (from repo root)::

    python src/classifier/scripts/evaluate_tier_disjoint.py \\
        --data-dir src/data/ \\
        --output-dir src/classifier/tier_disjoint/advaug \\
        --adversarial-augment \\
        --n-seeds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from classifier.data_loader import Session, is_augmented, load_from_directory
from classifier.features import SessionFeatureExtractor
from classifier.model import HumanLikelihoodClassifier
from rl_captcha.config import ClassifierConfig
from rl_captcha.data.loader import TIER_NAMES, bot_type_to_tier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tier-disjoint evaluation for the XGBoost classifier"
    )
    p.add_argument("--data-dir", type=str, default="src/data/")
    p.add_argument(
        "--output-dir",
        type=str,
        default="src/classifier/tier_disjoint/run",
        help="Directory where per-tier results, summary CSV, and JSON are saved",
    )
    p.add_argument(
        "--adversarial-augment",
        action="store_true",
        help="Include pre-generated augmented bot sessions (from "
        "data/bot_augmented/) for non-held-out tiers in the train split.",
    )
    p.add_argument(
        "--no-feature-adversarial",
        action="store_true",
        help="Disable the in-classifier feature-space humanization "
        "(ClassifierConfig.adversarial_augment).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of independent retrains per tier (default: 5)",
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base random seed; seeds = [base, base+1, ..., base+n_seeds-1]",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Stratified held-out fraction from (humans + non-T bots) used "
        "for a sanity test (default: 0.3, matching train_classifier.py)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for human vs bot (default: 0.5)",
    )
    p.add_argument(
        "--tiers",
        type=str,
        default="",
        help="Comma-separated tier integers to evaluate. Empty = every "
        "tier present in the data.",
    )
    return p.parse_args()


def _split_sessions(
    sessions: list[Session],
) -> tuple[list[Session], list[Session], list[Session], list[Session]]:
    humans: list[Session] = []
    bots: list[Session] = []
    aug_humans: list[Session] = []
    aug_bots: list[Session] = []
    for s in sessions:
        if s.label is None:
            continue
        aug = is_augmented(s)
        if s.label == 1:
            (aug_humans if aug else humans).append(s)
        else:
            (aug_bots if aug else bots).append(s)
    return humans, bots, aug_humans, aug_bots


def _tier_of(session: Session) -> int:
    """Resolve a session's adversarial tier (1-5). Unknown → 0."""
    explicit = session.metadata.get("tier")
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    return bot_type_to_tier(session.metadata.get("bot_type"))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"n": 0, "accuracy": float("nan"), "detection_rate": float("nan")}
    correct = (y_true == y_pred).astype(int)
    acc = float(correct.mean())
    bot_mask = y_true == 0
    if bot_mask.any():
        detection_rate = float((y_pred[bot_mask] == 0).mean())
    else:
        detection_rate = float("nan")
    human_mask = y_true == 1
    if human_mask.any():
        pass_through = float((y_pred[human_mask] == 1).mean())
    else:
        pass_through = float("nan")
    return {
        "n": int(len(y_true)),
        "accuracy": acc,
        "detection_rate": detection_rate,
        "human_pass_through": pass_through,
    }


def _run_one_seed(
    tier: int,
    seed: int,
    humans: list[Session],
    bots: list[Session],
    aug_bots: list[Session],
    args: argparse.Namespace,
    extractor: SessionFeatureExtractor,
) -> dict:
    """Train on (humans + non-T bots) and evaluate on held-out tier T."""
    from sklearn.model_selection import train_test_split

    train_bots_orig = [s for s in bots if _tier_of(s) != tier]
    heldout_orig = [s for s in bots if _tier_of(s) == tier]
    heldout_aug = [s for s in aug_bots if _tier_of(s) == tier]

    if not heldout_orig:
        return {"tier": tier, "seed": seed, "error": "no held-out originals"}

    pool = humans + train_bots_orig
    y_pool = np.array([s.label for s in pool], dtype=int)
    train_idx, test_idx = train_test_split(
        np.arange(len(pool)),
        test_size=args.test_size,
        stratify=y_pool if len(np.unique(y_pool)) > 1 else None,
        random_state=seed,
    )
    train_sessions = [pool[i] for i in train_idx]
    sanity_test_sessions = [pool[i] for i in test_idx]

    if args.adversarial_augment:
        train_aug = [s for s in aug_bots if _tier_of(s) != tier]
        train_sessions = train_sessions + train_aug
    X_train = extractor.extract_many(train_sessions)
    y_train = np.array([s.label for s in train_sessions], dtype=int)

    X_sanity = extractor.extract_many(sanity_test_sessions)
    y_sanity = np.array([s.label for s in sanity_test_sessions], dtype=int)

    X_held = extractor.extract_many(heldout_orig)
    y_held = np.array([s.label for s in heldout_orig], dtype=int)

    if heldout_aug:
        X_held_aug = extractor.extract_many(heldout_aug)
        y_held_aug = np.array([s.label for s in heldout_aug], dtype=int)
    else:
        X_held_aug = None
        y_held_aug = None

    cfg = ClassifierConfig(random_state=seed)
    if args.no_feature_adversarial:
        cfg.adversarial_augment = False

    clf = HumanLikelihoodClassifier(config=cfg)
    clf.fit(X_train, y_train)

    thr = args.threshold

    sanity_scores = clf.human_score(X_sanity)
    sanity_pred = (sanity_scores >= thr).astype(int)
    sanity = _binary_metrics(y_sanity, sanity_pred)

    held_scores = clf.human_score(X_held)
    held_pred = (held_scores >= thr).astype(int)
    held = _binary_metrics(y_held, held_pred)

    if X_held_aug is not None:
        held_aug_scores = clf.human_score(X_held_aug)
        held_aug_pred = (held_aug_scores >= thr).astype(int)
        held_aug = _binary_metrics(y_held_aug, held_aug_pred)
    else:
        held_aug = {"n": 0}

    return {
        "tier": tier,
        "seed": seed,
        "n_train": int(len(y_train)),
        "n_train_human": int((y_train == 1).sum()),
        "n_train_bot": int((y_train == 0).sum()),
        "n_heldout_orig": int(len(y_held)),
        "n_heldout_aug": int(held_aug.get("n", 0)),
        "sanity_accuracy": sanity["accuracy"],
        "sanity_detection_rate": sanity["detection_rate"],
        "sanity_human_pass_through": sanity["human_pass_through"],
        "heldout_detection_rate": held["detection_rate"],
        "heldout_aug_detection_rate": held_aug.get("detection_rate", float("nan")),
    }


def _aggregate(per_seed: list[dict]) -> dict:
    fields = [
        "sanity_accuracy",
        "sanity_detection_rate",
        "sanity_human_pass_through",
        "heldout_detection_rate",
        "heldout_aug_detection_rate",
    ]
    out: dict = {}
    for k in fields:
        values = [r[k] for r in per_seed if k in r and not np.isnan(r[k])]
        if not values:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
            continue
        out[f"{k}_mean"] = float(np.mean(values))
        out[f"{k}_std"] = float(np.std(values, ddof=0))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tier_disjoint] Loading sessions from {Path(args.data_dir).resolve()} ...")
    sessions = load_from_directory(
        args.data_dir, include_augmented=args.adversarial_augment
    )
    humans, bots, aug_humans, aug_bots = _split_sessions(sessions)
    if aug_humans:
        print(
            f"  WARNING: {len(aug_humans)} augmented human sessions found; "
            f"these are ignored (we only train on augmented bots)."
        )
    print(
        f"  Loaded: {len(humans)} humans, {len(bots)} bot originals, "
        f"{len(aug_bots)} augmented bots"
    )

    tier_counts: dict[int, int] = defaultdict(int)
    for s in bots:
        tier_counts[_tier_of(s)] += 1

    if args.tiers.strip():
        requested = [int(t.strip()) for t in args.tiers.split(",") if t.strip()]
        missing = [t for t in requested if t not in tier_counts]
        if missing:
            print(f"ERROR: requested tiers not found in data: {missing}")
            sys.exit(1)
        tiers = requested
    else:
        tiers = sorted(t for t in tier_counts.keys() if t > 0)

    print(f"  Tiers to evaluate ({len(tiers)}): {tiers}")

    extractor = SessionFeatureExtractor()
    seeds = [args.seed_base + i for i in range(args.n_seeds)]
    print(f"  Seeds: {seeds}")
    print(f"  Adversarial augmentation (data-level): {args.adversarial_augment}")
    print(
        f"  Adversarial augmentation (feature-level): "
        f"{not args.no_feature_adversarial}\n"
    )

    summary: list[dict] = []
    per_seed_dump: list[dict] = []

    for tier in tiers:
        n_tier = tier_counts[tier]
        print(
            f"=== Tier {tier} ({TIER_NAMES.get(tier, 'unknown')})  "
            f"(n_originals={n_tier}) ==="
        )
        per_seed_rows: list[dict] = []
        for seed in seeds:
            t0 = time.time()
            try:
                row = _run_one_seed(
                    tier=tier,
                    seed=seed,
                    humans=humans,
                    bots=bots,
                    aug_bots=aug_bots,
                    args=args,
                    extractor=extractor,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [seed={seed}] FAILED: {exc}")
                continue
            elapsed = time.time() - t0
            per_seed_rows.append(row)
            per_seed_dump.append(row)
            held = row.get("heldout_detection_rate", float("nan"))
            sanity = row.get("sanity_accuracy", float("nan"))
            print(
                f"  [seed={seed}] heldout_det={held:.4f}  "
                f"sanity_acc={sanity:.4f}  ({elapsed:.1f}s)"
            )

        if not per_seed_rows:
            print(f"  No successful seeds for tier {tier}; skipping aggregation.")
            continue

        agg = _aggregate(per_seed_rows)
        summary.append(
            {
                "tier": tier,
                "display_name": TIER_NAMES.get(tier, f"tier_{tier}"),
                "n_originals": n_tier,
                "n_seeds_ok": len(per_seed_rows),
                "n_train_human": per_seed_rows[0]["n_train_human"],
                "n_train_bot_mean": float(
                    np.mean([r["n_train_bot"] for r in per_seed_rows])
                ),
                "n_heldout_orig": per_seed_rows[0]["n_heldout_orig"],
                "n_heldout_aug": per_seed_rows[0]["n_heldout_aug"],
                **agg,
            }
        )
        print(
            f"  -> heldout_det = {agg['heldout_detection_rate_mean']:.4f} "
            f"± {agg['heldout_detection_rate_std']:.4f}\n"
        )

    csv_path = out_dir / "tier_disjoint_summary.csv"
    if summary:
        fields = list(summary[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
        print(f"[tier_disjoint] Wrote summary CSV -> {csv_path}")

    json_path = out_dir / "tier_disjoint_per_seed.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "args": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()
                },
                "seeds": seeds,
                "summary": summary,
                "per_seed": per_seed_dump,
            },
            fh,
            indent=2,
        )
    print(f"[tier_disjoint] Wrote per-seed JSON -> {json_path}")

    print("\n=== Tier-Disjoint Detection Summary ===")
    print(
        f"{'Tier':<22s} {'n_held':>7s} {'detect_mean':>12s} "
        f"{'detect_std':>11s} {'sanity_acc':>11s}"
    )
    for row in sorted(summary, key=lambda r: r["tier"]):
        print(
            f"{row['display_name']:<22s} {row['n_heldout_orig']:>7d} "
            f"{row['heldout_detection_rate_mean']:>12.4f} "
            f"{row['heldout_detection_rate_std']:>11.4f} "
            f"{row['sanity_accuracy_mean']:>11.4f}"
        )


if __name__ == "__main__":
    main()
