"""Train a scaffold-split exploratory model on the common multi-target matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "data" / "common" / "multi_target_features.npz"
MODEL_ROOT = ROOT / "models" / "multitarget"
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE_pIC50": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_pIC50": float(mean_absolute_error(y_true, y_pred)),
        "Spearman": float(spearman) if pd.notna(spearman) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--estimators", type=int, default=60)
    parser.add_argument("--max-features", type=float, default=0.08)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-rows-per-target", type=int, default=5000)
    parser.add_argument("--skip-target-holdout", action="store_true")
    args = parser.parse_args()
    if args.estimators < 10 or not 0 < args.max_features <= 1 or args.jobs < 1 or args.max_rows_per_target < 100:
        raise ValueError("Use --estimators >= 10, 0 < --max-features <= 1, and --jobs >= 1.")
    features_path = args.features if args.features.is_absolute() else ROOT / args.features
    if not features_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {features_path}; run build_multitarget_features.py first.")
    if args.out is None:
        existing = []
        for path in MODEL_ROOT.glob("baseline_v*"):
            try:
                existing.append(int(path.name.removeprefix("baseline_v")))
            except ValueError:
                pass
        output_dir = MODEL_ROOT / f"baseline_v{max(existing, default=0) + 1}"
    else:
        output_dir = args.out if args.out.is_absolute() else ROOT / args.out

    matrix = np.load(features_path, allow_pickle=False)
    X = matrix["X"]
    y = matrix["y"].astype(float)
    groups = matrix["scaffold_groups"].astype(str)
    target_ids = matrix["target_ids"].astype(str)
    smiles = matrix["canonical_smiles"].astype(str)
    # Balance targets deterministically so EGFR-size datasets cannot dominate the
    # common model. Sampling only affects the multi-target candidate, never the
    # per-target source files.
    selected = []
    for target_id in np.unique(target_ids):
        indices = np.flatnonzero(target_ids == target_id)
        ranked = sorted(indices, key=lambda idx: zlib.crc32(smiles[idx].encode("utf-8")))
        selected.extend(ranked[: args.max_rows_per_target])
    selected = np.asarray(sorted(selected), dtype=int)
    X, y, groups, target_ids, smiles = (
        X[selected], y[selected], groups[selected], target_ids[selected], smiles[selected]
    )
    if len(np.unique(target_ids)) < 2:
        raise ValueError("Multi-target training requires at least two registered target datasets.")

    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    dev_idx, test_idx = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 1)
    train_local, validation_local = next(inner.split(X[dev_idx], y[dev_idx], groups[dev_idx]))
    train_idx = dev_idx[train_local]
    validation_idx = dev_idx[validation_local]
    cv = GroupKFold(n_splits=3)
    models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=args.estimators, max_features=args.max_features, min_samples_leaf=2,
            random_state=SEED, n_jobs=args.jobs,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=args.estimators, max_features=args.max_features, min_samples_leaf=2,
            random_state=SEED, n_jobs=args.jobs,
        ),
    }
    candidates = {"MedianBaseline": DummyRegressor(strategy="median"), **models}
    cv_rows = []
    for name, estimator in candidates.items():
        print(f"Scaffold-grouped CV: {name}", flush=True)
        scores = cross_validate(
            estimator,
            X[train_idx],
            y[train_idx],
            groups=groups[train_idx],
            cv=cv,
            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
            n_jobs=1,
        )
        cv_rows.append(
            {
                "model": name,
                "CV_R2_mean": float(np.mean(scores["test_r2"])),
                "CV_RMSE_mean": float(-np.mean(scores["test_rmse"])),
                "CV_MAE_mean": float(-np.mean(scores["test_mae"])),
            }
        )

    selected_name = min(
        (row for row in cv_rows if row["model"] != "MedianBaseline"),
        key=lambda row: row["CV_RMSE_mean"],
    )["model"]
    selected = models[selected_name]
    selected.fit(X[train_idx], y[train_idx])
    validation_metrics = metric_report(y[validation_idx], selected.predict(X[validation_idx]))
    final_model = models[selected_name]
    final_model.fit(X[dev_idx], y[dev_idx])
    test_predictions = final_model.predict(X[test_idx])
    test_metrics = metric_report(y[test_idx], test_predictions)

    target_holdout_rows = []
    if not args.skip_target_holdout:
        for held_target in sorted(np.unique(target_ids)):
            held_idx = np.flatnonzero(target_ids == held_target)
            train_other = np.flatnonzero(target_ids != held_target)
            if len(held_idx) < 30 or len(train_other) < 100:
                continue
            # This deliberately excludes every measurement for one protein. It
            # is the relevant stress test for a sequence-conditioned model.
            holdout_model = ExtraTreesRegressor(
                n_estimators=args.estimators,
                max_features=args.max_features,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=args.jobs,
            )
            print(f"Leave-target-out evaluation: {held_target}", flush=True)
            holdout_model.fit(X[train_other], y[train_other])
            row = {"held_out_target": held_target, "n_train": int(len(train_other)), "n_test": int(len(held_idx))}
            row.update(metric_report(y[held_idx], holdout_model.predict(X[held_idx])))
            target_holdout_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "candidate_model.joblib"
    joblib.dump(final_model, model_path)
    pd_rows = [
        {
            "target_chembl_id": target_ids[index],
            "canonical_smiles": smiles[index],
            "scaffold_group": groups[index],
            "observed_pIC50": y[index],
            "predicted_pIC50": float(test_predictions[position]),
            "split": "scaffold_test",
        }
        for position, index in enumerate(test_idx)
    ]
    pd.DataFrame(pd_rows).to_csv(output_dir / "scaffold_test_predictions.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output_dir / "model_comparison_cv.csv", index=False)
    pd.DataFrame(target_holdout_rows).to_csv(output_dir / "leave_target_out_metrics.csv", index=False)
    report = {
        "model_version": output_dir.name,
        "status": "research_candidate_not_approved",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_feature_file": str(features_path.relative_to(ROOT)),
        "source_feature_sha256": sha256_file(features_path),
        "n_rows": int(len(y)),
        "target_ids": sorted(np.unique(target_ids).tolist()),
        "scaffold_count": int(len(np.unique(groups))),
        "target_counts": {key: int(value) for key, value in zip(*np.unique(target_ids, return_counts=True))},
        "max_rows_per_target": args.max_rows_per_target,
        "split_counts": {
            "train": int(len(train_idx)),
            "validation": int(len(validation_idx)),
            "scaffold_test": int(len(test_idx)),
        },
        "split_method": "scaffold-grouped development CV and nested validation/test split; scaffolds held intact across all targets",
        "feature_matrix": "Morgan fingerprints + RDKit descriptors + amino-acid and dipeptide protein sequence descriptors",
        "model_comparison_cv": cv_rows,
        "selected_model_by_train_cv_rmse": selected_name,
        "selected_model_parameters": final_model.get_params(deep=False),
        "validation_metrics": validation_metrics,
        "scaffold_test_metrics": test_metrics,
        "leave_target_out_metrics": target_holdout_rows,
        "leave_target_out_macro_metrics": (
            {
                "R2": float(np.mean([row["R2"] for row in target_holdout_rows])),
                "RMSE_pIC50": float(np.mean([row["RMSE_pIC50"] for row in target_holdout_rows])),
                "MAE_pIC50": float(np.mean([row["MAE_pIC50"] for row in target_holdout_rows])),
                "Spearman": float(np.mean([row["Spearman"] for row in target_holdout_rows])),
            }
            if target_holdout_rows else None
        ),
        "candidate_model_file": model_path.name,
        "candidate_model_sha256": sha256_file(model_path),
        "limitations": [
            "Leave-target-out metrics are the relevant stress test for unseen proteins; they do not replace independent assay-source validation.",
            "This multi-target candidate is not registered for predictions.",
            "Protein descriptors are sequence-composition features and do not represent binding-site structure.",
        ],
        "software": {"scikit_learn": sklearn.__version__, "numpy": np.__version__},
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
