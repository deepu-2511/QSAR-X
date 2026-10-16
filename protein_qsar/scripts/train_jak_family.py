"""Train and evaluate a conservative single-target Morgan-fingerprint baseline.

The input is a curated molecule-level CSV with canonical_smiles and pIC50.
Murcko-scaffold groups are kept intact across development, validation, and test
splits. The final held-out test is internal validation, not external validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate


ROOT = Path(__file__).resolve().parents[1]
RADIUS = 2
FP_SIZE = 2048
SEED = 42
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)


def fingerprint(smiles: str) -> tuple[np.ndarray, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES in curated data: {smiles!r}")
    vector = np.zeros((FP_SIZE,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(GENERATOR.GetFingerprint(mol), vector)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    # Empty Murcko scaffolds are common for acyclic compounds; use the full
    # canonical structure as a group so unrelated acyclic molecules do not
    # collapse into one artificial group.
    group = scaffold if scaffold else f"ACYCLIC:{smiles}"
    return vector, group


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(target_id: str, input_path: Path | None) -> Path:
    if input_path is not None:
        path = input_path if input_path.is_absolute() else ROOT / input_path
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")
        return path
    target_dir = ROOT / "data" / "targets" / target_id
    candidates = [target_dir / "IC50_clean.csv", *sorted(target_dir.glob("*_IC50_clean.csv"))]
    if target_id == "CHEMBL203":
        candidates.append(ROOT / "CHEMBL203_IC50_clean.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No cleaned IC50 data found for {target_id}. Run target_pipeline.py collect and curate first."
    )


def next_model_version(target_id: str) -> Path:
    target_model_dir = ROOT / "models" / target_id
    versions = []
    for path in target_model_dir.glob("baseline_v*"):
        try:
            versions.append(int(path.name.removeprefix("baseline_v")))
        except ValueError:
            continue
    return target_model_dir / f"baseline_v{max(versions, default=0) + 1}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.target_id = args.target_id.strip().upper()
    if args.target_id != "JAK_FAMILY":
        raise ValueError("This trainer is reserved for JAK_FAMILY.")
    args.input = resolve_input(args.target_id, args.input)
    args.out = args.out or next_model_version(args.target_id)
    if not args.out.is_absolute():
        args.out = ROOT / args.out

    df = pd.read_csv(args.input)
    required = {"target_chembl_id", "canonical_smiles", "pIC50"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Curated CSV is missing required columns: {missing}")
    if set(df["target_chembl_id"].dropna().astype(str)) != {args.target_id}:
        raise ValueError("Input dataset target does not match --target-id")
    df = df.dropna(subset=["canonical_smiles", "pIC50"]).copy()
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df.dropna(subset=["pIC50"]).reset_index(drop=True)
    if df["canonical_smiles"].duplicated().any():
        raise ValueError("Curated data must have one row per canonical structure")

    encoded = [fingerprint(str(smiles)) for smiles in df["canonical_smiles"]]
    X = np.stack([item[0] for item in encoded])
    groups = np.asarray([item[1] for item in encoded], dtype=object)
    y = df["pIC50"].to_numpy(dtype=float)
    unique_groups = np.unique(groups)
    if len(df) < 100 or len(unique_groups) < 10:
        raise ValueError("Insufficient data/scaffold diversity for this baseline workflow")

    # Reserve a scaffold-disjoint final test first; model selection never sees it.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    dev_idx, test_idx = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 1)
    train_local, val_local = next(inner.split(X[dev_idx], y[dev_idx], groups[dev_idx]))
    train_idx = dev_idx[train_local]
    val_idx = dev_idx[val_local]

    if set(groups[train_idx]) & set(groups[val_idx]) or set(groups[dev_idx]) & set(groups[test_idx]):
        raise RuntimeError("Scaffold leakage detected between data splits")

    models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=160, max_features=0.3, min_samples_leaf=2,
            max_depth=40, random_state=SEED, n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=160, max_features=0.3, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        ),
    }
    dummy = DummyRegressor(strategy="median")
    cv = GroupKFold(n_splits=3)
    cv_results: list[dict[str, object]] = []
    for name, estimator in {"MedianBaseline": dummy, **models}.items():
        print(f"Cross-validating {name} on scaffold-grouped folds...", flush=True)
        scores = cross_validate(
            estimator,
            X[train_idx],
            y[train_idx],
            groups=groups[train_idx],
            cv=cv,
            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
            n_jobs=1,
            return_train_score=False,
        )
        cv_results.append(
            {
                "model": name,
                "CV_R2_mean": float(np.mean(scores["test_r2"])),
                "CV_R2_std": float(np.std(scores["test_r2"], ddof=1)),
                "CV_RMSE_mean": float(-np.mean(scores["test_rmse"])),
                "CV_RMSE_std": float(np.std(-scores["test_rmse"], ddof=1)),
                "CV_MAE_mean": float(-np.mean(scores["test_mae"])),
                "CV_MAE_std": float(np.std(-scores["test_mae"], ddof=1)),
            }
        )

    eligible_cv = [row for row in cv_results if row["model"] != "MedianBaseline"]
    selected_name = min(eligible_cv, key=lambda row: row["CV_RMSE_mean"])["model"]
    selected_model = models[str(selected_name)]
    selected_model.fit(X[train_idx], y[train_idx])
    validation_metrics = metrics(y[val_idx], selected_model.predict(X[val_idx]))

    # Refit on train + validation only after selection; evaluate the test once.
    final_model = models[str(selected_name)]
    final_model.fit(X[dev_idx], y[dev_idx])
    test_metrics = metrics(y[test_idx], final_model.predict(X[test_idx]))

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "candidate_model.joblib"
    joblib.dump(final_model, model_path)
    split_df = pd.DataFrame(
        {
            "canonical_smiles": df["canonical_smiles"],
            "pIC50": y,
            "scaffold_group": groups,
            "split": "",
        }
    )
    split_df.loc[train_idx, "split"] = "train"
    split_df.loc[val_idx, "split"] = "validation"
    split_df.loc[test_idx, "split"] = "scaffold_test"
    split_df.to_csv(args.out / "split_assignments.csv", index=False)
    pd.DataFrame(cv_results).to_csv(args.out / "model_comparison_cv.csv", index=False)

    report = {
        "target_chembl_id": args.target_id,
        "endpoint": "user-supplied pChEMBL value",
        "target_value": "User-supplied pChEMBL value; per-row endpoint metadata unavailable",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only_not_production_approved",
        "dataset": args.input.resolve().relative_to(ROOT.resolve()).as_posix(),
        "dataset_sha256": sha256_file(args.input),
        "n_compounds": int(len(df)),
        "n_scaffolds": int(len(unique_groups)),
        "fingerprint": {"type": "Morgan", "radius": RADIUS, "bits": FP_SIZE},
        "split_method": "two-stage GroupShuffleSplit by chirality-aware Bemis-Murcko scaffold; intended 64/16/20 proportions are approximate because groups remain intact",
        "split_counts": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "scaffold_test": int(len(test_idx)),
            "train_scaffolds": int(len(np.unique(groups[train_idx]))),
            "validation_scaffolds": int(len(np.unique(groups[val_idx]))),
            "test_scaffolds": int(len(np.unique(groups[test_idx]))),
        },
        "cv_model_comparison": cv_results,
        "selected_model_by_train_cv_rmse": str(selected_name),
        "selected_model_parameters": final_model.get_params(deep=False),
        "candidate_model_file": model_path.name,
        "candidate_model_sha256": sha256_file(model_path),
        "validation_metrics": validation_metrics,
        "scaffold_test_metrics": test_metrics,
        "limitations": [
            "Scaffold-held-out test is internal validation, not independent external validation.",
            "User source lacks per-row JAK1/JAK2/JAK3 labels, assay type, relation, units, and provenance.",
            "All unique source structures overlap the current local JAK2 training set; no independent test claim is possible.",
            "Do not claim performance for other targets, endpoints, or out-of-domain chemistry.",
            "Model is a research candidate and is not registered for production prediction.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__,
        },
    }
    (args.out / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
