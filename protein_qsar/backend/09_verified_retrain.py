
"""
QSAR-X Professional
Verified experimental data ingestion, candidate retraining, and safe promotion.

Run from the backend folder with the project venv activated.

IMPORTANT:
- Target: human EGFR (CHEMBL203)
- Endpoint: exact IC50 / measured pIC50
- Never trains on predicted activity values.
- Never includes fixed-test compounds or their Murcko scaffolds in candidate training.
- Keeps the incumbent model unless all promotion checks pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent

MODEL_FILE = ROOT / "step13_qsar_model.pkl"
TRAIN_FILE = ROOT / "CHEMBL203_train.csv"
VALIDATION_FILE = ROOT / "CHEMBL203_validation.csv"
TEST_FILE = ROOT / "CHEMBL203_test.csv"

ARTIFACT_DIR = BACKEND / "retraining_artifacts"
BACKUP_DIR = ROOT / "model_backups"

TARGET_ID = "CHEMBL203"
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

MODEL_PARAMS = {
    "n_estimators": 400,
    "max_features": 0.3,
    "min_samples_leaf": 2,
    "max_depth": 40,
    "random_state": 42,
    "n_jobs": -1,
}

BOOTSTRAP_ITERATIONS = 2000
DEFAULT_MIN_NEW_COMPOUNDS = 5

# Candidate may not regress beyond these small tolerances.
MAX_MAE_REGRESSION = 0.01
MAX_R2_REGRESSION = 0.01

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def log(message: str) -> None:
    print(f"[QSAR-X] {message}", flush=True)


def normalize_column(name) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df: pd.DataFrame, aliases: list[str]):
    lookup = {normalize_column(c): c for c in df.columns}

    for alias in aliases:
        key = normalize_column(alias)
        if key in lookup:
            return lookup[key]

    return None


def canonicalize_smiles(smiles) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles.strip())

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def get_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)

    # Empty Murcko scaffolds are grouped together conservatively.
    return scaffold_smiles if scaffold_smiles else "__ACYCLIC__"


def make_fingerprints(smiles_list: list[str]) -> np.ndarray:
    X = np.zeros(
        (len(smiles_list), MORGAN_BITS),
        dtype=np.uint8,
    )

    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(
                f"Invalid SMILES encountered during fingerprinting: {smiles}"
            )

        fp = GENERATOR.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, X[index])

    return X


def validate_numeric_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["pIC50"] = pd.to_numeric(
        result["pIC50"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["canonical_smiles", "pIC50"]
    )

    result = result[
        np.isfinite(result["pIC50"])
    ]

    return result


# ---------------------------------------------------------------------
# Existing dataset loading
# ---------------------------------------------------------------------

def load_existing_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required existing dataset not found: {path}"
        )

    df = pd.read_csv(path)

    smiles_col = find_column(
        df,
        ["canonical_smiles", "smiles"],
    )

    label_col = find_column(
        df,
        ["pIC50"],
    )

    if smiles_col is None or label_col is None:
        raise ValueError(
            f"{path.name} must contain canonical_smiles/SMILES and pIC50."
        )

    result = pd.DataFrame({
        "canonical_smiles": df[smiles_col].astype(str).map(
            canonicalize_smiles
        ),
        "pIC50": pd.to_numeric(
            df[label_col],
            errors="coerce",
        ),
    })

    result = validate_numeric_labels(result)

    if result.empty:
        raise ValueError(
            f"No usable records in {path.name}"
        )

    return result


# ---------------------------------------------------------------------
# Experimental CSV curation
# ---------------------------------------------------------------------

def validate_upload_target(df: pd.DataFrame) -> None:
    target_id_col = find_column(
        df,
        ["target_chembl_id"],
    )

    target_name_col = find_column(
        df,
        ["target", "target_name", "protein_name"],
    )

    if target_id_col is None and target_name_col is None:
        raise ValueError(
            "Upload must identify the target using target_chembl_id "
            "or target/target_name. This model accepts human EGFR only."
        )

    if target_id_col is not None:
        values = (
            df[target_id_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        bad = values != TARGET_ID

        if bad.any():
            examples = values[bad].head(5).tolist()

            raise ValueError(
                f"Only {TARGET_ID} is accepted. "
                f"Wrong/mixed target IDs found: {examples}"
            )

    elif target_name_col is not None:
        values = (
            df[target_name_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        accepted = values.str.contains(
            r"EGFR|EPIDERMAL GROWTH FACTOR RECEPTOR|CHEMBL203",
            regex=True,
            na=False,
        )

        if not accepted.all():
            examples = values[~accepted].head(5).tolist()

            raise ValueError(
                "Some target names do not clearly identify EGFR. "
                f"Examples: {examples}"
            )


def concentration_to_nm_factor(unit: str) -> float:
    normalized = (
        str(unit)
        .strip()
        .lower()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace(" ", "")
    )

    factors = {
        "m": 1e9,
        "mol/l": 1e9,
        "molar": 1e9,
        "mm": 1e6,
        "mmol/l": 1e6,
        "um": 1e3,
        "umol/l": 1e3,
        "nm": 1.0,
        "nmol/l": 1.0,
        "pm": 1e-3,
        "pmol/l": 1e-3,
    }

    if normalized not in factors:
        raise ValueError(
            f"Unsupported concentration unit '{unit}'. "
            "Supported: M, mM, uM, nM, pM."
        )

    return factors[normalized]


def curate_uploaded_csv(input_path: Path):
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    raw = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
    )

    if raw.empty:
        raise ValueError("Uploaded CSV is empty.")

    smiles_col = find_column(
        raw,
        [
            "canonical_smiles",
            "smiles",
            "molecule_smiles",
            "structure",
        ],
    )

    if smiles_col is None:
        raise ValueError(
            "CSV must contain a SMILES column."
        )

    validate_upload_target(raw)

    type_col = find_column(
        raw,
        [
            "standard_type",
            "activity_type",
            "measurement_type",
            "endpoint",
        ],
    )

    relation_col = find_column(
        raw,
        ["standard_relation", "relation"],
    )

    units_col = find_column(
        raw,
        ["standard_units", "units", "unit"],
    )

    nm_col = find_column(
        raw,
        ["ic50_nm"],
    )

    value_col = find_column(
        raw,
        [
            "standard_value",
            "ic50_value",
            "ic50",
        ],
    )

    pic50_col = find_column(
        raw,
        ["pIC50"],
    )

    source_col = find_column(
        raw,
        [
            "source",
            "data_source",
            "source_name",
            "lab_name",
        ],
    )

    document_col = find_column(
        raw,
        [
            "document_chembl_id",
            "doc_chembl_id",
        ],
    )

    assay_col = find_column(
        raw,
        ["assay_chembl_id"],
    )

    assay_description_col = find_column(
        raw,
        [
            "assay_description",
            "assay_name",
            "protocol",
        ],
    )

    if nm_col is None and value_col is None and pic50_col is None:
        raise ValueError(
            "CSV needs measured IC50_nM, standard_value + units, "
            "or measured pIC50."
        )

    if (
        source_col is None
        and document_col is None
        and assay_col is None
    ):
        raise ValueError(
            "CSV must include provenance: source/data_source/lab_name, "
            "document_chembl_id, or assay_chembl_id."
        )

    accepted_records = []
    rejected_records = []

    for index, row in raw.iterrows():
        row_number = index + 2

        canonical = canonicalize_smiles(
            str(row[smiles_col]).strip()
        )

        if canonical is None:
            rejected_records.append({
                "row_number": row_number,
                "reason": "Invalid or empty SMILES",
            })
            continue

        # Only exact IC50 or explicitly measured pIC50 labels.
        if type_col is not None:
            endpoint = (
                str(row[type_col])
                .strip()
                .upper()
                .replace(" ", "")
            )

            if endpoint not in {"IC50", "PIC50"}:
                rejected_records.append({
                    "row_number": row_number,
                    "reason": f"Incompatible endpoint: {endpoint}",
                })
                continue

        # Exclude censored results such as <, >, <=, >=.
        if relation_col is not None:
            relation = str(row[relation_col]).strip()

            if relation not in {"", "="}:
                rejected_records.append({
                    "row_number": row_number,
                    "reason": f"Non-exact measurement relation: {relation}",
                })
                continue

        try:
            if nm_col is not None and str(row[nm_col]).strip():
                value_nm = float(row[nm_col])

                if not math.isfinite(value_nm) or value_nm <= 0:
                    raise ValueError(
                        "IC50_nM must be a positive finite value."
                    )

                pic50 = 9.0 - math.log10(value_nm)

            elif value_col is not None and str(row[value_col]).strip():
                if units_col is None:
                    raise ValueError(
                        "Concentration units are required for standard_value."
                    )

                value = float(row[value_col])

                if not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        "IC50 concentration must be positive and finite."
                    )

                value_nm = (
                    value
                    * concentration_to_nm_factor(row[units_col])
                )

                if not math.isfinite(value_nm) or value_nm <= 0:
                    raise ValueError(
                        "Converted IC50 concentration is invalid."
                    )

                pic50 = 9.0 - math.log10(value_nm)

            elif pic50_col is not None and str(row[pic50_col]).strip():
                # This column must contain experimentally measured pIC50.
                pic50 = float(row[pic50_col])

                if not math.isfinite(pic50):
                    raise ValueError(
                        "Measured pIC50 must be finite."
                    )

            else:
                raise ValueError(
                    "Missing numeric experimental activity value."
                )

            if not 0 <= pic50 <= 16:
                raise ValueError(
                    f"pIC50 {pic50:.3f} outside screening range [0,16]."
                )

        except Exception as exc:
            rejected_records.append({
                "row_number": row_number,
                "reason": str(exc),
            })
            continue

        if source_col is not None and str(row[source_col]).strip():
            source = str(row[source_col]).strip()

        elif (
            document_col is not None
            and str(row[document_col]).strip()
        ):
            source = (
                "ChEMBL document "
                + str(row[document_col]).strip()
            )

        elif assay_col is not None and str(row[assay_col]).strip():
            source = (
                "ChEMBL assay "
                + str(row[assay_col]).strip()
            )

        else:
            rejected_records.append({
                "row_number": row_number,
                "reason": "Missing provenance value",
            })
            continue

        assay_id = (
            str(row[assay_col]).strip()
            if assay_col is not None
            else ""
        )

        assay_description = (
            str(row[assay_description_col]).strip()
            if assay_description_col is not None
            else ""
        )

        accepted_records.append({
            "canonical_smiles": canonical,
            "pIC50": float(pic50),
            "source": source,
            "assay_chembl_id": assay_id,
            "assay_description": assay_description,
            "input_row": row_number,
        })

    curated = pd.DataFrame(accepted_records)
    rejected = pd.DataFrame(rejected_records)

    if curated.empty:
        raise ValueError(
            "No valid experimental records remained after curation. "
            "Review the rejected_rows report or check your CSV columns."
        )

    # Robust per-compound aggregation. Assay differences are preserved in
    # provenance columns; this aggregation does not resolve assay incompatibility.
    aggregated = (
        curated.groupby(
            "canonical_smiles",
            as_index=False,
        )
        .agg(
            pIC50=("pIC50", "median"),
            measurement_count=("pIC50", "size"),
            sources=(
                "source",
                lambda values: " | ".join(
                    sorted(set(values))
                ),
            ),
            assay_ids=(
                "assay_chembl_id",
                lambda values: " | ".join(
                    sorted(set(v for v in values if v))
                ),
            ),
            assay_descriptions=(
                "assay_description",
                lambda values: " | ".join(
                    sorted(set(v for v in values if v))
                ),
            ),
        )
    )

    return raw, curated, aggregated, rejected


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
    }


def paired_bootstrap_rmse_gain(
    y_true,
    incumbent_pred,
    candidate_pred,
    iterations=BOOTSTRAP_ITERATIONS,
):
    """
    Positive gain means candidate RMSE is lower than incumbent RMSE.
    Resamples fixed-test rows in paired fashion.
    """
    y = np.asarray(y_true, dtype=float)
    old_pred = np.asarray(incumbent_pred, dtype=float)
    new_pred = np.asarray(candidate_pred, dtype=float)

    old_squared_error = (y - old_pred) ** 2
    new_squared_error = (y - new_pred) ** 2

    rng = np.random.default_rng(20261013)
    n = len(y)

    gains = np.empty(iterations, dtype=float)

    for i in range(iterations):
        indices = rng.integers(0, n, size=n)

        old_rmse = np.sqrt(
            np.mean(old_squared_error[indices])
        )

        new_rmse = np.sqrt(
            np.mean(new_squared_error[indices])
        )

        gains[i] = old_rmse - new_rmse

    low, median, high = np.percentile(
        gains,
        [2.5, 50, 97.5],
    )

    return {
        "rmse_gain_95ci_low": float(low),
        "rmse_gain_bootstrap_median": float(median),
        "rmse_gain_95ci_high": float(high),
    }


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "QSAR-X verified experimental data retraining "
            "with safe candidate promotion."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to uploaded experimental CSV.",
    )

    parser.add_argument(
        "--min-new-compounds",
        type=int,
        default=DEFAULT_MIN_NEW_COMPOUNDS,
        help="Minimum usable new structures outside the fixed test scaffolds.",
    )

    args = parser.parse_args()

    if args.min_new_compounds < 1:
        raise ValueError(
            "--min-new-compounds must be at least 1."
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dir = ARTIFACT_DIR / f"run_{timestamp}"
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Verify all required files before doing any training.
    required_files = [
        MODEL_FILE,
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    log("Loading incumbent model and fixed datasets.")

    incumbent_model = joblib.load(MODEL_FILE)

    old_train = load_existing_dataset(TRAIN_FILE)
    old_validation = load_existing_dataset(VALIDATION_FILE)
    fixed_test = load_existing_dataset(TEST_FILE)

    # Do not use the test labels for candidate training.
    fixed_test_smiles = set(
        fixed_test["canonical_smiles"]
    )

    fixed_test_scaffolds = {
        get_scaffold(smiles)
        for smiles in fixed_test["canonical_smiles"]
    }

    fixed_test_scaffolds.discard(None)

    log(f"Fixed test compounds: {len(fixed_test):,}")

    # Curate the experimental upload.
    input_path = Path(args.input).resolve()

    log(f"Reading experimental upload: {input_path}")

    raw, curated_measurements, uploaded_compounds, rejected = (
        curate_uploaded_csv(input_path)
    )

    raw.to_csv(
        run_dir / "uploaded_original.csv",
        index=False,
    )

    curated_measurements.to_csv(
        run_dir / "curated_measurements.csv",
        index=False,
    )

    uploaded_compounds.to_csv(
        run_dir / "curated_by_compound.csv",
        index=False,
    )

    rejected.to_csv(
        run_dir / "rejected_rows.csv",
        index=False,
    )

    # Prepare existing train + validation records.
    existing_trainval = pd.concat(
        [
            old_train[["canonical_smiles", "pIC50"]],
            old_validation[["canonical_smiles", "pIC50"]],
        ],
        ignore_index=True,
    )

    existing_structures = set(
        existing_trainval["canonical_smiles"]
    )

    uploaded_compounds["scaffold"] = (
        uploaded_compounds["canonical_smiles"].map(
            get_scaffold
        )
    )

    uploaded_compounds["already_in_train_validation"] = (
        uploaded_compounds["canonical_smiles"].isin(
            existing_structures
        )
    )

    uploaded_compounds["is_fixed_test_compound"] = (
        uploaded_compounds["canonical_smiles"].isin(
            fixed_test_smiles
        )
    )

    uploaded_compounds["shares_fixed_test_scaffold"] = (
        uploaded_compounds["scaffold"].isin(
            fixed_test_scaffolds
        )
    )

    # Exclude any uploaded compound whose scaffold is in the fixed test set.
    excluded = uploaded_compounds[
        uploaded_compounds["is_fixed_test_compound"]
        | uploaded_compounds["shares_fixed_test_scaffold"]
    ].copy()

    excluded.to_csv(
        run_dir / "excluded_test_scaffolds.csv",
        index=False,
    )

    eligible_upload = uploaded_compounds[
        ~uploaded_compounds["is_fixed_test_compound"]
        & ~uploaded_compounds["shares_fixed_test_scaffold"]
    ].copy()

    genuinely_new = eligible_upload[
        ~eligible_upload["already_in_train_validation"]
    ].copy()

    log(f"Uploaded CSV rows: {len(raw):,}")
    log(f"Valid experimental measurements: {len(curated_measurements):,}")
    log(f"Unique uploaded compounds: {len(uploaded_compounds):,}")
    log(f"Rejected rows: {len(rejected):,}")
    log(f"Excluded test-scaffold compounds: {len(excluded):,}")
    log(f"New eligible compounds: {len(genuinely_new):,}")

    if len(genuinely_new) < args.min_new_compounds:
        raise ValueError(
            f"Only {len(genuinely_new)} genuinely new compounds remain. "
            f"Minimum requested: {args.min_new_compounds}. "
            "No production model was changed."
        )

    # Combine existing training data and eligible uploaded experimental labels.
    combined = pd.concat(
        [
            existing_trainval.assign(data_source="existing_train_validation"),
            eligible_upload[
                ["canonical_smiles", "pIC50"]
            ].assign(data_source="uploaded_experimental"),
        ],
        ignore_index=True,
    )

    # Aggregate repeated structures using median activity.
    combined = (
        combined.groupby(
            "canonical_smiles",
            as_index=False,
        )
        .agg(
            pIC50=("pIC50", "median"),
            measurement_count=("pIC50", "size"),
        )
    )

    # Hard safety assertions.
    if set(combined["canonical_smiles"]) & fixed_test_smiles:
        raise RuntimeError(
            "Safety stop: fixed-test structure leaked into training."
        )

    combined["scaffold"] = (
        combined["canonical_smiles"].map(
            get_scaffold
        )
    )

    if combined["scaffold"].isin(
        fixed_test_scaffolds
    ).any():
        raise RuntimeError(
            "Safety stop: fixed-test scaffold leaked into training."
        )

    combined = combined.drop(
        columns=["scaffold"]
    )

    combined.to_csv(
        run_dir / "candidate_training_data.csv",
        index=False,
    )

    log(f"Candidate training structures: {len(combined):,}")
    log("Creating Morgan fingerprints.")

    X_train = make_fingerprints(
        combined["canonical_smiles"].tolist()
    )

    y_train = combined["pIC50"].to_numpy(
        dtype=float
    )

    X_test = make_fingerprints(
        fixed_test["canonical_smiles"].tolist()
    )

    y_test = fixed_test["pIC50"].to_numpy(
        dtype=float
    )

    log("Fitting candidate ExtraTrees model.")

    candidate_model = ExtraTreesRegressor(
        **MODEL_PARAMS
    )

    candidate_model.fit(
        X_train,
        y_train,
    )

    log("Evaluating incumbent and candidate on identical fixed test compounds.")

    incumbent_predictions = np.asarray(
        incumbent_model.predict(X_test),
        dtype=float,
    )

    candidate_predictions = np.asarray(
        candidate_model.predict(X_test),
        dtype=float,
    )

    incumbent_metrics = calculate_metrics(
        y_test,
        incumbent_predictions,
    )

    candidate_metrics = calculate_metrics(
        y_test,
        candidate_predictions,
    )

    bootstrap = paired_bootstrap_rmse_gain(
        y_test,
        incumbent_predictions,
        candidate_predictions,
    )

    checks = {
        "bootstrap_95ci_confirms_rmse_improvement": (
            bootstrap["rmse_gain_95ci_low"] > 0.0
        ),
        "mae_not_materially_worse": (
            candidate_metrics["MAE"]
            <= incumbent_metrics["MAE"] + MAX_MAE_REGRESSION
        ),
        "r2_not_materially_worse": (
            candidate_metrics["R2"]
            >= incumbent_metrics["R2"] - MAX_R2_REGRESSION
        ),
    }

    promote = all(checks.values())

    candidate_path = (
        run_dir / "candidate_model.pkl"
    )

    joblib.dump(
        candidate_model,
        candidate_path,
    )

    # Save detailed evaluation before considering promotion.
    metrics_table = pd.DataFrame([
        {
            "model": "incumbent",
            **incumbent_metrics,
        },
        {
            "model": "candidate",
            **candidate_metrics,
        },
    ])

    metrics_table.to_csv(
        run_dir / "test_metrics.csv",
        index=False,
    )

    predictions_table = pd.DataFrame({
        "canonical_smiles": fixed_test["canonical_smiles"],
        "observed_pIC50": y_test,
        "incumbent_predicted_pIC50": incumbent_predictions,
        "candidate_predicted_pIC50": candidate_predictions,
        "incumbent_absolute_error": np.abs(
            y_test - incumbent_predictions
        ),
        "candidate_absolute_error": np.abs(
            y_test - candidate_predictions
        ),
    })

    predictions_table.to_csv(
        run_dir / "fixed_test_predictions.csv",
        index=False,
    )

    report = {
        "run_timestamp": timestamp,
        "target_chembl_id": TARGET_ID,
        "endpoint": "IC50",
        "fingerprint": {
            "type": "Morgan",
            "radius": MORGAN_RADIUS,
            "bits": MORGAN_BITS,
        },
        "model": "ExtraTreesRegressor",
        "model_parameters": MODEL_PARAMS,
        "fixed_test_compounds": int(len(fixed_test)),
        "uploaded_rows": int(len(raw)),
        "valid_measurement_rows": int(len(curated_measurements)),
        "unique_uploaded_compounds": int(len(uploaded_compounds)),
        "rejected_rows": int(len(rejected)),
        "excluded_test_scaffold_compounds": int(len(excluded)),
        "new_eligible_compounds": int(len(genuinely_new)),
        "candidate_training_compounds": int(len(combined)),
        "incumbent_metrics": incumbent_metrics,
        "candidate_metrics": candidate_metrics,
        "paired_bootstrap_rmse_gain": bootstrap,
        "acceptance_checks": checks,
        "promote_candidate": bool(promote),
        "production_model_path": str(MODEL_FILE),
        "candidate_model_path": str(candidate_path),
        "warning": (
            "The fixed test set has been used in previous model development. "
            "It is not a pristine external validation set. Use an independent "
            "dataset for publication-grade claims."
        ),
    }

    report_path = (
        run_dir / "evaluation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    log(f"Incumbent metrics: {incumbent_metrics}")
    log(f"Candidate metrics: {candidate_metrics}")
    log(f"Paired bootstrap RMSE gain: {bootstrap}")
    log(f"Acceptance checks: {checks}")
    log(f"Detailed report: {report_path}")
    log(f"Candidate model: {candidate_path}")

    if not promote:
        log("CANDIDATE REJECTED: production model remains unchanged.")
        log(f"Current production model: {MODEL_FILE}")
        return

    # Back up production model before replacement.
    backup_path = (
        BACKUP_DIR
        / f"step13_qsar_model_{timestamp}.pkl"
    )

    shutil.copy2(
        MODEL_FILE,
        backup_path,
    )

    # Validate candidate artifact can be loaded before replacing production.
    checked_model = joblib.load(candidate_path)

    if not hasattr(checked_model, "predict"):
        raise RuntimeError(
            "Candidate artifact does not expose predict()."
        )

    # Atomic replacement on the same filesystem.
    temporary_model_path = (
        MODEL_FILE.with_name(
            MODEL_FILE.name + ".new"
        )
    )

    shutil.copy2(
        candidate_path,
        temporary_model_path,
    )

    os.replace(
        temporary_model_path,
        MODEL_FILE,
    )

    # Save the accepted training reference for future domain calculations.
    combined.to_csv(
        ROOT / "accepted_model_training_reference.csv",
        index=False,
    )

    report["production_model_promoted"] = True
    report["production_backup_path"] = str(backup_path)

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    log("CANDIDATE PROMOTED: all acceptance checks passed.")
    log(f"Previous production model backup: {backup_path}")
    log("Restart the API process to load the promoted model.")
    log(
        "Important: update predictor.py applicability-domain reference "
        "loading to use accepted_model_training_reference.csv after promotion."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\n[QSAR-X ERROR] {exc}",
            file=sys.stderr,
        )
        print(
            "The production model was not promoted by this failed run.",
            file=sys.stderr,
        )
        sys.exit(1)
