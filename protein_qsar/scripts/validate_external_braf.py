"""Evaluate the BRAF candidate on structure-disjoint BindingDB IC50 data.

BindingDB's target API does not return the relation operator or full assay
conditions. The script therefore reports the result as exploratory and fails
the promotion gate unless those fields are independently documented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "CHEMBL5145"
ACCESSION = "P15056"
TARGET_DIR = ROOT / "data" / "targets" / TARGET_ID
MODEL_DIR = ROOT / "models" / TARGET_ID / "baseline_v1"
RAW_PATH = TARGET_DIR / "BindingDB_IC50_external_raw.json"
DATA_PATH = TARGET_DIR / "BindingDB_IC50_external.csv"
PREDICTIONS_PATH = MODEL_DIR / "external_validation_predictions.csv"
REPORT_PATH = MODEL_DIR / "external_validation_report.json"
MODEL_PATH = MODEL_DIR / "candidate_model.joblib"
TRAINING_DATA = TARGET_DIR / "BRAF_IC50_clean.csv"
RADIUS = 2
BITS = 2048
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=BITS)


def request_json(accession: str, cutoff_nm: int) -> dict:
    url = (
        "https://bindingdb.org/rest/getLigandsByUniprot?"
        f"uniprot={accession};{cutoff_nm}&response=application/json"
    )
    request = Request(url, headers={"User-Agent": "QSAR-X/1.0 research validation"})
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def canonicalize(smiles: object) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_records(payload: dict) -> tuple[pd.DataFrame, dict]:
    response = payload.get("getLindsByUniprotResponse", {})
    if response.get("bdb.primary") != ACCESSION:
        raise ValueError(f"BindingDB resolved unexpected accession: {response.get('bdb.primary')!r}")
    records = response.get("bdb.affinities", [])
    rows = []
    numeric_pattern = re.compile(r"^[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
    for record in records:
        if str(record.get("bdb.affinity_type", "")).upper() != "IC50":
            continue
        raw_value = str(record.get("bdb.affinity", "")).strip()
        raw_smiles = record.get("bdb.smile")
        canonical = canonicalize(raw_smiles)
        if not canonical or not numeric_pattern.fullmatch(raw_value):
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0:
            continue
        rows.append(
            {
                "target_chembl_id": TARGET_ID,
                "target_uniprot_accession": ACCESSION,
                "canonical_smiles": canonical,
                "activity_nM": value,
                "pIC50": 9.0 - math.log10(value),
                "standard_type": "IC50",
                "standard_units": "nM",
                "relation": None,
                "source_database": "BindingDB REST",
                "bindingdb_monomer_id": record.get("bdb.monomerid"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("BindingDB returned no parseable BRAF IC50 records.")
    grouped = frame.groupby("canonical_smiles", as_index=False).agg(
        target_chembl_id=("target_chembl_id", "first"),
        target_uniprot_accession=("target_uniprot_accession", "first"),
        activity_nM=("activity_nM", "median"),
        pIC50=("pIC50", "median"),
        standard_type=("standard_type", "first"),
        standard_units=("standard_units", "first"),
        source_database=("source_database", "first"),
        bindingdb_record_count=("bindingdb_monomer_id", "size"),
        bindingdb_monomer_ids=("bindingdb_monomer_id", lambda values: "|".join(sorted({str(value) for value in values if pd.notna(value)}))),
    )
    return grouped, response


def run(cutoff_nm: int) -> dict:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists() or not TRAINING_DATA.exists():
        raise FileNotFoundError("Train the BRAF candidate model before external validation.")
    payload = request_json(ACCESSION, cutoff_nm)
    RAW_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    external, response = collect_records(payload)

    training = pd.read_csv(TRAINING_DATA)
    training_smiles = set(training["canonical_smiles"].dropna().astype(str))
    overlap_mask = external["canonical_smiles"].isin(training_smiles)
    overlapping_compounds = int(overlap_mask.sum())
    external = external.loc[~overlap_mask].reset_index(drop=True)
    DATA_PATH.write_text(external.to_csv(index=False), encoding="utf-8")

    model = joblib.load(MODEL_PATH)
    features = []
    for smiles in external["canonical_smiles"]:
        molecule = Chem.MolFromSmiles(smiles)
        vector = np.zeros((BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(GENERATOR.GetFingerprint(molecule), vector)
        features.append(vector)
    X = np.stack(features) if features else np.empty((0, BITS), dtype=np.uint8)
    y_true = external["pIC50"].to_numpy(dtype=float)
    y_pred = model.predict(X) if len(external) else np.asarray([], dtype=float)
    if len(external) >= 2:
        errors = y_pred - y_true
        metrics = {
            "n_compounds": int(len(external)),
            "R2": float(r2_score(y_true, y_pred)),
            "RMSE_pIC50": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE_pIC50": float(mean_absolute_error(y_true, y_pred)),
            "Spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
            "observed_pIC50_mean": float(np.mean(y_true)),
            "predicted_pIC50_mean": float(np.mean(y_pred)),
            "error_bias_predicted_minus_observed": float(np.mean(errors)),
        }
        predictions = external.copy()
        predictions["predicted_pIC50"] = y_pred
        predictions["residual_pIC50"] = errors
        predictions["absolute_error_pIC50"] = np.abs(errors)
        predictions.to_csv(PREDICTIONS_PATH, index=False)
    else:
        metrics = {"n_compounds": int(len(external)), "R2": None, "RMSE_pIC50": None, "MAE_pIC50": None, "Spearman": None}
        pd.DataFrame().to_csv(PREDICTIONS_PATH, index=False)

    # The REST response omits relation operators, assay context and publication
    # identifiers, so this external-source check cannot promote a model.
    gates = {
        "target_accession_matches_human_braf": response.get("bdb.primary") == ACCESSION,
        "endpoint_is_ic50": bool(len(external) == 0 or external["standard_type"].eq("IC50").all()),
        "units_are_nm": bool(len(external) == 0 or external["standard_units"].eq("nM").all()),
        "no_compound_overlap_with_chembl_target_data": not bool(
            set(external["canonical_smiles"]) & training_smiles
        ),
        "minimum_30_external_compounds": len(external) >= 30,
        "relation_and_assay_context_documented": False,
        "metrics_meet_screening_floor": bool(
            metrics["n_compounds"] >= 30
            and metrics["RMSE_pIC50"] is not None
            and metrics["RMSE_pIC50"] <= 1.0
            and metrics["MAE_pIC50"] <= 0.75
            and metrics["Spearman"] is not None
            and metrics["Spearman"] >= 0.5
        ),
    }
    report = {
        "target_chembl_id": TARGET_ID,
        "target_uniprot_accession": ACCESSION,
        "model_version": MODEL_DIR.name,
        "model_sha256": sha256_file(MODEL_PATH),
        "source_database": "BindingDB REST",
        "source_url": f"https://bindingdb.org/rest/getLigandsByUniprot?uniprot={ACCESSION};{cutoff_nm}&response=application/json",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "raw_response_file": RAW_PATH.name,
        "raw_response_sha256": sha256_file(RAW_PATH),
        "curated_external_file": DATA_PATH.name,
        "external_predictions_file": PREDICTIONS_PATH.name,
        "bindingdb_record_count_returned": int(response.get("bdb.hit", 0)),
        "parseable_unique_compounds_before_overlap_removal": int(len(external) + overlapping_compounds),
        "exact_structure_overlaps_removed": overlapping_compounds,
        "remaining_exact_structure_overlaps": int(
            len(set(external["canonical_smiles"]) & training_smiles)
        ),
        "model_training_compound_count": int(len(training_smiles)),
        "metrics_on_external_source_nonoverlap": metrics,
        "quality_gates": gates,
        "quality_gates_passed": all(gates.values()),
        "status": "external_source_exploratory_only" if not all(gates.values()) else "external_validation_gates_passed_review_required",
        "limitations": [
            "BindingDB REST returns affinity type and numeric value but omits relation operators and detailed assay conditions.",
            "Publication-level independence from ChEMBL cannot be fully audited from this REST payload.",
            "Numeric external-source results are exploratory; the quality gates fail closed while provenance fields are missing.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-nm", type=int, default=100000)
    args = parser.parse_args()
    report = run(args.cutoff_nm)
    if not report["quality_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
