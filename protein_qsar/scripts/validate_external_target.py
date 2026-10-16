"""Evaluate a target candidate on a user-supplied, normalized external IC50 dataset.

Input CSV columns: target_chembl_id, target_uniprot_accession, canonical_smiles,
activity_nM, standard_type, standard_units, relation, assay_context, publication_id.
Missing provenance fields fail the promotion gates rather than being inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "models" / "target_registry.json"
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canonicalize(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    molecule = Chem.MolFromSmiles(value.strip())
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_id")
    parser.add_argument("model_version")
    parser.add_argument("--external-csv", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--max-rmse", type=float, default=1.0)
    parser.add_argument("--max-mae", type=float, default=0.75)
    parser.add_argument("--min-spearman", type=float, default=0.5)
    args = parser.parse_args()
    target_id = args.target_id.strip().upper()
    if not re.fullmatch(r"CHEMBL\d+", target_id):
        raise ValueError("target_id must look like CHEMBL203.")
    if not re.fullmatch(r"baseline_v\d+", args.model_version):
        raise ValueError("model_version must look like baseline_v1.")
    external_path = args.external_csv.resolve()
    if not external_path.is_file():
        raise FileNotFoundError(external_path)
    model_dir = ROOT / "models" / target_id / args.model_version
    model_path = model_dir / "candidate_model.joblib"
    eval_path = model_dir / "evaluation_report.json"
    split_path = model_dir / "split_assignments.csv"
    meta_path = ROOT / "data" / "targets" / target_id / "target_metadata.json"
    for path in (model_path, eval_path, split_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required target workflow artifact is missing: {path}")
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if evaluation.get("target_chembl_id", "").upper() != target_id:
        raise ValueError("Candidate report target ID mismatch.")
    expected_accession = metadata.get("protein_accession")

    frame = pd.read_csv(external_path, low_memory=False)
    required = {
        "target_chembl_id", "target_uniprot_accession", "canonical_smiles", "activity_nM",
        "standard_type", "standard_units", "relation", "assay_context", "publication_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"External CSV missing required columns: {missing}")
    input_quality = {
        "target_chembl_id_matches": bool(frame["target_chembl_id"].fillna("").astype(str).str.upper().eq(target_id).all()),
        "target_accession_matches": bool(expected_accession and frame["target_uniprot_accession"].fillna("").astype(str).eq(expected_accession).all()),
        "endpoint_is_ic50": bool(frame["standard_type"].fillna("").astype(str).str.upper().eq("IC50").all()),
        "units_are_nm": bool(frame["standard_units"].fillna("").astype(str).str.lower().eq("nm").all()),
        "relation_exact": bool(frame["relation"].fillna("").astype(str).str.strip().eq("=").all()),
        "assay_context_documented": bool(frame["assay_context"].fillna("").astype(str).str.strip().ne("").all()),
        "publication_id_documented": bool(frame["publication_id"].fillna("").astype(str).str.strip().ne("").all()),
    }
    frame["canonical_smiles"] = frame["canonical_smiles"].map(canonicalize)
    frame["activity_nM"] = pd.to_numeric(frame["activity_nM"], errors="coerce")
    frame = frame.dropna(subset=["canonical_smiles", "activity_nM"]).copy()
    frame = frame[frame["activity_nM"] > 0]
    frame["pIC50"] = 9.0 - np.log10(frame["activity_nM"].astype(float))
    frame = frame.groupby("canonical_smiles", as_index=False).agg(
        target_chembl_id=("target_chembl_id", "first"),
        target_uniprot_accession=("target_uniprot_accession", "first"),
        activity_nM=("activity_nM", "median"),
        pIC50=("pIC50", "median"),
        standard_type=("standard_type", "first"),
        standard_units=("standard_units", "first"),
        relation=("relation", "first"),
        assay_context=("assay_context", "first"),
        publication_id=("publication_id", "first"),
    )

    training_path = ROOT / "data" / "targets" / target_id / "IC50_clean.csv"
    if not training_path.is_file():
        raise FileNotFoundError(f"Curated training data not found: {training_path}")
    training = pd.read_csv(training_path, low_memory=False)
    training_structures = set(training["canonical_smiles"].dropna().astype(str))
    training_publications = set()
    if "document_ids" in training:
        for value in training["document_ids"].dropna().astype(str):
            training_publications.update(part.strip() for part in value.split("|") if part.strip())
    overlap = frame["canonical_smiles"].isin(training_structures)
    overlap_count = int(overlap.sum())
    frame = frame.loc[~overlap].reset_index(drop=True)
    external_publications = set()
    for value in frame["publication_id"].dropna().astype(str):
        external_publications.update(part.strip() for part in re.split(r"[|;,]", value) if part.strip())
    publication_overlap = external_publications & training_publications

    model = joblib.load(model_path)
    feature_rows = []
    for smiles in frame["canonical_smiles"]:
        molecule = Chem.MolFromSmiles(smiles)
        vector = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(GENERATOR.GetFingerprint(molecule), vector)
        feature_rows.append(vector)
    X = np.stack(feature_rows) if feature_rows else np.empty((0, 2048), dtype=np.uint8)
    y_true = frame["pIC50"].to_numpy(float)
    y_pred = model.predict(X) if len(frame) else np.asarray([], dtype=float)
    if len(frame) >= 2:
        errors = y_pred - y_true
        spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        metrics = {
            "n_compounds": int(len(frame)),
            "R2": float(r2_score(y_true, y_pred)),
            "RMSE_pIC50": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE_pIC50": float(mean_absolute_error(y_true, y_pred)),
            "Spearman": float(spearman) if pd.notna(spearman) else None,
        }
        predictions = frame.copy()
        predictions["predicted_pIC50"] = y_pred
        predictions["residual_pIC50"] = errors
        predictions["absolute_error_pIC50"] = np.abs(errors)
        predictions.to_csv(model_dir / "external_validation_predictions.csv", index=False)
    else:
        metrics = {"n_compounds": int(len(frame)), "R2": None, "RMSE_pIC50": None, "MAE_pIC50": None, "Spearman": None}

    split_data = pd.read_csv(split_path)
    candidate_digest = sha256(model_path)
    internal_digest = evaluation.get("candidate_model_sha256")
    gates = {
        **input_quality,
        "no_compound_overlap_with_training": not bool(set(frame["canonical_smiles"]) & training_structures),
        "no_publication_overlap_with_training": not publication_overlap,
        "minimum_30_external_compounds": len(frame) >= 30,
        "metrics_meet_screening_floor": bool(
            metrics["n_compounds"] >= 30
            and metrics["RMSE_pIC50"] is not None and metrics["RMSE_pIC50"] <= args.max_rmse
            and metrics["MAE_pIC50"] is not None and metrics["MAE_pIC50"] <= args.max_mae
            and metrics["Spearman"] is not None and metrics["Spearman"] >= args.min_spearman
        ),
        "model_version_and_hash_match": bool(internal_digest and candidate_digest == internal_digest),
        "external_source_provided": bool(args.source_name.strip() and args.source_url.strip()),
    }
    output = {
        "target_chembl_id": target_id,
        "target_uniprot_accession": expected_accession,
        "model_version": args.model_version,
        "model_sha256": candidate_digest,
        "source_database": args.source_name,
        "source_url": args.source_url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "external_input_file": external_path.name,
        "external_input_sha256": sha256(external_path),
        "external_predictions_file": "external_validation_predictions.csv",
        "training_scaffold_count": int(split_data["scaffold_group"].nunique()) if "scaffold_group" in split_data else None,
        "exact_structure_overlaps_removed": overlap_count,
        "metrics_on_external_source_nonoverlap": metrics,
        "quality_gates": gates,
        "quality_gates_passed": all(gates.values()),
        "status": "external_validation_gates_passed_review_required" if all(gates.values()) else "external_validation_gates_failed",
        "limitations": [
            "External assay data must represent the same target construct and IC50 endpoint.",
            "Passing automated gates is not proof of prospective experimental performance or publication-level independence.",
            "A human review of assay provenance remains necessary before production promotion.",
        ],
    }
    report_path = model_dir / "external_validation_report.json"
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
