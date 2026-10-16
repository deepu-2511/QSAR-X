"""Register a collected, single-protein target and its trained candidate artifact."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "models" / "target_registry.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_id")
    parser.add_argument("model_version")
    parser.add_argument("--gene", help="Gene symbol override if ChEMBL does not provide one.")
    args = parser.parse_args()
    target_id = args.target_id.strip().upper()
    if not re.fullmatch(r"CHEMBL\d+", target_id):
        raise ValueError("target_id must look like CHEMBL203.")
    if not re.fullmatch(r"baseline_v\d+", args.model_version):
        raise ValueError("model_version must look like baseline_v1.")

    data_dir = ROOT / "data" / "targets" / target_id
    metadata_path = data_dir / "target_metadata.json"
    sequence_path = data_dir / "protein.fasta"
    model_dir = ROOT / "models" / target_id / args.model_version
    model_path = model_dir / "candidate_model.joblib"
    evaluation_path = model_dir / "evaluation_report.json"
    splits_path = model_dir / "split_assignments.csv"
    for path in (metadata_path, sequence_path, model_path, evaluation_path, splits_path):
        if not path.is_file():
            raise FileNotFoundError(f"Workflow artifact is missing: {path}")

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if meta.get("target_chembl_id", "").upper() != target_id:
        raise ValueError("Collected target metadata does not match the requested ID.")
    if evaluation.get("target_chembl_id", "").upper() != target_id:
        raise ValueError("Model evaluation report does not match the requested target.")
    if meta.get("target_type", "").upper() != "SINGLE PROTEIN":
        raise ValueError("Only single-protein targets are supported by this sequence-feature workflow.")
    protein_components = [
        component for component in meta.get("target_components", [])
        if component.get("component_type") == "PROTEIN"
    ]
    synonyms = [
        synonym.get("component_synonym")
        for component in protein_components
        for synonym in component.get("target_component_synonyms", [])
        if synonym.get("syn_type") == "GENE_SYMBOL"
    ]
    gene = args.gene or (synonyms[0] if synonyms else target_id)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    existing = next((row for row in registry["targets"] if row["target_id"].upper() == target_id), None)
    if existing and existing.get("model_status") in {"legacy_active", "validated_active"}:
        raise ValueError("Refusing to replace an active registered model through candidate registration.")
    target = {
        "target_id": target_id,
        "name": meta.get("pref_name") or target_id,
        "gene": gene,
        "organism": meta.get("organism") or "unknown",
        "uniprot_accession": meta.get("protein_accession"),
        "target_type": "single protein",
        "endpoint": "IC50",
        "activity_unit": "nM",
        "model_status": "research_candidate_pending_validation",
        "prediction_available": False,
        "prediction_mode": "research",
        "model_version": args.model_version,
        "model_path": (model_path.relative_to(ROOT)).as_posix(),
        "evaluation_report": (evaluation_path.relative_to(ROOT)).as_posix(),
        "quality_gates_passed": False,
        "validation": {
            "split": "scaffold-held-out internal test",
            "R2": evaluation.get("scaffold_test_metrics", {}).get("R2"),
            "RMSE_pIC50": evaluation.get("scaffold_test_metrics", {}).get("RMSE_pIC50", evaluation.get("scaffold_test_metrics", {}).get("RMSE")),
            "MAE_pIC50": evaluation.get("scaffold_test_metrics", {}).get("MAE_pIC50", evaluation.get("scaffold_test_metrics", {}).get("MAE")),
            "warning": "Candidate model is not enabled until independent external-validation gates have been reviewed.",
        },
    }
    if existing:
        registry["targets"][registry["targets"].index(existing)] = target
    else:
        registry["targets"].append(target)
    registry["registry_version"] = int(registry.get("registry_version", 1)) + 1
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps({key: target[key] for key in ("target_id", "gene", "model_version", "model_status", "prediction_available")}, indent=2))


if __name__ == "__main__":
    main()
