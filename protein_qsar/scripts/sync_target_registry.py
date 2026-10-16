"""Build the runtime target registry from local target-specific model artifacts."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TARGETS = ROOT / "data" / "targets"


def gene(metadata: dict) -> str:
    for component in metadata.get("target_components", []):
        for synonym in component.get("target_component_synonyms", []):
            if synonym.get("syn_type") == "GENE_SYMBOL":
                return str(synonym.get("component_synonym"))
    return str(metadata.get("pref_name", "Unknown"))


def main() -> None:
    entries=[]
    for meta_path in sorted(TARGETS.glob("CHEMBL*/target_metadata.json")):
        metadata=json.loads(meta_path.read_text(encoding="utf-8"))
        target_id=metadata["target_chembl_id"]
        versions=sorted((MODELS/target_id).glob("baseline_v*"), key=lambda p: int(p.name.rsplit("v",1)[1]))
        versions=[p for p in versions if (p/"candidate_model.joblib").is_file() and (p/"evaluation_report.json").is_file()]
        if not versions:
            continue
        model_dir=versions[-1]
        evaluation=json.loads((model_dir/"evaluation_report.json").read_text(encoding="utf-8"))
        external_path=model_dir/"external_validation_report.json"
        external=json.loads(external_path.read_text(encoding="utf-8")) if external_path.is_file() else {}
        gates_passed=bool(external.get("quality_gates_passed"))
        external_metrics=external.get("metrics_on_external_source_nonoverlap", {})
        status="external_validation_gates_passed_review_required" if gates_passed else "target_specific_screening_external_review_required"
        entries.append({
            "target_id":target_id, "name":metadata.get("pref_name"), "gene":gene(metadata),
            "organism":metadata.get("organism"), "uniprot_accession":metadata.get("protein_accession"),
            "target_type":str(metadata.get("target_type", "single protein")).lower(), "endpoint":"IC50", "activity_unit":"nM",
            "model_status":status, "training_stage":"validated" if gates_passed else "external_evidence_recorded", "prediction_available":True, "prediction_mode":"target_specific_screening",
            "model_version":model_dir.name, "model_path":str(model_dir.relative_to(ROOT)/"candidate_model.joblib").replace("\\","/"),
            "evaluation_report":str(model_dir.relative_to(ROOT)/"evaluation_report.json").replace("\\","/"),
            "external_validation_report":str(external_path.relative_to(ROOT)).replace("\\","/") if external_path.is_file() else None,
            "quality_gates_passed":gates_passed,
            "validation":{
                "split":"scaffold-held-out internal test", **evaluation.get("scaffold_test_metrics", {}),
                "external_source_metrics":external_metrics,
                "external_validation_status":external.get("status", "not_recorded"),
                "warning":"External validation quality gates must pass before a model can be represented as validated."
            }
        })
    output={"registry_version":4,"targets":entries}
    (MODELS/"target_registry.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps({"registered_target_specific_models":len(entries),"targets":[x["gene"] for x in entries]},indent=2))
if __name__=="__main__": main()
