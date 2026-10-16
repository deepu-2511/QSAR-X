"""Apply explicit independent-validation gates to a target model version."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = ROOT / "models" / "target_registry.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_id")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--external-report", type=Path, required=True)
    parser.add_argument("--apply-registry", action="store_true")
    parser.add_argument("--approve-reviewed-model", action="store_true", help="Record that a human reviewed the passing gates and allow validated-mode predictions.")
    parser.add_argument("--reviewed-by", help="Reviewer name/ID; required with --approve-reviewed-model.")
    args = parser.parse_args()
    target_id = args.target_id.strip().upper()
    if not re.fullmatch(r"CHEMBL\d+", target_id):
        raise ValueError("target_id must look like CHEMBL203.")
    if not re.fullmatch(r"baseline_v\d+", args.model_version):
        raise ValueError("model_version must look like baseline_v1.")
    model_dir = ROOT / "models" / target_id / args.model_version
    external_path = args.external_report
    if not external_path.is_absolute():
        external_path = ROOT / external_path
    if not external_path.exists():
        raise FileNotFoundError(f"External validation report not found: {external_path}")
    external = json.loads(external_path.read_text(encoding="utf-8"))
    internal_path = model_dir / "evaluation_report.json"
    internal = json.loads(internal_path.read_text(encoding="utf-8")) if internal_path.exists() else {}

    gates = dict(external.get("quality_gates", {}))
    gates["model_version_matches_candidate"] = bool(
        internal.get("candidate_model_sha256")
        and external.get("model_version") == args.model_version
        and external.get("model_sha256") == internal.get("candidate_model_sha256")
    )
    passed = bool(gates) and all(gates.values())
    if args.approve_reviewed_model and not args.reviewed_by:
        raise ValueError("Provide --reviewed-by when recording a human approval.")
    if args.reviewed_by and not args.approve_reviewed_model:
        raise ValueError("Use --reviewed-by only with --approve-reviewed-model.")
    if args.approve_reviewed_model and not passed:
        raise ValueError("Cannot record human approval because one or more automated quality gates failed.")
    production_approval = bool(passed and args.approve_reviewed_model)
    report = {
        "target_chembl_id": target_id,
        "model_version": args.model_version,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "internal_evaluation_report": str(internal_path.relative_to(ROOT)) if internal_path.exists() else None,
        "external_validation_report": str(external_path.relative_to(ROOT)),
        "quality_gates": gates,
        "quality_gates_passed": passed,
        "decision": "passed_pending_independent_review" if passed else "reject_or_improve_model",
        "production_approval": production_approval,
        "human_review_recorded": bool(args.approve_reviewed_model),
        "reviewed_by": args.reviewed_by,
        "reason": "Automated gates do not replace assay-context review or prospective experimental validation.",
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    output = model_dir / "quality_gate_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.apply_registry:
        registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        target = next((row for row in registry["targets"] if row["target_id"] == target_id), None)
        if target is None:
            raise ValueError(f"Target {target_id} is not in the model registry.")
        metrics = external.get("metrics_on_external_source_nonoverlap", {})
        target["model_version"] = args.model_version
        target["external_validation_report"] = str(external_path.relative_to(ROOT))
        target["quality_gate_report"] = str(output.relative_to(ROOT))
        target["quality_gates_passed"] = passed
        target["prediction_available"] = True
        target["prediction_mode"] = "validated" if production_approval else "research"
        if production_approval:
            target["reviewed_by"] = args.reviewed_by
        target["model_status"] = (
            "validated_active" if production_approval
            else "candidate_gates_passed_pending_review" if passed
            else "research_candidate_gates_failed"
        )
        target.setdefault("validation", {})["external_source_metrics"] = metrics
        target["validation"]["quality_gates"] = gates
        target["validation"]["warning"] = (
            "Automated independent-validation gates passed; human review has been recorded."
            if production_approval
            else "Automated gates passed, but a human review has not been recorded; prediction remains research-only."
            if passed
            else "External validation gates failed. Predictions remain exploratory and must not be represented as validated."
        )
        registry["registry_version"] = int(registry.get("registry_version", 1)) + 1
        REGISTRY_FILE.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
