"""Publish a multi-target candidate manifest for the research prediction endpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "multitarget"
MANIFEST = MODEL_ROOT / "active_candidate.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_version")
    parser.add_argument("--allow-research-predictions", action="store_true")
    args = parser.parse_args()
    model_dir = MODEL_ROOT / args.model_version
    report_path = model_dir / "evaluation_report.json"
    model_path = model_dir / "candidate_model.joblib"
    if not report_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Candidate model or report is missing.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    holdout = report.get("leave_target_out_macro_metrics")
    n_targets = len(report.get("target_ids", []))
    gates = {
        "at_least_10_targets": n_targets >= 10,
        "leave_target_out_available": holdout is not None,
        "leave_target_out_rmse_le_1": bool(holdout and holdout.get("RMSE_pIC50") is not None and holdout["RMSE_pIC50"] <= 1.0),
        "leave_target_out_spearman_ge_0_5": bool(holdout and holdout.get("Spearman") is not None and holdout["Spearman"] >= 0.5),
    }
    passed = all(gates.values())
    output = {
        "model_version": args.model_version,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "evaluation_report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "target_ids": report.get("target_ids", []),
        "research_prediction_enabled": bool(args.allow_research_predictions),
        "quality_gates": gates,
        "quality_gates_passed": passed,
        "status": "research_candidate_gates_passed" if passed else "research_candidate_gates_failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "Sequence-conditioned multi-target prediction is research-only and is not a replacement for a target-specific validated model."
        ),
    }
    MANIFEST.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
