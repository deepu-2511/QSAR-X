"""Create GitHub-compatible compressed copies of the models used by the API.

The source model artifacts remain unchanged under ``models/``.  The deployment
registry points to ``models_deploy/`` copies containing the same estimators
serialized with joblib compression, so the repository stays below GitHub's
100 MB single-file limit without Git LFS.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "target_registry.json"
OUT = ROOT / "models_deploy"


def compress(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = joblib.load(source)
    joblib.dump(model, destination, compress=3)
    print(f"{source.relative_to(ROOT)} -> {destination.relative_to(ROOT)} ({destination.stat().st_size / 1024 / 1024:.2f} MB)")


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for target in registry["targets"]:
        # Source artifacts remain under models/ even after a previous packaging run.
        original_dir = ROOT / "models" / target["target_id"] / target["model_version"]
        original = original_dir / "candidate_model.joblib"
        destination = OUT / target["target_id"] / target["model_version"] / "candidate_model.joblib"
        compress(original, destination)
        target["model_path"] = destination.relative_to(ROOT).as_posix()
        for name in ("split_assignments.csv", "evaluation_report.json", "external_validation_report.json", "quality_gate_report.json"):
            source_report = original_dir / name
            if source_report.is_file():
                copied_report = destination.parent / name
                shutil.copy2(source_report, copied_report)
                if name == "evaluation_report.json":
                    target["evaluation_report"] = copied_report.relative_to(ROOT).as_posix()
                elif name == "external_validation_report.json":
                    target["external_validation_report"] = copied_report.relative_to(ROOT).as_posix()
                elif name == "quality_gate_report.json":
                    target["quality_gate_report"] = copied_report.relative_to(ROOT).as_posix()
    legacy = ROOT / "step13_qsar_model.pkl"
    if legacy.exists():
        compress(legacy, OUT / "legacy" / "step13_qsar_model.pkl")
    manifest_path = ROOT / "models" / "multitarget" / "active_candidate.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = ROOT / "models" / "multitarget" / manifest["model_version"] / "candidate_model.joblib"
        destination = OUT / "multitarget" / manifest["model_version"] / "candidate_model.joblib"
        compress(source, destination)
        manifest["model_path"] = destination.relative_to(ROOT).as_posix()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
