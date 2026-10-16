"""Train one scaffold-validated QSAR candidate per locally curated panel target.

The script uses every curated unique compound for each target. It does not
promote a model for API prediction: independent external validation remains a
separate gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINIMUM = 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="*", help="Optional ChEMBL target IDs.")
    parser.add_argument("--minimum-compounds", type=int, default=DEFAULT_MINIMUM)
    args = parser.parse_args()
    requested = {target.upper() for target in args.targets} if args.targets else None

    discovered = []
    for dataset in sorted((ROOT / "data" / "targets").glob("CHEMBL*/IC50_clean.csv")):
        target_id = dataset.parent.name.upper()
        if requested and target_id not in requested:
            continue
        n_compounds = len(pd.read_csv(dataset, usecols=["canonical_smiles"]))
        if n_compounds >= args.minimum_compounds:
            discovered.append((target_id, n_compounds))

    results = []
    for target_id, n_compounds in discovered:
        print(f"Training {target_id} with {n_compounds} curated compounds", flush=True)
        command = [sys.executable, str(ROOT / "scripts" / "train_target_baseline.py"), "--target-id", target_id]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        result = {"target_id": target_id, "n_compounds": n_compounds, "return_code": completed.returncode}
        if completed.returncode == 0:
            versions = sorted((ROOT / "models" / target_id).glob("baseline_v*"), key=lambda path: int(path.name.removeprefix("baseline_v")))
            report = json.loads((versions[-1] / "evaluation_report.json").read_text(encoding="utf-8"))
            result.update({"model_version": versions[-1].name, "scaffold_test_metrics": report["scaffold_test_metrics"]})
        else:
            result["error_tail"] = completed.stderr[-2000:] or completed.stdout[-2000:]
        results.append(result)
        print(json.dumps(result), flush=True)

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_compounds": args.minimum_compounds,
        "trained": results,
        "note": "Each candidate is internal scaffold-validated only. External assay-source validation is required before prediction promotion.",
    }
    path = ROOT / "models" / "panel_training_report.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
