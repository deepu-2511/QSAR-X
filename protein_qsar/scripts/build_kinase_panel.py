"""Collect and curate a reproducible, broad human single-protein kinase panel.

This is a data-building job for the multi-target research model. Every entry is
resolved by ChEMBL before collection; unresolved/non-single-protein entries are
recorded and skipped instead of guessed.
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from target_pipeline import build_common_dataset, curate_target, download_target, fetch_target


ROOT = Path(__file__).resolve().parents[1]
PANEL_FILE = ROOT / "data" / "common" / "kinase_panel_report.json"

# Curated starting panel. ChEMBL validates every ID, protein type and organism.
PANEL = {
    "EGFR": "CHEMBL203",
    "BRAF": "CHEMBL5145",
    "ABL1": "CHEMBL1862",
    "JAK2": "CHEMBL2971",
    "CDK2": "CHEMBL301",
    "MAP2K1": "CHEMBL3587",
    "MAPK14": "CHEMBL260",
    "AKT1": "CHEMBL4282",
    "KDR": "CHEMBL279",
    "ERBB2": "CHEMBL1824",
    "SRC": "CHEMBL267",
    "MTOR": "CHEMBL2842",
    "PIK3CA": "CHEMBL4005",
    "FGFR1": "CHEMBL3650",
    "MET": "CHEMBL3717",
    "RET": "CHEMBL2041",
    "KIT": "CHEMBL1936",
    "PDGFRA": "CHEMBL2007",
    "IGF1R": "CHEMBL1957",
    "FLT3": "CHEMBL1974",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Refresh raw ChEMBL files that already exist.")
    parser.add_argument("--minimum-compounds", type=int, default=500)
    parser.add_argument("--targets", nargs="*", help="Optional ChEMBL IDs to run from the panel.")
    args = parser.parse_args()
    requested = {target.upper() for target in args.targets} if args.targets else None
    completed, skipped, failed = [], [], []
    for gene, target_id in PANEL.items():
        if requested and target_id not in requested:
            continue
        try:
            metadata = fetch_target(target_id)
            if metadata.get("target_type") != "SINGLE PROTEIN" or metadata.get("organism") != "Homo sapiens":
                skipped.append({"gene": gene, "target_id": target_id, "reason": "not a human single protein"})
                continue
            target_dir = ROOT / "data" / "targets" / target_id
            raw_path = target_dir / "IC50_raw.csv"
            if args.overwrite or not raw_path.exists():
                print(f"Collecting {gene} ({target_id})", flush=True)
                download_target(target_id, "IC50", "nM", overwrite=True)
            print(f"Curating {gene} ({target_id})", flush=True)
            curate_target(target_id, "IC50", "nM")
            clean_path = target_dir / "IC50_clean.csv"
            import pandas as pd
            count = int(len(pd.read_csv(clean_path)))
            if count < args.minimum_compounds:
                skipped.append({"gene": gene, "target_id": target_id, "reason": f"only {count} curated compounds", "n_compounds": count})
            else:
                completed.append({"gene": gene, "target_id": target_id, "n_compounds": count, "uniprot": metadata.get("protein_accession")})
        except Exception as exc:
            failed.append({"gene": gene, "target_id": target_id, "error": str(exc), "traceback": traceback.format_exc(limit=1)})
            print(f"FAILED {gene} ({target_id}): {exc}", flush=True)
    if len(completed) >= 2:
        output = ROOT / "data" / "common" / "kinase_panel_ic50.csv"
        build_common_dataset([row["target_id"] for row in completed], "IC50", "nM", output)
    else:
        output = None
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "panel_requested": PANEL,
        "minimum_compounds": args.minimum_compounds,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "common_dataset": str(output.relative_to(ROOT)) if output else None,
    }
    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    PANEL_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
