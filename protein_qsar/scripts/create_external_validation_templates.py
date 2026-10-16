"""Create provenance-complete IC50 external-validation intake templates.

Templates deliberately contain no synthetic assay values. Fill them only with
independent, target-matched measurements whose source and experimental context
can be audited.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data" / "targets"
OUT = ROOT / "data" / "external_validation"
FIELDS = [
    "target_chembl_id", "target_uniprot_accession", "canonical_smiles",
    "activity_nM", "standard_type", "standard_units", "relation",
    "assay_context", "publication_id",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for metadata_path in sorted(TARGETS.glob("CHEMBL*/target_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target_id = metadata["target_chembl_id"]
        output = OUT / f"{target_id}_external_ic50.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
        manifest.append({
            "target_id": target_id,
            "uniprot_accession": metadata.get("protein_accession"),
            "target_name": metadata.get("pref_name"),
            "template": output.relative_to(ROOT).as_posix(),
            "minimum_records": 30,
        })
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# External IC50 validation intake\n\n"
        "Each target template requires independent IC50 records. Do not copy rows from the ChEMBL training CSV. "
        "Use exact `=` relations, nM units, canonical SMILES, a concise assay description (construct, assay format, conditions) in `assay_context`, and a publication or dataset identifier in `publication_id`.\n\n"
        "At least 30 post-overlap-removal compounds are required. Validate a filled template with:\n\n"
        "```powershell\n"
        "python scripts\\validate_external_target.py CHEMBLXXXX baseline_v1 `\\n"
        "  --external-csv data\\external_validation\\CHEMBLXXXX_external_ic50.csv `\\n"
        "  --source-name \"Independent source name\" `\\n"
        "  --source-url \"https://source.example\"\n"
        "```\n",
        encoding="utf-8",
    )
    print(json.dumps({"templates": len(manifest), "directory": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
