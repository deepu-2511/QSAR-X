"""Extract auditable independent PubChem IC50 datasets from the discovered assay inventory.

Only accepts non-ChEMBL assay sources, an explicit numeric IC50 column, an explicit
unit in the field description, exact '=' relations, canonical SMILES, target
accession discovered from PubChem, assay context, and a PubMed identifier.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data" / "external_validation" / "pubchem_assay_inventory.json"
OUT_DIR = ROOT / "data" / "external_validation" / "pubchem_independent"
HEADERS = {"User-Agent": "QSAR-X/1.0 external IC50 curation"}
UNIT_FACTORS = {"nm": 1.0, "um": 1000.0, "µm": 1000.0, "mm": 1_000_000.0, "pm": 0.001, "m": 1_000_000_000.0}


def fetch_csv(aid: int) -> str:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=90) as response:
                return response.read().decode("utf-8-sig")
        except HTTPError as exc:
            if exc.code == 400:
                raise RuntimeError("assay export exceeds PubChem record limit") from exc
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def normalized_unit(description: str) -> str | None:
    text = description.lower().replace("μ", "µ")
    patterns = [("nm", r"nanomolar|\bnm\b"), ("um", r"micromolar|\b[µu]m\b"), ("mm", r"millimolar|\bmm\b"), ("pm", r"picomolar|\bpm\b"), ("m", r"\bmolar\b|\bmol/l\b")]
    for unit, pattern in patterns:
        if re.search(pattern, text):
            return unit
    return None


def extract_assay(target: dict, assay: dict) -> tuple[list[dict], dict]:
    aid = int(assay["aid"])
    text = fetch_csv(aid)
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 4:
        raise RuntimeError("CSV has no assay rows")
    headers, _types, descriptions = rows[:3]
    ic50_indices = [index for index, name in enumerate(headers) if name.strip().upper() == "IC50"]
    if len(ic50_indices) != 1:
        raise RuntimeError("requires exactly one IC50 result column")
    ic50_index = ic50_indices[0]
    unit = normalized_unit(descriptions[ic50_index])
    if unit is None:
        raise RuntimeError("IC50 unit absent from PubChem field description")
    try:
        smiles_index = headers.index("PUBCHEM_EXT_DATASOURCE_SMILES")
    except ValueError as exc:
        raise RuntimeError("SMILES absent from PubChem assay export") from exc
    qualifier_index = next((i for i, name in enumerate(headers) if name.strip().lower() in {"qualifier", "standard relation"}), None)
    if qualifier_index is None:
        raise RuntimeError("relation/qualifier absent from PubChem assay export")
    publication_ids = [str(item) for item in assay.get("publication_ids", []) if str(item).strip()]
    if not publication_ids:
        raise RuntimeError("PubChem assay has no PMID provenance")
    records = []
    for values in rows[3:]:
        values += [""] * (len(headers) - len(values))
        smiles, relation, value = values[smiles_index].strip(), values[qualifier_index].strip(), values[ic50_index].strip()
        if not smiles or relation != "=":
            continue
        try:
            activity = float(value)
        except ValueError:
            continue
        if not math.isfinite(activity) or activity <= 0:
            continue
        records.append({
            "target_chembl_id": target["target_id"],
            "target_uniprot_accession": target["uniprot_accession"],
            "canonical_smiles": smiles,
            "activity_nM": activity * UNIT_FACTORS[unit],
            "standard_type": "IC50",
            "standard_units": "nM",
            "relation": "=",
            "assay_context": f"PubChem AID {aid}; source={assay.get('source')}; {assay.get('name')}; {assay.get('assay_context', '')}",
            "publication_id": ";".join(publication_ids),
            "pubchem_aid": aid,
            "pubchem_source": assay.get("source", ""),
            "pubchem_url": assay.get("url", ""),
        })
    return records, {"aid": aid, "rows_accepted": len(records), "unit": unit, "url": assay.get("url")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    per_target: dict[str, list[dict]] = defaultdict(list)
    audit: list[dict] = []
    for target in inventory["targets"]:
        for assay in target.get("ic50_candidates", []):
            if str(assay.get("source", "")).strip().lower() == "chembl":
                continue
            try:
                records, result = extract_assay(target, assay)
                per_target[target["target_id"]].extend(records)
                audit.append({"target_id": target["target_id"], "status": "extracted", **result})
            except Exception as exc:
                audit.append({"target_id": target["target_id"], "aid": assay.get("aid"), "status": "rejected", "reason": str(exc), "url": assay.get("url")})
            time.sleep(0.5)
    files = []
    columns = ["target_chembl_id", "target_uniprot_accession", "canonical_smiles", "activity_nM", "standard_type", "standard_units", "relation", "assay_context", "publication_id", "pubchem_aid", "pubchem_source", "pubchem_url"]
    for target_id, records in sorted(per_target.items()):
        path = OUT_DIR / f"{target_id}_pubchem_independent_ic50.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        files.append({"target_id": target_id, "file": path.name, "raw_records": len(records), "unique_smiles_before_canonicalization": len({r['canonical_smiles'] for r in records})})
    manifest = {"source": "PubChem BioAssay PUG-REST", "selection": "Non-ChEMBL source, explicit IC50, explicit unit, exact relation, SMILES, PMID", "files": files, "audit": audit}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT_DIR), "files": files, "audit": audit}, indent=2))

if __name__ == "__main__":
    main()
