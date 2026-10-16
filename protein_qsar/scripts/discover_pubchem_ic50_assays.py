"""Discover PubChem BioAssay records mentioning IC50 for each local target.

The inventory preserves assay source, protocol text and PubMed references so a
reviewer can select genuinely independent assays before numeric ingestion.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external_validation" / "pubchem_assay_inventory.json"
HEADERS = {"User-Agent": "QSAR-X/1.0 external-assay discovery"}


def fetch_json(url: str) -> dict:
    last_error = None
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(3 * (attempt + 1))
    raise last_error


def assay_summary(aid: int) -> dict | None:
    try:
        payload = fetch_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/description/JSON")
        descr = payload["PC_AssayContainer"][0]["assay"]["descr"]
        text = " ".join(descr.get("description", []) + descr.get("protocol", []) + descr.get("comment", []))
        result_names = [str(item.get("name", "")) for item in descr.get("results", [])]
        if "IC50" not in (text + " " + " ".join(result_names)).upper():
            return None
        pmids = [xref["xref"].get("pmid") for xref in descr.get("xref", []) if xref.get("xref", {}).get("pmid")]
        source = descr.get("aid_source", {}).get("db", {})
        return {
            "aid": aid,
            "name": descr.get("name"),
            "source": source.get("name"),
            "source_id": source.get("source_id", {}).get("str"),
            "publication_ids": pmids,
            "result_names": result_names,
            "assay_context": text[:4000],
            "url": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
        }
    except Exception as exc:
        return {"aid": aid, "discovery_error": str(exc)}


def discover_target(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    accession = metadata.get("protein_accession")
    target_id = metadata["target_chembl_id"]
    payload = fetch_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/accession/{accession}/aids/JSON")
    aids = payload.get("IdentifierList", {}).get("AID", [])[:20]
    candidates = []
    for aid in aids:
        item = assay_summary(int(aid))
        if item and "discovery_error" not in item:
            candidates.append(item)
        time.sleep(0.35)
    return {"target_id": target_id, "uniprot_accession": accession, "assays_checked": len(aids), "ic50_candidates": sorted(candidates, key=lambda item: item["aid"])}


def main() -> None:
    targets = sorted((ROOT / "data" / "targets").glob("CHEMBL*/target_metadata.json"))
    entries = []
    for metadata_path in targets:
        try:
            result = discover_target(metadata_path)
        except Exception as exc:
            target_id = metadata_path.parent.name
            result = {"target_id": target_id, "assays_checked": 0, "ic50_candidates": [], "discovery_error": str(exc)}
        entries.append(result)
        print(f"{result['target_id']}: {len(result['ic50_candidates'])} IC50 candidate assays from {result['assays_checked']} checked", flush=True)
    OUT.write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(), "source": "PubChem BioAssay PUG-REST", "targets": entries}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "targets": len(entries), "ic50_candidates": sum(len(item['ic50_candidates']) for item in entries)}, indent=2))


if __name__ == "__main__":
    main()
