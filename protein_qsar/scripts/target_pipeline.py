"""Reusable ChEMBL target-data collection and common-schema preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "targets"
REGISTRY_FILE = ROOT / "models" / "target_registry.json"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"
PAGE_SIZE = 1000
COMMON_COLUMNS = [
    "target_chembl_id", "target_gene", "target_name", "organism",
    "standard_type", "standard_units", "activity_nM", "pIC50",
    "canonical_smiles", "measurement_count", "source_database",
    "molecule_ids", "assay_ids", "document_ids", "aggregation",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str, retries: int = 4) -> dict:
    last_error = None
    for attempt in range(retries):
        request = Request(url, headers={"User-Agent": "QSAR-X/1.0 research workflow"})
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}: {last_error}")


def fetch_target(target_id: str) -> dict:
    target_id = target_id.strip().upper()
    if not re.fullmatch(r"CHEMBL\d+", target_id):
        raise ValueError("Target must be a ChEMBL ID such as CHEMBL203.")
    target = get_json(f"{CHEMBL_API}/target/{target_id}.json")
    if target.get("target_chembl_id", "").upper() != target_id:
        raise ValueError(f"ChEMBL did not resolve target {target_id} exactly.")
    components = target.get("target_components") or []
    protein_accession = next(
        (item.get("accession") for item in components if item.get("component_type") == "PROTEIN" and item.get("accession")),
        None,
    )
    return {
        "target_chembl_id": target_id,
        "pref_name": target.get("pref_name"),
        "target_type": target.get("target_type"),
        "organism": target.get("organism"),
        "protein_accession": protein_accession,
        "target_components": components,
    }


def collect_activity(target_id: str, endpoint: str, units: str) -> list[dict]:
    offset = 0
    rows: list[dict] = []
    while True:
        params = urlencode(
            {
                "target_chembl_id": target_id,
                "standard_type": endpoint,
                "standard_relation": "=",
                "standard_units": units,
                "standard_value__isnull": "false",
                "molecule_chembl_id__isnull": "false",
                "limit": PAGE_SIZE,
                "offset": offset,
            }
        )
        page = get_json(f"{CHEMBL_API}/activity.json?{params}")
        batch = page.get("activities") or []
        if not batch:
            break
        rows.extend(batch)
        total = (page.get("page_meta") or {}).get("total_count")
        print(f"Fetched {len(rows)}" + (f" / {total}" if total is not None else "") + " activity rows", flush=True)
        if len(batch) < PAGE_SIZE or (total is not None and len(rows) >= int(total)):
            break
        offset += len(batch)
    return rows


def download_target(target_id: str, endpoint: str, units: str, overwrite: bool) -> None:
    target = fetch_target(target_id)
    target_id = target["target_chembl_id"]
    out_dir = DATA_ROOT / target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{endpoint.upper()}_raw.csv"
    metadata_path = out_dir / "target_metadata.json"
    if raw_path.exists() and not overwrite:
        raise FileExistsError(f"{raw_path} already exists; pass --overwrite to replace it.")

    records = collect_activity(target_id, endpoint.upper(), units)
    frame = pd.DataFrame(records)
    columns = [
        "activity_id", "assay_chembl_id", "assay_type", "molecule_chembl_id",
        "target_chembl_id", "standard_type", "standard_relation", "standard_value",
        "standard_units", "pchembl_value", "canonical_smiles", "document_chembl_id",
    ]
    for column in columns:
        if column not in frame:
            frame[column] = None
    frame = frame[columns]
    frame.to_csv(raw_path, index=False)

    sequence_path = None
    accession = target.get("protein_accession")
    if accession:
        fasta = urlopen(
            Request(
                f"{UNIPROT_API}/{accession}.fasta",
                headers={"User-Agent": "QSAR-X/1.0 research workflow"},
            ),
            timeout=60,
        ).read().decode("utf-8")
        sequence_path = out_dir / "protein.fasta"
        sequence_path.write_text(fasta, encoding="utf-8")

    metadata = {
        **target,
        "source_database": "ChEMBL",
        "source_api": f"{CHEMBL_API}/activity.json",
        "endpoint": endpoint.upper(),
        "units": units,
        "relation_filter": "=",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "raw_activity_file": raw_path.name,
        "raw_activity_sha256": sha256_file(raw_path),
        "raw_activity_rows": int(len(frame)),
        "protein_sequence_file": sequence_path.name if sequence_path else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "target_components"}, indent=2))


def canonicalize(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    molecule = Chem.MolFromSmiles(value.strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def curate_target(target_id: str, endpoint: str, units: str) -> None:
    target_id = target_id.strip().upper()
    out_dir = DATA_ROOT / target_id
    raw_path = out_dir / f"{endpoint.upper()}_raw.csv"
    metadata_path = out_dir / "target_metadata.json"
    if not raw_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Run collect first for {target_id}; expected {raw_path} and {metadata_path}.")
    target = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = pd.read_csv(raw_path, low_memory=False)
    raw["standard_value"] = pd.to_numeric(raw["standard_value"], errors="coerce")
    relation = raw["standard_relation"].fillna("").astype(str).str.strip()
    types = raw["standard_type"].fillna("").astype(str).str.strip().str.upper()
    unit_values = raw["standard_units"].fillna("").astype(str).str.strip().str.lower()
    eligible = (
        raw["target_chembl_id"].fillna("").astype(str).str.upper().eq(target_id)
        & types.eq(endpoint.upper())
        & relation.eq("=")
        & unit_values.eq(units.lower())
        & raw["standard_value"].gt(0)
    )
    selected = raw.loc[eligible].copy()
    selected["canonical_smiles"] = selected["canonical_smiles"].map(canonicalize)
    invalid_smiles = int(selected["canonical_smiles"].isna().sum())
    selected = selected.dropna(subset=["canonical_smiles"])
    if selected.empty:
        raise ValueError(f"No valid exact positive {endpoint} measurements in {units} for {target_id}.")

    if endpoint.upper() == "IC50" and units.lower() == "nm":
        selected["activity_nM"] = selected["standard_value"].astype(float)
        selected["pIC50"] = 9.0 - selected["activity_nM"].map(math.log10)
    else:
        raise ValueError("This baseline curation currently standardizes IC50 in nM to pIC50.")

    def join_ids(values: pd.Series) -> str:
        return "|".join(sorted({str(value) for value in values.dropna() if str(value).strip()}))

    clean = selected.groupby("canonical_smiles", as_index=False, sort=True).agg(
        target_chembl_id=("target_chembl_id", "first"),
        pIC50=("pIC50", "median"),
        activity_nM=("activity_nM", "median"),
        measurement_count=("pIC50", "size"),
        molecule_ids=("molecule_chembl_id", join_ids),
        assay_ids=("assay_chembl_id", join_ids),
        document_ids=("document_chembl_id", join_ids),
    )
    clean["standard_type"] = endpoint.upper()
    clean["standard_units"] = units
    clean["aggregation"] = "median pIC50 per isomeric canonical SMILES"
    clean_path = out_dir / f"{endpoint.upper()}_clean.csv"
    clean.to_csv(clean_path, index=False)
    report = {
        "target_chembl_id": target_id,
        "endpoint": endpoint.upper(),
        "units": units,
        "raw_file": raw_path.name,
        "raw_sha256": sha256_file(raw_path),
        "curated_file": clean_path.name,
        "curated_sha256": sha256_file(clean_path),
        "raw_rows": int(len(raw)),
        "eligible_rows": int(eligible.sum()),
        "invalid_smiles": invalid_smiles,
        "curated_compounds": int(len(clean)),
        "measurement_rows": int(len(selected)),
        "repeated_measurements_collapsed": int(len(selected) - len(clean)),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Only exact positive values in the requested endpoint/unit are retained.",
            "Assay-condition harmonization beyond endpoint, relation, units and structure is not performed.",
            "Activity records from multiple assays are summarized by median per canonical structure.",
        ],
    }
    (out_dir / f"{endpoint.upper()}_curation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_common_dataset(target_ids: list[str], endpoint: str, units: str, output: Path) -> None:
    registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    metadata = {row["target_id"]: row for row in registry["targets"]}
    frames = []
    for target_id in target_ids:
        target_id = target_id.upper()
        target_dir = DATA_ROOT / target_id
        candidates = [target_dir / f"{endpoint.upper()}_clean.csv"]
        if target_id == "CHEMBL203":
            candidates.append(ROOT / "CHEMBL203_IC50_clean.csv")
        if target_id == "CHEMBL5145":
            candidates.append(target_dir / "BRAF_IC50_clean.csv")
        clean_path = next((path for path in candidates if path.exists()), None)
        if clean_path is None:
            raise FileNotFoundError(f"No curated data found for {target_id}; looked for {candidates}.")
        frame = pd.read_csv(clean_path)
        value_column = "activity_nM" if "activity_nM" in frame else "ic50_nM"
        required = {"canonical_smiles", "pIC50", value_column}
        if required - set(frame.columns):
            raise ValueError(f"{clean_path} is missing required columns: {sorted(required - set(frame.columns))}")
        target_meta = metadata.get(target_id, {})
        row = pd.DataFrame(
            {
                "target_chembl_id": target_id,
                "target_gene": target_meta.get("gene", target_id),
                "target_name": target_meta.get("name", target_id),
                "organism": target_meta.get("organism", ""),
                "standard_type": endpoint.upper(),
                "standard_units": units,
                "activity_nM": pd.to_numeric(frame[value_column], errors="coerce"),
                "pIC50": pd.to_numeric(frame["pIC50"], errors="coerce"),
                "canonical_smiles": frame["canonical_smiles"].astype(str),
                "measurement_count": frame.get("measurement_count", 1),
                "source_database": "ChEMBL",
                "molecule_ids": frame.get("molecule_chembl_ids", frame.get("molecule_ids", "")),
                "assay_ids": frame.get("assay_chembl_ids", frame.get("assay_ids", "")),
                "document_ids": frame.get("document_chembl_ids", frame.get("document_ids", "")),
                "aggregation": frame.get("aggregation", "median per canonical structure"),
            }
        )
        row = row[COMMON_COLUMNS].dropna(subset=["activity_nM", "pIC50", "canonical_smiles"])
        frames.append(row)
    common = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["target_chembl_id", "canonical_smiles"], keep="first"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    common.to_csv(output, index=False)
    print(f"Wrote {len(common)} target-compound rows across {common['target_chembl_id'].nunique()} targets to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="download target metadata, protein FASTA and exact ChEMBL activities")
    collect.add_argument("target_id")
    collect.add_argument("--endpoint", default="IC50")
    collect.add_argument("--units", default="nM")
    collect.add_argument("--overwrite", action="store_true")
    curate = subparsers.add_parser("curate", help="standardize a collected target dataset")
    curate.add_argument("target_id")
    curate.add_argument("--endpoint", default="IC50")
    curate.add_argument("--units", default="nM")
    common = subparsers.add_parser("common", help="combine curated target files into the common schema")
    common.add_argument("--targets", nargs="+", required=True)
    common.add_argument("--endpoint", default="IC50")
    common.add_argument("--units", default="nM")
    common.add_argument("--output", type=Path, default=ROOT / "data" / "common" / "multi_target_ic50.csv")
    args = parser.parse_args()
    if args.command == "collect":
        download_target(args.target_id, args.endpoint, args.units, args.overwrite)
    elif args.command == "curate":
        curate_target(args.target_id, args.endpoint, args.units)
    else:
        build_common_dataset(args.targets, args.endpoint, args.units, args.output)


if __name__ == "__main__":
    main()
