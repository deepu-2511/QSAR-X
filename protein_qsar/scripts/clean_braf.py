"""Curate exact BRAF IC50 measurements into a molecule-level QSAR dataset.

The source file is retained unchanged. This script deliberately keeps only
exact, positive IC50 values reported in nM and valid canonicalizable SMILES.
Repeated measurements for a canonical structure are summarized by median pIC50;
the measurement spread and source identifiers remain in the output for audit.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "CHEMBL5145"
TARGET_DIR = ROOT / "data" / "targets" / TARGET_ID
INPUT_FILE = TARGET_DIR / "BRAF_IC50_raw.csv"
OUTPUT_FILE = TARGET_DIR / "BRAF_IC50_clean.csv"
REPORT_JSON = TARGET_DIR / "BRAF_IC50_curation_report.json"
REPORT_TXT = TARGET_DIR / "BRAF_IC50_curation_report.txt"


def canonicalize(smiles: object) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def joined_unique(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values.dropna() if str(value).strip()}))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"BRAF raw dataset not found: {INPUT_FILE}")

    raw = pd.read_csv(INPUT_FILE, low_memory=False)
    required = {
        "activity_id", "assay_chembl_id", "assay_type", "molecule_chembl_id",
        "target_chembl_id", "standard_type", "standard_relation",
        "standard_value", "standard_units", "canonical_smiles",
        "document_chembl_id",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")

    df = raw.copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    units = df["standard_units"].fillna("").astype(str).str.strip().str.lower()
    relation = df["standard_relation"].fillna("").astype(str).str.strip()
    endpoint = df["standard_type"].fillna("").astype(str).str.strip().str.upper()
    target = df["target_chembl_id"].fillna("").astype(str).str.strip()

    masks = {
        "wrong_target": target.ne(TARGET_ID),
        "wrong_endpoint": endpoint.ne("IC50"),
        "non_exact_relation": relation.ne("="),
        "wrong_or_missing_units": units.ne("nm"),
        "invalid_or_nonpositive_value": df["standard_value"].isna() | df["standard_value"].le(0),
    }
    eligible = target.eq(TARGET_ID) & endpoint.eq("IC50") & relation.eq("=") & units.eq("nm")
    eligible &= df["standard_value"].notna() & df["standard_value"].gt(0)
    selected = df.loc[eligible].copy()
    selected["canonical_smiles"] = selected["canonical_smiles"].map(canonicalize)
    invalid_smiles = int(selected["canonical_smiles"].isna().sum())
    selected = selected.loc[selected["canonical_smiles"].notna()].copy()

    selected["ic50_nM"] = selected["standard_value"].astype(float)
    selected["pIC50"] = 9.0 - np.log10(selected["ic50_nM"])
    selected["target_chembl_id"] = TARGET_ID

    grouped = selected.groupby("canonical_smiles", sort=True, as_index=False).agg(
        target_chembl_id=("target_chembl_id", "first"),
        pIC50=("pIC50", "median"),
        ic50_nM=("ic50_nM", "median"),
        measurement_count=("pIC50", "size"),
        pIC50_min=("pIC50", "min"),
        pIC50_max=("pIC50", "max"),
        pIC50_std=("pIC50", "std"),
        molecule_chembl_ids=("molecule_chembl_id", joined_unique),
        assay_chembl_ids=("assay_chembl_id", joined_unique),
        document_chembl_ids=("document_chembl_id", joined_unique),
        assay_types=("assay_type", joined_unique),
    )
    grouped["standard_type"] = "IC50"
    grouped["standard_units"] = "nM"
    grouped["aggregation"] = "median pIC50 over exact IC50 measurements per canonical SMILES"
    grouped = grouped[
        [
            "target_chembl_id", "canonical_smiles", "pIC50", "ic50_nM",
            "measurement_count", "pIC50_min", "pIC50_max", "pIC50_std",
            "standard_type", "standard_units", "aggregation", "molecule_chembl_ids",
            "assay_chembl_ids", "document_chembl_ids", "assay_types",
        ]
    ]

    measurement_spread = grouped["pIC50_max"] - grouped["pIC50_min"]
    assay_type_counts = {
        str(key): int(value)
        for key, value in selected["assay_type"].fillna("MISSING").astype(str).value_counts().items()
    }

    report = {
        "target_chembl_id": TARGET_ID,
        "endpoint": "IC50",
        "source_file": str(INPUT_FILE.relative_to(ROOT)),
        "source_file_sha256": sha256_file(INPUT_FILE),
        "clean_file": str(OUTPUT_FILE.relative_to(ROOT)),
        "activity_transform": "pIC50 = 9 - log10(IC50_nM)",
        "aggregation_policy": "median pIC50 per isomeric canonical SMILES; retain measurement spread and source IDs",
        "raw_rows": int(len(raw)),
        "raw_unique_molecule_ids": int(raw["molecule_chembl_id"].nunique(dropna=True)),
        "filter_exclusions_nonexclusive": {key: int(mask.sum()) for key, mask in masks.items()},
        "eligible_exact_positive_nM_rows": int(eligible.sum()),
        "invalid_smiles_after_endpoint_filter": invalid_smiles,
        "retained_measurements": int(len(selected)),
        "unique_canonical_compounds": int(len(grouped)),
        "repeated_measurements_collapsed": int(len(selected) - len(grouped)),
        "unique_assays": int(selected["assay_chembl_id"].nunique(dropna=True)),
        "unique_documents": int(selected["document_chembl_id"].nunique(dropna=True)),
        "assay_type_measurement_counts": assay_type_counts,
        "compounds_with_repeated_measurements": int(grouped["measurement_count"].gt(1).sum()),
        "repeated_measurement_spread": {
            "compounds_with_pIC50_range_ge_1": int(measurement_spread.ge(1).sum()),
            "compounds_with_pIC50_range_ge_2": int(measurement_spread.ge(2).sum()),
            "maximum_pIC50_range": float(measurement_spread.max()),
        },
        "pIC50_summary": {
            "min": float(grouped["pIC50"].min()),
            "median": float(grouped["pIC50"].median()),
            "mean": float(grouped["pIC50"].mean()),
            "max": float(grouped["pIC50"].max()),
        },
        "caveat": "Curation does not establish assay comparability or model validity; inspect assay/source heterogeneity before training.",
    }

    grouped.to_csv(OUTPUT_FILE, index=False)
    report["clean_file_sha256"] = sha256_file(OUTPUT_FILE)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_lines = [
        "BRAF (CHEMBL5145) IC50 CURATION REPORT",
        "=" * 48,
        f"Raw activity rows: {report['raw_rows']}",
        f"Exact positive IC50 nM measurements: {report['eligible_exact_positive_nM_rows']}",
        f"Invalid SMILES among eligible measurements: {invalid_smiles}",
        f"Retained measurements: {len(selected)}",
        f"Unique canonical compounds: {len(grouped)}",
        f"Repeated measurements collapsed: {len(selected) - len(grouped)}",
        f"Unique assays / documents: {report['unique_assays']} / {report['unique_documents']}",
        f"Repeated compounds with pIC50 range >= 1 / >= 2: {report['repeated_measurement_spread']['compounds_with_pIC50_range_ge_1']} / {report['repeated_measurement_spread']['compounds_with_pIC50_range_ge_2']}",
        f"pIC50 range: {grouped['pIC50'].min():.3f}–{grouped['pIC50'].max():.3f}",
        "",
        "Non-exclusive exclusion counts:",
        *[f"  {key}: {count}" for key, count in report["filter_exclusions_nonexclusive"].items()],
        "",
        report["caveat"],
        f"\nClean data: {OUTPUT_FILE.name}",
        f"Machine-readable report: {REPORT_JSON.name}",
    ]
    REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
