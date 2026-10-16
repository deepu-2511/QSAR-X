# ============================================================
# QSAR-X Professional
# Scientific Dataset Audit
# Version 1.0.0
# ============================================================

import json
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")

VERSION = "1.0.0"
INPUT_FILE = Path("curation_output/clean_dataset.csv")
OUTPUT_DIR = Path("dataset_audit_output")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return json_safe(obj.item())
        except Exception:
            pass
    if obj is None:
        return None
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def distribution(series):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return {"count": 0}

    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
        "q01": float(s.quantile(0.01)),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
    }


def value_counts(series):
    return {
        str(k): int(v)
        for k, v in series.fillna("<MISSING>")
        .astype(str).value_counts().items()
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run this script from the backend folder."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = ["canonical_smiles", "pIC50"]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    n = len(df)

    # --------------------------------------------------------
    # 1. Missing values and basic structure
    # --------------------------------------------------------
    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [
            round(float(df[c].isna().mean() * 100), 4)
            for c in df.columns
        ],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })

    missing_report.to_csv(
        OUTPUT_DIR / "missing_values_audit.csv", index=False
    )

    # --------------------------------------------------------
    # 2. Validate SMILES with RDKit
    # --------------------------------------------------------
    smiles_cache = {}

    def check_smiles(value):
        if pd.isna(value) or not str(value).strip():
            return False

        value = str(value).strip()

        if value not in smiles_cache:
            mol = Chem.MolFromSmiles(value)
            smiles_cache[value] = mol is not None

        return smiles_cache[value]

    df["audit_smiles_valid"] = (
        df["canonical_smiles"].apply(check_smiles)
    )

    invalid_smiles = df.loc[
        ~df["audit_smiles_valid"]
    ].copy()

    invalid_smiles.to_csv(
        OUTPUT_DIR / "invalid_structures.csv", index=False
    )

    # --------------------------------------------------------
    # 3. Activity distribution and suspicious values
    # --------------------------------------------------------
    df["pIC50_numeric"] = pd.to_numeric(
        df["pIC50"], errors="coerce"
    )

    invalid_activity = df[
        df["pIC50_numeric"].isna()
        | ~df["pIC50_numeric"].map(
            lambda x: math.isfinite(x) if pd.notna(x) else False
        )
    ].copy()

    invalid_activity.to_csv(
        OUTPUT_DIR / "invalid_activity_values.csv", index=False
    )

    # Broad screening flags, NOT automatic rejection rules.
    df["audit_flags"] = ""

    def add_flag(mask, label):
        current = df.loc[mask, "audit_flags"]
        df.loc[mask, "audit_flags"] = current.apply(
            lambda x: f"{x};{label}" if x else label
        )

    p = df["pIC50_numeric"]
    add_flag(p < 1, "VERY_LOW_PIC50_REVIEW")
    add_flag(p > 12, "VERY_HIGH_PIC50_REVIEW")

    if "molecular_weight" in df.columns:
        mw = pd.to_numeric(df["molecular_weight"], errors="coerce")
        add_flag(mw < 50, "LOW_MOLECULAR_WEIGHT_REVIEW")
        add_flag(mw > 1000, "HIGH_MOLECULAR_WEIGHT_REVIEW")

    df.loc[~df["audit_smiles_valid"], "audit_flags"] = (
        df.loc[~df["audit_smiles_valid"], "audit_flags"]
        .apply(lambda x: f"{x};INVALID_SMILES" if x else "INVALID_SMILES")
    )

    df[df["audit_flags"] != ""].to_csv(
        OUTPUT_DIR / "quality_flagged_records.csv", index=False
    )

    # --------------------------------------------------------
    # 4. Duplicate canonical structures
    # --------------------------------------------------------
    duplicate_mask = df["canonical_smiles"].duplicated(keep=False)

    duplicates = df.loc[duplicate_mask].sort_values(
        "canonical_smiles"
    )

    duplicates.to_csv(
        OUTPUT_DIR / "duplicate_structures.csv", index=False
    )

    # --------------------------------------------------------
    # 5. Repeated measurement consistency
    # --------------------------------------------------------
    repeat_report = pd.DataFrame()

    if "measurement_count" in df.columns:
        counts = pd.to_numeric(
            df["measurement_count"], errors="coerce"
        )
    else:
        counts = pd.Series(1, index=df.index, dtype=float)

    if "pIC50_std" in df.columns:
        within_std = pd.to_numeric(
            df["pIC50_std"], errors="coerce"
        )
    else:
        within_std = pd.Series(float("nan"), index=df.index)

    repeat_report = pd.DataFrame({
        "canonical_smiles": df["canonical_smiles"],
        "measurement_count": counts,
        "pIC50_std": within_std,
        "pIC50_min": (
            pd.to_numeric(df["pIC50_min"], errors="coerce")
            if "pIC50_min" in df.columns else p
        ),
        "pIC50_max": (
            pd.to_numeric(df["pIC50_max"], errors="coerce")
            if "pIC50_max" in df.columns else p
        ),
    })

    repeat_report["pIC50_range"] = (
        repeat_report["pIC50_max"]
        - repeat_report["pIC50_min"]
    )

    repeated = repeat_report[
        repeat_report["measurement_count"] > 1
    ].copy()

    repeated.to_csv(
        OUTPUT_DIR / "repeated_measurement_audit.csv", index=False
    )

    # Large within-compound differences require review.
    repeated[
        repeated["pIC50_range"] >= 2
    ].to_csv(
        OUTPUT_DIR / "high_variability_compounds.csv", index=False
    )

    # --------------------------------------------------------
    # 6. Data provenance / synthetic test rows
    # --------------------------------------------------------
    provenance = {}

    if "data_source" in df.columns:
        provenance["data_source_counts"] = value_counts(
            df["data_source"]
        )

        synthetic_mask = df["data_source"].astype(str).str.contains(
            "synthetic|test", case=False, na=False
        )

        df.loc[synthetic_mask].to_csv(
            OUTPUT_DIR / "synthetic_or_test_records.csv",
            index=False
        )

        provenance["synthetic_or_test_rows"] = int(
            synthetic_mask.sum()
        )
    else:
        provenance["data_source_counts"] = "No data_source column"

    if "test_case" in df.columns:
        provenance["test_case_counts"] = value_counts(
            df["test_case"]
        )

    # --------------------------------------------------------
    # 7. Endpoint, relation and unit consistency
    # --------------------------------------------------------
    endpoint_report = {}

    for column in ["endpoint", "relation", "units", "activity_mode"]:
        if column in df.columns:
            endpoint_report[column] = value_counts(df[column])

    # --------------------------------------------------------
    # 8. Measurement-count integrity
    # --------------------------------------------------------
    measurement_audit = {}

    if "measurement_count" in df.columns:
        mc = pd.to_numeric(df["measurement_count"], errors="coerce")

        measurement_audit = {
            "missing_counts": int(mc.isna().sum()),
            "counts_below_one": int((mc < 1).sum()),
            "total_measurements_represented": (
                int(mc.sum()) if mc.notna().all() else None
            ),
            "compounds_with_multiple_measurements": int((mc > 1).sum()),
            "maximum_measurement_count": (
                int(mc.max()) if mc.notna().any() else None
            ),
        }

    # --------------------------------------------------------
    # 9. Generate plots where matplotlib is available
    # --------------------------------------------------------
    plot_status = "Not generated"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid_p = p.dropna()

        if not valid_p.empty:
            plt.figure(figsize=(9, 5))
            plt.hist(valid_p, bins=40)
            plt.xlabel("pIC50")
            plt.ylabel("Number of compounds")
            plt.title("pIC50 Distribution")
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / "pic50_distribution.png", dpi=180
            )
            plt.close()

        if "molecular_weight" in df.columns:
            mw = pd.to_numeric(
                df["molecular_weight"], errors="coerce"
            ).dropna()

            if not mw.empty:
                plt.figure(figsize=(9, 5))
                plt.hist(mw, bins=40)
                plt.xlabel("Molecular weight (Da)")
                plt.ylabel("Number of compounds")
                plt.title("Molecular Weight Distribution")
                plt.tight_layout()
                plt.savefig(
                    OUTPUT_DIR / "molecular_weight_distribution.png",
                    dpi=180
                )
                plt.close()

        plot_status = "Generated if data were available"

    except ImportError:
        plot_status = "Matplotlib not installed; plots skipped"

    # --------------------------------------------------------
    # 10. Summary report
    # --------------------------------------------------------
    report = {
        "audit_version": VERSION,
        "input_file": str(INPUT_FILE),
        "total_compounds": int(n),
        "columns": list(df.columns),
        "duplicate_canonical_smiles_rows": int(duplicate_mask.sum()),
        "unique_canonical_smiles": int(
            df["canonical_smiles"].nunique(dropna=True)
        ),
        "invalid_smiles_rows": int((~df["audit_smiles_valid"]).sum()),
        "missing_pic50": int(df["pIC50_numeric"].isna().sum()),
        "pic50_statistics": distribution(p),
        "molecular_weight_statistics": (
            distribution(df["molecular_weight"])
            if "molecular_weight" in df.columns else None
        ),
        "quality_flagged_rows": int((df["audit_flags"] != "").sum()),
        "repeated_measurement_summary": {
            "compounds_with_repeats": int(len(repeated)),
            "max_measurement_count": (
                int(counts.max()) if counts.notna().any() else None
            ),
            "compounds_with_pic50_range_ge_2": int(
                (repeated["pIC50_range"] >= 2).sum()
            ),
        },
        "measurement_count_integrity": measurement_audit,
        "provenance": provenance,
        "endpoint_distribution": endpoint_report,
        "plot_status": plot_status,
        "important_note": (
            "This is a screening audit. Flags do not automatically "
            "mean a record is erroneous. Review provenance and raw "
            "assay evidence before excluding compounds."
        ),
    }

    with open(
        OUTPUT_DIR / "dataset_audit_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(json_safe(report), f, indent=2, ensure_ascii=False)

    with open(
        OUTPUT_DIR / "dataset_audit_report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("QSAR-X PROFESSIONAL — DATASET AUDIT\n")
        f.write("=" * 60 + "\n\n")

        for key, value in report.items():
            f.write(f"{key}:\n{json.dumps(json_safe(value), indent=2)}\n\n")

    # Save enriched audit copy, not a replacement for the clean dataset.
    df.to_csv(
        OUTPUT_DIR / "audit_enriched_dataset.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("QSAR-X DATASET AUDIT COMPLETED")
    print("=" * 60)
    print(f"Total compounds: {n}")
    print(f"Unique structures: {report['unique_canonical_smiles']}")
    print(f"Invalid structures: {report['invalid_smiles_rows']}")
    print(f"Missing pIC50: {report['missing_pic50']}")
    print(f"Duplicate-structure rows: {report['duplicate_canonical_smiles_rows']}")
    print(f"Quality-flagged rows: {report['quality_flagged_rows']}")
    print(f"Audit output: {OUTPUT_DIR.resolve()}")
    print("\nKey report: dataset_audit_report.txt")
    print("Also inspect: synthetic_or_test_records.csv")
    print("              high_variability_compounds.csv")
    print("              quality_flagged_records.csv")


if __name__ == "__main__":
    main()