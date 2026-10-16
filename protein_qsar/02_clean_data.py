from rdkit import Chem
import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = "CHEMBL203_IC50_raw.csv"
OUTPUT_FILE = "CHEMBL203_IC50_clean.csv"
REPORT_FILE = "CHEMBL203_cleaning_report.txt"


def validate_smiles(smiles):
    if pd.isna(smiles):
        return False

    mol = Chem.MolFromSmiles(str(smiles))

    return mol is not None


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True
    )


def main():

    print("=" * 70)
    print("QSAR DATA CURATION")
    print("=" * 70)

    input_path = Path(INPUT_FILE)

    if not input_path.exists():

        print()
        print(f"ERROR: {INPUT_FILE} not found.")
        return

    print()
    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    initial_records = len(df)

    print(f"Initial activity records: {initial_records}")

    # ------------------------------------------------------------
    # 1. Keep only exact IC50 measurements
    # ------------------------------------------------------------

    before = len(df)

    df = df[
        (df["standard_type"] == "IC50") &
        (df["standard_relation"] == "=") &
        (df["standard_units"].str.lower() == "nm")
    ].copy()

    removed_endpoint = before - len(df)

    print()
    print("1. Endpoint filtering")
    print(f"   Removed: {removed_endpoint}")
    print(f"   Remaining: {len(df)}")

    # ------------------------------------------------------------
    # 2. Numeric activity values
    # ------------------------------------------------------------

    df["standard_value"] = pd.to_numeric(
        df["standard_value"],
        errors="coerce"
    )

    before = len(df)

    df = df[
        df["standard_value"].notna() &
        (df["standard_value"] > 0)
    ].copy()

    removed_activity = before - len(df)

    print()
    print("2. Activity-value filtering")
    print(f"   Removed: {removed_activity}")
    print(f"   Remaining: {len(df)}")

    # ------------------------------------------------------------
    # 3. Validate SMILES
    # ------------------------------------------------------------

    print()
    print("3. Validating SMILES...")

    df["valid_smiles"] = df["smiles"].apply(
        validate_smiles
    )

    invalid_smiles = (
        ~df["valid_smiles"]
    ).sum()

    df = df[
        df["valid_smiles"]
    ].copy()

    df.drop(
        columns=["valid_smiles"],
        inplace=True
    )

    print(f"   Invalid SMILES removed: {invalid_smiles}")
    print(f"   Remaining: {len(df)}")

    # ------------------------------------------------------------
    # 4. Canonical SMILES
    # ------------------------------------------------------------

    print()
    print("4. Canonicalizing SMILES...")

    df["canonical_smiles"] = df[
        "smiles"
    ].apply(canonicalize_smiles)

    df = df[
        df["canonical_smiles"].notna()
    ].copy()

    # ------------------------------------------------------------
    # 5. Calculate pIC50
    #
    # IC50 is currently in nM.
    #
    # pIC50 = -log10(IC50 in M)
    #
    # Example:
    # 10 nM = 1e-8 M
    # pIC50 = 8
    # ------------------------------------------------------------

    print()
    print("5. Calculating pIC50...")

    df["ic50_nM"] = df[
        "standard_value"
    ]

    df["ic50_M"] = (
        df["ic50_nM"] * 1e-9
    )

    df["pIC50"] = -np.log10(
        df["ic50_M"]
    )

    # ------------------------------------------------------------
    # 6. Remove chemically identical duplicate measurements
    #
    # We DO NOT simply discard repeated measurements.
    # Instead, they will be handled in the aggregation stage.
    # ------------------------------------------------------------

    print()
    print("6. Checking duplicate molecules...")

    unique_molecules = (
        df["canonical_smiles"]
        .nunique()
    )

    duplicate_measurements = (
        len(df) - unique_molecules
    )

    print(
        f"   Unique molecules: {unique_molecules}"
    )

    print(
        f"   Additional measurements: "
        f"{duplicate_measurements}"
    )

    # ------------------------------------------------------------
    # 7. Aggregate repeated measurements
    #
    # Median pIC50 is used when multiple measurements
    # exist for the same canonical molecule.
    # ------------------------------------------------------------

    print()
    print(
        "7. Aggregating repeated measurements..."
    )

    aggregation = (
        df.groupby(
            "canonical_smiles",
            as_index=False
        )
        .agg(
            pIC50=("pIC50", "median"),
            ic50_nM=("ic50_nM", "median"),
            measurement_count=(
                "pIC50",
                "count"
            )
        )
    )

    # ------------------------------------------------------------
    # 8. Add target information
    # ------------------------------------------------------------

    aggregation["target_chembl_id"] = (
        "CHEMBL203"
    )

    aggregation["standard_type"] = (
        "IC50"
    )

    aggregation["standard_units"] = (
        "nM"
    )

    # ------------------------------------------------------------
    # 9. Reorder columns
    # ------------------------------------------------------------

    final_df = aggregation[
        [
            "target_chembl_id",
            "canonical_smiles",
            "ic50_nM",
            "pIC50",
            "measurement_count",
            "standard_type",
            "standard_units",
        ]
    ].copy()

    # ------------------------------------------------------------
    # 10. Save cleaned dataset
    # ------------------------------------------------------------

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # 11. Generate cleaning report
    # ------------------------------------------------------------

    report = []

    report.append(
        "QSAR DATA CURATION REPORT"
    )

    report.append(
        "=" * 60
    )

    report.append(
        f"Input file: {INPUT_FILE}"
    )

    report.append(
        f"Initial records: {initial_records}"
    )

    report.append(
        f"Final unique molecules: {len(final_df)}"
    )

    report.append(
        ""
    )

    report.append(
        "Filtering summary:"
    )

    report.append(
        f"Endpoint filtering removed: "
        f"{removed_endpoint}"
    )

    report.append(
        f"Invalid/non-positive activity removed: "
        f"{removed_activity}"
    )

    report.append(
        f"Invalid SMILES removed: "
        f"{invalid_smiles}"
    )

    report.append(
        f"Unique canonical molecules: "
        f"{unique_molecules}"
    )

    report.append(
        f"Repeated measurements aggregated: "
        f"{duplicate_measurements}"
    )

    report.append(
        ""
    )

    report.append(
        "pIC50 statistics:"
    )

    report.append(
        f"Minimum: {final_df['pIC50'].min():.3f}"
    )

    report.append(
        f"Maximum: {final_df['pIC50'].max():.3f}"
    )

    report.append(
        f"Mean: {final_df['pIC50'].mean():.3f}"
    )

    report.append(
        f"Median: {final_df['pIC50'].median():.3f}"
    )

    report.append(
        f"Standard deviation: "
        f"{final_df['pIC50'].std():.3f}"
    )

    report.append(
        ""
    )

    report.append(
        f"Clean dataset: {OUTPUT_FILE}"
    )

    report.append(
        f"Cleaning report: {REPORT_FILE}"
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CURATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Initial records: {initial_records}"
    )

    print(
        f"Final unique molecules: {len(final_df)}"
    )

    print(
        f"pIC50 range: "
        f"{final_df['pIC50'].min():.3f} - "
        f"{final_df['pIC50'].max():.3f}"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Saved: {REPORT_FILE}"
    )

    print()


if __name__ == "__main__":
    main()