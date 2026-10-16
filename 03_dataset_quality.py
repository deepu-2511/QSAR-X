import pandas as pd
import numpy as np


INPUT_FILE = "CHEMBL203_IC50_clean.csv"


def main():

    print("=" * 70)
    print("EGFR QSAR DATASET QUALITY REPORT")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print()
    print(f"Dataset shape: {df.shape}")

    print()
    print("Columns:")
    print(df.columns.tolist())

    print()
    print("-" * 70)
    print("BASIC STATISTICS")
    print("-" * 70)

    print(f"Unique molecules: {df['canonical_smiles'].nunique()}")
    print(f"Minimum pIC50: {df['pIC50'].min():.3f}")
    print(f"Maximum pIC50: {df['pIC50'].max():.3f}")
    print(f"Mean pIC50: {df['pIC50'].mean():.3f}")
    print(f"Median pIC50: {df['pIC50'].median():.3f}")
    print(f"Std pIC50: {df['pIC50'].std():.3f}")

    print()
    print("-" * 70)
    print("PERCENTILES")
    print("-" * 70)

    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

    values = df["pIC50"].quantile(
        np.array(percentiles) / 100
    )

    for p, value in zip(percentiles, values):

        print(
            f"{p:>3}% : {value:.3f}"
        )

    print()
    print("-" * 70)
    print("MEASUREMENT COUNT")
    print("-" * 70)

    print(
        df["measurement_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    print(
        df.isna()
        .sum()
        .to_string()
    )

    print()
    print("-" * 70)
    print("EXTREME pIC50 VALUES")
    print("-" * 70)

    print()
    print("Lowest 10:")

    print(
        df[
            [
                "canonical_smiles",
                "pIC50",
                "ic50_nM",
                "measurement_count"
            ]
        ]
        .sort_values("pIC50")
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Highest 10:")

    print(
        df[
            [
                "canonical_smiles",
                "pIC50",
                "ic50_nM",
                "measurement_count"
            ]
        ]
        .sort_values(
            "pIC50",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("QUALITY REPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()