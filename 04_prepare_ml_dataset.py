from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = "CHEMBL203_IC50_clean.csv"

TRAIN_FILE = "CHEMBL203_train.csv"
VALIDATION_FILE = "CHEMBL203_validation.csv"
TEST_FILE = "CHEMBL203_test.csv"

RANDOM_SEED = 42


def get_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    return Chem.MolToSmiles(
        scaffold,
        canonical=True
    )


def scaffold_split(df, train_fraction=0.70,
                   validation_fraction=0.15):

    print()
    print("Generating Bemis-Murcko scaffolds...")

    df = df.copy()

    df["scaffold"] = df[
        "canonical_smiles"
    ].apply(get_scaffold)

    invalid = df["scaffold"].isna().sum()

    if invalid > 0:

        print(
            f"Warning: removing {invalid} "
            "molecules without scaffolds."
        )

        df = df[
            df["scaffold"].notna()
        ].copy()

    # Group molecules by scaffold
    scaffold_groups = (
        df.groupby("scaffold")
        .indices
    )

    groups = list(
        scaffold_groups.items()
    )

    # Largest scaffold groups first
    groups.sort(
        key=lambda x: len(x[1]),
        reverse=True
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # Shuffle groups with deterministic seed
    rng.shuffle(groups)

    # Re-sort by group size so large scaffolds
    # are distributed deliberately.
    groups.sort(
        key=lambda x: len(x[1]),
        reverse=True
    )

    total = len(df)

    target_train = int(
        total * train_fraction
    )

    target_validation = int(
        total * validation_fraction
    )

    train_indices = []
    validation_indices = []
    test_indices = []

    for scaffold, indices in groups:

        if len(train_indices) < target_train:

            train_indices.extend(
                indices
            )

        elif (
            len(validation_indices)
            < target_validation
        ):

            validation_indices.extend(
                indices
            )

        else:

            test_indices.extend(
                indices
            )

    train_df = df.loc[
        train_indices
    ].copy()

    validation_df = df.loc[
        validation_indices
    ].copy()

    test_df = df.loc[
        test_indices
    ].copy()

    # Remove scaffold column from saved datasets
    for dataset in [
        train_df,
        validation_df,
        test_df
    ]:

        dataset.drop(
            columns=["scaffold"],
            inplace=True
        )

    return (
        train_df,
        validation_df,
        test_df
    )


def main():

    print("=" * 70)
    print("QSAR MACHINE-LEARNING DATA PREPARATION")
    print("=" * 70)

    if not Path(INPUT_FILE).exists():

        print()
        print(
            f"ERROR: {INPUT_FILE} not found."
        )

        return

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Input molecules: {len(df)}"
    )

    # ------------------------------------------------------------
    # Scaffold split
    # ------------------------------------------------------------

    train_df, validation_df, test_df = (
        scaffold_split(df)
    )

    print()
    print("-" * 70)
    print("SCaffold split results")
    print("-" * 70)

    print(
        f"Training:   {len(train_df)}"
    )

    print(
        f"Validation: {len(validation_df)}"
    )

    print(
        f"Test:       {len(test_df)}"
    )

    # ------------------------------------------------------------
    # Save datasets
    # ------------------------------------------------------------

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    print()
    print("Datasets saved:")

    print(
        f"  {TRAIN_FILE}"
    )

    print(
        f"  {VALIDATION_FILE}"
    )

    print(
        f"  {TEST_FILE}"
    )

    # ------------------------------------------------------------
    # Morgan fingerprint test
    # ------------------------------------------------------------

    print()
    print(
        "Testing Morgan fingerprints..."
    )

    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048
        )
    )

    example_smiles = train_df[
        "canonical_smiles"
    ].iloc[0]

    mol = Chem.MolFromSmiles(
        example_smiles
    )

    fingerprint = generator.GetFingerprint(
        mol
    )

    print(
        f"Fingerprint size: "
        f"{fingerprint.GetNumBits()}"
    )

    print(
        f"Example molecule: "
        f"{example_smiles}"
    )

    print()
    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()