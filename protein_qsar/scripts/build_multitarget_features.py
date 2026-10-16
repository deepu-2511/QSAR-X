"""Create reproducible chemical plus protein-sequence feature matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
COMMON_DATA = ROOT / "data" / "common" / "multi_target_ic50.csv"
OUTPUT_FILE = ROOT / "data" / "common" / "multi_target_features.npz"
FEATURE_REPORT = ROOT / "data" / "common" / "feature_manifest.json"
FP_BITS = 2048
FP_RADIUS = 2
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
HYDROPHOBIC = set("AILMFWVY")


def read_fasta(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    sequence = "".join(line.strip().upper() for line in lines if line.strip() and not line.startswith(">"))
    sequence = "".join(letter for letter in sequence if letter in AMINO_ACIDS)
    if len(sequence) < 50:
        raise ValueError(f"No full protein sequence found in {path}.")
    return sequence


def protein_features(sequence: str) -> tuple[list[float], list[str]]:
    length = len(sequence)
    aa = [sequence.count(residue) / length for residue in AMINO_ACIDS]
    pairs = [sequence[index:index + 2] for index in range(length - 1)]
    pair_denominator = max(len(pairs), 1)
    pair_counts = Counter(pairs)
    dipeptide = [pair_counts.get(pair, 0) / pair_denominator for pair in DIPEPTIDES]
    hydrophobic_fraction = sum(residue in HYDROPHOBIC for residue in sequence) / length
    values = aa + dipeptide + [length / 1000.0, hydrophobic_fraction]
    names = (
        [f"protein_AAC_{residue}" for residue in AMINO_ACIDS]
        + [f"protein_DPC_{pair}" for pair in DIPEPTIDES]
        + ["protein_length_per_1000aa", "protein_hydrophobic_fraction"]
    )
    return values, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=COMMON_DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    source = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    frame = pd.read_csv(source)
    required = {"target_chembl_id", "canonical_smiles", "pIC50"}
    if required - set(frame.columns):
        raise ValueError(f"Common dataset missing columns: {sorted(required - set(frame.columns))}")

    sequence_by_target = {}
    protein_vector_by_target = {}
    protein_feature_names = None
    for target_id in frame["target_chembl_id"].dropna().astype(str).unique():
        fasta_path = ROOT / "data" / "targets" / target_id / "protein.fasta"
        if not fasta_path.exists():
            raise FileNotFoundError(f"Missing target sequence file: {fasta_path}. Run target_pipeline.py collect {target_id}.")
        sequence = read_fasta(fasta_path)
        vector, names = protein_features(sequence)
        sequence_by_target[target_id] = sequence
        protein_vector_by_target[target_id] = np.asarray(vector, dtype=np.float32)
        protein_feature_names = names

    chemical_feature_names = (
        [f"morgan_r{FP_RADIUS}_{index:04d}" for index in range(FP_BITS)]
        + ["chem_MW", "chem_MolLogP", "chem_TPSA", "chem_HBD", "chem_HBA", "chem_rotatable_bonds", "chem_formal_charge"]
    )
    X_rows = []
    scaffolds = []
    kept = []
    for row_index, row in frame.iterrows():
        smiles = str(row["canonical_smiles"])
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        chemical = np.zeros((FP_BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(GENERATOR.GetFingerprint(molecule), chemical)
        descriptors = np.asarray(
            [
                Descriptors.MolWt(molecule),
                Descriptors.MolLogP(molecule),
                Descriptors.TPSA(molecule),
                Descriptors.NumHDonors(molecule),
                Descriptors.NumHAcceptors(molecule),
                Descriptors.NumRotatableBonds(molecule),
                Chem.GetFormalCharge(molecule),
            ],
            dtype=np.float32,
        )
        target_id = str(row["target_chembl_id"])
        combined = np.concatenate(
            [chemical.astype(np.float32), descriptors, protein_vector_by_target[target_id]]
        )
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
        scaffolds.append(scaffold if scaffold else f"ACYCLIC:{smiles}")
        X_rows.append(combined)
        kept.append(row_index)

    if not X_rows:
        raise ValueError("No valid structures remain after fingerprint generation.")
    kept_frame = frame.loc[kept].reset_index(drop=True)
    X = np.stack(X_rows).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=X,
        y=kept_frame["pIC50"].to_numpy(dtype=np.float32),
        target_ids=kept_frame["target_chembl_id"].astype(str).to_numpy(dtype=str),
        canonical_smiles=kept_frame["canonical_smiles"].astype(str).to_numpy(dtype=str),
        scaffold_groups=np.asarray(scaffolds, dtype=str),
    )
    manifest = {
        "source_dataset": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_file": str(output.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "n_rows": int(len(X)),
        "target_ids": sorted(kept_frame["target_chembl_id"].astype(str).unique().tolist()),
        "chemical_features": {
            "morgan_radius": FP_RADIUS,
            "morgan_bits": FP_BITS,
            "additional_descriptors": chemical_feature_names[FP_BITS:],
            "feature_count": len(chemical_feature_names),
        },
        "protein_features": {
            "encoding": "amino-acid composition + dipeptide composition + length + hydrophobic fraction",
            "feature_count": len(protein_feature_names or []),
            "sequence_files": {
                target: {
                    "file": str((ROOT / "data" / "targets" / target / "protein.fasta").relative_to(ROOT)),
                    "length_aa": len(sequence),
                    "sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                }
                for target, sequence in sequence_by_target.items()
            },
        },
        "total_feature_count": int(X.shape[1]),
        "feature_names": chemical_feature_names + (protein_feature_names or []),
        "limitation": "Protein sequence descriptors are basic and this dataset has too few targets to claim unseen-target generalization.",
    }
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ["n_rows", "target_ids", "total_feature_count", "output_file", "limitation"]}, indent=2))


if __name__ == "__main__":
    main()
