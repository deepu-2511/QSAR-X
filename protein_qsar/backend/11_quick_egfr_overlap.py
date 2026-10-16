
from pathlib import Path
import time
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from chembl_webresource_client.new_client import new_client


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent

ACTIVITY_FILE = BACKEND / "verified_egfr_ic50.csv"
EXISTING_FILE = ROOT / "CHEMBL203_IC50_clean.csv"

PROGRESS_FILE = BACKEND / "egfr_next3000_progress.csv"
OUTPUT_FILE = BACKEND / "egfr_next3000_overlap.csv"
REPORT_FILE = BACKEND / "egfr_next3000_overlap_report.txt"

START_INDEX = 3000       # zero-based: start from molecule 3001
COUNT = 3000             # molecules 3001 through 6000
BATCH_SIZE = 50
RETRIES = 3


# ============================================================
# CHEMICAL STRUCTURE HELPERS
# ============================================================

def canonicalize_smiles(smiles):
    if pd.isna(smiles) or not str(smiles).strip():
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_murcko_scaffold(smiles):
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        # Acyclic compounds can have an empty scaffold.
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "ACYCLIC"

        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def save_progress(records):
    if records:
        pd.DataFrame(records).drop_duplicates(
            subset=["molecule_chembl_id"],
            keep="last"
        ).to_csv(PROGRESS_FILE, index=False)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    for file_path in [ACTIVITY_FILE, EXISTING_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("=" * 65)
    print("EGFR NEXT 3,000 MOLECULE OVERLAP AUDIT")
    print("=" * 65)

    # Load activity data and obtain unique molecule IDs.
    activities = pd.read_csv(ACTIVITY_FILE, low_memory=False)

    if "molecule_chembl_id" not in activities.columns:
        raise ValueError(
            "verified_egfr_ic50.csv must contain molecule_chembl_id"
        )

    all_ids = (
        activities["molecule_chembl_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    target_ids = all_ids[START_INDEX:START_INDEX + COUNT]

    if not target_ids:
        raise ValueError(
            f"No molecule IDs available for indices "
            f"{START_INDEX} to {START_INDEX + COUNT - 1}. "
            f"Only {len(all_ids)} unique IDs were found."
        )

    print(f"Total unique molecule IDs: {len(all_ids)}")
    print(f"Selected ID range: {START_INDEX + 1}–"
          f"{START_INDEX + len(target_ids)}")
    print(f"Unique IDs selected: {len(target_ids)}")

    # Load existing QSAR dataset and canonicalize structures.
    existing = pd.read_csv(EXISTING_FILE, low_memory=False)

    if "canonical_smiles" in existing.columns:
        smiles_col = "canonical_smiles"
    elif "smiles" in existing.columns:
        smiles_col = "smiles"
    else:
        raise ValueError(
            "Existing dataset must contain canonical_smiles or smiles"
        )

    existing["canonical"] = existing[smiles_col].apply(
        canonicalize_smiles
    )
    existing = existing.dropna(subset=["canonical"])

    existing_smiles = set(existing["canonical"])

    existing["scaffold"] = existing["canonical"].apply(
        get_murcko_scaffold
    )
    existing_scaffolds = set(
        existing["scaffold"].dropna()
    )

    print(f"Existing unique canonical compounds: "
          f"{len(existing_smiles)}")
    print(f"Existing unique scaffolds: {len(existing_scaffolds)}")

    # Resume only from valid progress records in this target range.
    records_by_id = {}

    if PROGRESS_FILE.exists():
        old_progress = pd.read_csv(
            PROGRESS_FILE, low_memory=False
        )

        required = {"molecule_chembl_id", "canonical_smiles"}
        if required.issubset(old_progress.columns):
            allowed_ids = set(target_ids)

            for _, row in old_progress.iterrows():
                mol_id = str(row["molecule_chembl_id"])
                smiles = row["canonical_smiles"]

                if mol_id not in allowed_ids or pd.isna(smiles):
                    continue

                canonical = canonicalize_smiles(smiles)
                if canonical:
                    records_by_id[mol_id] = {
                        "molecule_chembl_id": mol_id,
                        "canonical_smiles": canonical
                    }

        print(f"Valid structures already saved: "
              f"{len(records_by_id)}")

    molecule = new_client.molecule
    remaining_ids = [
        mol_id for mol_id in target_ids
        if mol_id not in records_by_id
    ]

    total_batches = (
        len(remaining_ids) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(f"IDs still to retrieve: {len(remaining_ids)}")
    print("Starting structure retrieval...\n")

    # Retrieve structures in batches with retry handling.
    for batch_number, start in enumerate(
        range(0, len(remaining_ids), BATCH_SIZE), start=1
    ):
        batch = remaining_ids[start:start + BATCH_SIZE]
        fetched = {}
        last_error = None

        for attempt in range(1, RETRIES + 1):
            try:
                result = molecule.filter(
                    molecule_chembl_id__in=batch
                ).only(
                    ["molecule_chembl_id", "molecule_structures"]
                )

                for item in result:
                    mol_id = item.get("molecule_chembl_id")
                    structures = item.get("molecule_structures") or {}
                    smiles = structures.get("canonical_smiles")

                    canonical = canonicalize_smiles(smiles)

                    if mol_id and canonical:
                        fetched[str(mol_id)] = {
                            "molecule_chembl_id": str(mol_id),
                            "canonical_smiles": canonical
                        }

                break

            except Exception as exc:
                last_error = exc
                print(
                    f"Batch {batch_number}/{total_batches}, "
                    f"attempt {attempt} failed: {exc}"
                )
                if attempt < RETRIES:
                    time.sleep(2 * attempt)

        records_by_id.update(fetched)

        # Save after every batch so interrupted runs can resume.
        save_progress(list(records_by_id.values()))

        print(
            f"Batch {batch_number}/{total_batches}: "
            f"retrieved {len(fetched)}/{len(batch)}; "
            f"total saved {len(records_by_id)}/{len(target_ids)}"
        )

        if last_error and not fetched:
            print("This batch may need retrying on a later run.")

        time.sleep(0.3)

    # Retry missing IDs once more individually, if necessary.
    missing_ids = [
        mol_id for mol_id in target_ids
        if mol_id not in records_by_id
    ]

    if missing_ids:
        print(f"\nRetrying {len(missing_ids)} missing structures...")

        for index, mol_id in enumerate(missing_ids, start=1):
            for attempt in range(1, RETRIES + 1):
                try:
                    item = molecule.get(mol_id)
                    structures = item.get("molecule_structures") or {}
                    canonical = canonicalize_smiles(
                        structures.get("canonical_smiles")
                    )

                    if canonical:
                        records_by_id[mol_id] = {
                            "molecule_chembl_id": mol_id,
                            "canonical_smiles": canonical
                        }
                        break

                except Exception as exc:
                    if attempt == RETRIES:
                        print(f"Could not retrieve {mol_id}: {exc}")
                    else:
                        time.sleep(attempt)

            if index % 25 == 0:
                save_progress(list(records_by_id.values()))
                print(f"Missing-ID retry progress: "
                      f"{index}/{len(missing_ids)}")

        save_progress(list(records_by_id.values()))

    # Build a structure-level table: one row per unique molecule ID.
    structures_df = pd.DataFrame(
        list(records_by_id.values())
    )

    if structures_df.empty:
        raise RuntimeError(
            "No structures were retrieved. Check internet/API access."
        )

    structures_df["canonical_smiles"] = (
        structures_df["canonical_smiles"].apply(canonicalize_smiles)
    )
    structures_df = structures_df.dropna(
        subset=["canonical_smiles"]
    ).drop_duplicates(
        subset=["molecule_chembl_id"],
        keep="last"
    )

    structures_df["murcko_scaffold"] = (
        structures_df["canonical_smiles"].apply(get_murcko_scaffold)
    )

    structures_df["compound_overlap"] = (
        structures_df["canonical_smiles"].isin(existing_smiles)
    )

    structures_df["scaffold_overlap"] = (
        structures_df["murcko_scaffold"].isin(existing_scaffolds)
    )

    # Join activity rows to the unique molecule-level audit.
    selected_activities = activities[
        activities["molecule_chembl_id"]
        .astype(str).isin(set(target_ids))
    ].copy()

    selected_activities["molecule_chembl_id"] = (
        selected_activities["molecule_chembl_id"].astype(str)
    )

    output = selected_activities.merge(
        structures_df,
        on="molecule_chembl_id",
        how="left"
    )

    output.to_csv(OUTPUT_FILE, index=False)

    # Calculate unique-molecule counts, not activity-row counts.
    total_selected = len(target_ids)
    retrieved_ids = set(structures_df["molecule_chembl_id"])
    retrieved_count = len(retrieved_ids)

    compound_overlap_count = int(
        structures_df["compound_overlap"].sum()
    )
    scaffold_overlap_count = int(
        structures_df["scaffold_overlap"].sum()
    )

    unique_structures = structures_df[
        "canonical_smiles"
    ].nunique()

    report = [
        "EGFR NEXT 3,000 MOLECULE OVERLAP AUDIT",
        "=" * 50,
        f"Source activity file: {ACTIVITY_FILE.name}",
        f"Existing QSAR dataset: {EXISTING_FILE.name}",
        f"Unique IDs in source: {len(all_ids)}",
        f"Selected ID range: {START_INDEX + 1}–"
        f"{START_INDEX + len(target_ids)}",
        f"Unique IDs selected: {total_selected}",
        f"Unique molecule IDs with structures: {retrieved_count}",
        f"Unique canonical structures retrieved: {unique_structures}",
        f"Unique compounds overlapping existing dataset: "
        f"{compound_overlap_count}",
        f"Unique compounds not overlapping existing dataset: "
        f"{retrieved_count - compound_overlap_count}",
        f"Unique compounds with an existing Murcko scaffold: "
        f"{scaffold_overlap_count}",
        f"Unique compounds with unseen Murcko scaffolds: "
        f"{retrieved_count - scaffold_overlap_count}",
        f"Activity rows in selected subset: {len(selected_activities)}",
        f"Output CSV: {OUTPUT_FILE.name}",
        f"Progress CSV: {PROGRESS_FILE.name}",
        "",
        "IMPORTANT:",
        "This is a compound/scaffold overlap audit, not independent",
        "external validation. Molecules can have multiple activity rows.",
        "Shared scaffolds do not necessarily mean identical compounds.",
        "The production QSAR model was not changed by this script."
    ]

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)
    print(f"Unique IDs selected: {total_selected}")
    print(f"Structures retrieved: {retrieved_count}")
    print(f"Exact compound overlap: {compound_overlap_count}")
    print(f"Non-overlapping compounds: "
          f"{retrieved_count - compound_overlap_count}")
    print(f"Existing scaffold overlap: {scaffold_overlap_count}")
    print(f"Unseen scaffolds: {retrieved_count - scaffold_overlap_count}")
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Saved: {REPORT_FILE}")
    print("\nProduction model remains unchanged.")


if __name__ == "__main__":
    main()
