
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from chembl_webresource_client.new_client import new_client

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent

ACTIVITY_FILE = BACKEND / "verified_egfr_ic50.csv"
EXISTING_FILE = ROOT / "CHEMBL203_IC50_clean.csv"

STRUCTURE_FILE = BACKEND / "verified_egfr_with_structures.csv"
REPORT_FILE = BACKEND / "egfr_overlap_report.txt"

print("[QSAR-X] Loading downloaded activities...")
activities = pd.read_csv(ACTIVITY_FILE)

if not EXISTING_FILE.exists():
    raise SystemExit(f"Existing dataset not found: {EXISTING_FILE}")

existing = pd.read_csv(EXISTING_FILE)

def find_col(df, names):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None

activity_id_col = find_col(activities, ["molecule_chembl_id"])
existing_smiles_col = find_col(
    existing, ["canonical_smiles", "smiles"]
)
existing_pic50_col = find_col(existing, ["pIC50", "pic50"])

if not activity_id_col or not existing_smiles_col:
    raise SystemExit("Required molecule ID or SMILES column missing.")

if not existing_pic50_col:
    raise SystemExit("Existing dataset has no pIC50 column.")

ids = activities[activity_id_col].dropna().astype(str).unique().tolist()
molecule_api = new_client.molecule

print(f"[QSAR-X] Retrieving structures for {len(ids)} molecule IDs...")

structures = []
batch_size = 100

for start in range(0, len(ids), batch_size):
    batch = ids[start:start + batch_size]

    try:
        records = molecule_api.filter(
            molecule_chembl_id__in=",".join(batch)
        ).only([
            "molecule_chembl_id",
            "molecule_structures",
        ])

        for record in records:
            structure_data = record.get("molecule_structures") or {}
            smiles = structure_data.get("canonical_smiles")

            if smiles:
                structures.append({
                    "molecule_chembl_id": record["molecule_chembl_id"],
                    "canonical_smiles": smiles,
                })

    except Exception as exc:
        raise SystemExit(
            f"Structure retrieval failed at batch {start}: {exc}"
        )

    print(
        f"[QSAR-X] Processed {min(start + batch_size, len(ids))}"
        f" / {len(ids)} IDs; structures retrieved: {len(structures)}"
    )

structure_df = pd.DataFrame(structures).drop_duplicates(
    subset=["molecule_chembl_id"]
)

if structure_df.empty:
    raise SystemExit("No structures retrieved; stopping safely.")

data = activities.merge(
    structure_df,
    on="molecule_chembl_id",
    how="inner",
    validate="many_to_one",
)

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

data["canonical_smiles"] = data["canonical_smiles"].map(canonicalize)
existing_smiles = existing[existing_smiles_col].map(canonicalize)

data = data.dropna(subset=["canonical_smiles"]).copy()
existing_set = set(existing_smiles.dropna())

data["in_existing_dataset"] = data["canonical_smiles"].isin(existing_set)

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)

data["murcko_scaffold"] = data["canonical_smiles"].map(get_scaffold)

existing_scaffolds = set(
    existing_smiles.dropna().map(get_scaffold).dropna()
)
data["scaffold_seen_in_existing"] = data["murcko_scaffold"].isin(
    existing_scaffolds
)

data.to_csv(STRUCTURE_FILE, index=False)

report = [
    f"Downloaded activity rows: {len(activities)}",
    f"Activity molecule IDs: {len(ids)}",
    f"Molecules with retrieved structures: {len(structure_df)}",
    f"Rows with valid canonical SMILES: {len(data)}",
    f"Unique valid molecules: {data['canonical_smiles'].nunique()}",
    f"Rows overlapping existing compounds: {int(data['in_existing_dataset'].sum())}",
    f"Rows with previously seen Murcko scaffolds: {int(data['scaffold_seen_in_existing'].sum())}",
    f"Rows with unseen Murcko scaffolds: {int((~data['scaffold_seen_in_existing']).sum())}",
    "",
    "IMPORTANT: Unseen scaffold does not by itself establish external independence.",
    "Assay provenance, publication dates, assay quality and label consistency must be reviewed.",
]

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print(f"[QSAR-X] Structure dataset: {STRUCTURE_FILE}")
print(f"[QSAR-X] Report: {REPORT_FILE}")
print("[QSAR-X] No model was retrained or promoted.")
