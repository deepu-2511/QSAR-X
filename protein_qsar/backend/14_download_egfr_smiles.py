
import time
import requests
import pandas as pd
from pathlib import Path

INPUT = Path("verified_egfr_exact_ic50_audit.csv")
OUTPUT = Path("verified_egfr_smiles.csv")
PROGRESS = Path("verified_egfr_smiles_progress.csv")

df = pd.read_csv(INPUT, usecols=["molecule_chembl_id"])
ids = sorted(df["molecule_chembl_id"].dropna().unique())

session = requests.Session()
session.headers.update({"User-Agent": "QSAR-X-Professional/1.0"})
results = {}

if PROGRESS.exists():
    old = pd.read_csv(PROGRESS)
    for _, row in old.iterrows():
        results[str(row["molecule_chembl_id"])] = row.get("canonical_smiles", "")

pending = [x for x in ids if x not in results]
print("Unique molecule IDs:", len(ids))
print("Already saved:", len(results))
print("Remaining:", len(pending))

for i, mol_id in enumerate(pending, 1):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_id}.json"

    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        structures = data.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles") or ""
        results[mol_id] = smiles
    except Exception as exc:
        print(f"Retry later: {mol_id}: {exc}")
        continue

    if i % 50 == 0:
        pd.DataFrame([
            {"molecule_chembl_id": k, "canonical_smiles": v}
            for k, v in results.items()
        ]).to_csv(PROGRESS, index=False)
        print(f"Processed {i}/{len(pending)}; saved {len(results)}")

    time.sleep(0.15)

pd.DataFrame([
    {"molecule_chembl_id": k, "canonical_smiles": v}
    for k, v in results.items()
]).to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("IDs with structures:", sum(bool(v) for v in results.values()))
print("IDs without structures:", sum(not bool(v) for v in results.values()))
print("If interrupted, rerun to resume from progress file.")
