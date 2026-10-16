import requests
import pandas as pd
from pathlib import Path

TARGET = "CHEMBL5145"
OUT = Path("data") / "targets" / TARGET
OUT.mkdir(parents=True, exist_ok=True)

url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
params = {
    "target_chembl_id": TARGET,
    "standard_type": "IC50",
    "standard_units": "nM",
    "limit": 1000,
    "offset": 0,
}

records = []
session = requests.Session()

while True:
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    batch = payload.get("activities", [])
    records.extend(batch)

    print(f"Downloaded {len(records)} / {payload['page_meta']['total_count']} records")

    if not payload["page_meta"].get("next"):
        break

    params["offset"] += len(batch)
    if not batch:
        break

columns = [
    "activity_id", "assay_chembl_id", "assay_type",
    "molecule_chembl_id", "target_chembl_id",
    "standard_type", "standard_relation", "standard_value",
    "standard_units", "pchembl_value", "canonical_smiles",
    "document_chembl_id"
]

df = pd.DataFrame(records)
for col in columns:
    if col not in df.columns:
        df[col] = None

df = df[columns]
output = OUT / "BRAF_IC50_raw.csv"
df.to_csv(output, index=False)

print("\nDownload complete")
print("Target:", TARGET)
print("Rows:", len(df))
print("Unique molecule IDs:", df["molecule_chembl_id"].nunique())
print("Rows with SMILES:", df["canonical_smiles"].notna().sum())
print("Output:", output.resolve())
