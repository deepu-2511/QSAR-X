
import time
import requests
import pandas as pd
from pathlib import Path

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT = Path(__file__).resolve().parent / "verified_egfr_ic50.csv"
PAGE_SIZE = 500
session = requests.Session()

params_base = {
    "target_chembl_id": "CHEMBL203",
    "standard_type": "IC50",
    "standard_relation": "=",
    "standard_units": "nM",
    "limit": PAGE_SIZE,
    "format": "json",
    "only": ",".join([
        "activity_id",
        "assay_chembl_id",
        "document_chembl_id",
        "molecule_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "target_chembl_id",
    ]),
}

rows = []
offset = 0
total = None

print("[QSAR-X] Checking ChEMBL API...")

try:
    while True:
        params = {**params_base, "offset": offset}
        response = session.get(
            f"{BASE}/activity.json",
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        page = payload.get("activities", [])
        meta = payload.get("page_meta", {})
        total = meta.get("total_count", total)

        if not page:
            break

        rows.extend(page)
        offset += len(page)

        print(
            f"[QSAR-X] Downloaded {len(rows)}"
            + (f" / {total}" if total is not None else "")
        )

        if total is not None and offset >= total:
            break

        time.sleep(0.2)

except requests.RequestException as exc:
    print(f"[QSAR-X ERROR] ChEMBL request failed: {exc}")
    raise SystemExit(1)

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit(
        "[QSAR-X ERROR] No matching activities returned."
    )

df["standard_value"] = pd.to_numeric(
    df["standard_value"], errors="coerce"
)
df = df.dropna(
    subset=[
        "activity_id",
        "assay_chembl_id",
        "molecule_chembl_id",
        "standard_value",
    ]
)
df = df[df["standard_value"] > 0].copy()
df = df.drop_duplicates(subset=["activity_id"])

# Save raw activity records first; structures are retrieved separately.
df["source"] = "ChEMBL"
df.to_csv(OUT, index=False)

print(f"[QSAR-X] Saved activity records: {len(df)}")
print(
    "[QSAR-X] Unique molecules:",
    df["molecule_chembl_id"].nunique()
)
print(
    "[QSAR-X] Unique assays:",
    df["assay_chembl_id"].nunique()
)
print(f"[QSAR-X] Output: {OUT}")
print(
    "[QSAR-X] NOTE: This file does not yet contain canonical SMILES. "
    "Retrieve structures and check overlap before external validation."
)
