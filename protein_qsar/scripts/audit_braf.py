import pandas as pd

path = r"data\targets\CHEMBL5145\BRAF_IC50_raw.csv"
df = pd.read_csv(path)

value = pd.to_numeric(df["standard_value"], errors="coerce")
relation = df["standard_relation"].fillna("MISSING").astype(str).str.strip()
units = df["standard_units"].fillna("MISSING").astype(str).str.strip()
smiles = df["canonical_smiles"].fillna("").astype(str).str.strip()

print("=== BRAF RAW DATA AUDIT ===")
print("Total rows:", len(df))
print("Unique molecule IDs:", df["molecule_chembl_id"].nunique())
print("Missing SMILES:", (smiles == "").sum())
print("Missing/non-numeric IC50:", value.isna().sum())
print("Non-positive IC50:", (value.notna() & (value <= 0)).sum())
print("Exact '=' relation:", (relation == "=").sum())
print("Censored '<' relation:", (relation == "<").sum())
print("Censored '>' relation:", (relation == ">").sum())
print("Other/missing relations:", (~relation.isin(["=", "<", ">"])).sum())
print("Units breakdown:")
print(units.value_counts().to_string())
print("Unique assay IDs:", df["assay_chembl_id"].nunique())
print("Duplicate activity IDs:", df["activity_id"].duplicated().sum())
print("Duplicate molecule + assay + value rows:",
      df.duplicated(
          subset=["molecule_chembl_id", "assay_chembl_id",
                  "standard_value", "standard_relation"]
      ).sum())
