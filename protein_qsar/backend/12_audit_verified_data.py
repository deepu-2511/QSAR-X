import pandas as pd
import numpy as np

df = pd.read_csv("verified_egfr_ic50.csv")

print("=== EGFR EXPERIMENTAL DATA AUDIT ===")
print("Total rows:", len(df))
print("Unique molecule IDs:", df["molecule_chembl_id"].nunique())
print("Unique assays:", df["assay_chembl_id"].nunique())
print("Unique documents:", df["document_chembl_id"].nunique())
print("\nTarget IDs:")
print(df["target_chembl_id"].value_counts(dropna=False))
print("\nEndpoint types:")
print(df["standard_type"].value_counts(dropna=False))
print("\nRelations:")
print(df["standard_relation"].value_counts(dropna=False))
print("\nUnits:")
print(df["standard_units"].value_counts(dropna=False))
print("\nMissing SMILES column:", "smiles" not in df.columns)
print("Missing pChEMBL:", df["pchembl_value"].isna().sum())
print("Missing standard value:", df["standard_value"].isna().sum())

df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
exact = df[
    (df["target_chembl_id"] == "CHEMBL203") &
    (df["standard_type"].str.upper() == "IC50") &
    (df["standard_relation"] == "=") &
    (df["standard_units"].str.lower() == "nm") &
    (df["standard_value"] > 0)
].copy()

exact["measured_pIC50"] = 9 - np.log10(exact["standard_value"])

print("\n=== EXACT IC50 IN nM ===")
print("Eligible measurement rows:", len(exact))
print("Unique eligible molecules:", exact["molecule_chembl_id"].nunique())
print("Measured pIC50 range:",
      exact["measured_pIC50"].min(),
      "to",
      exact["measured_pIC50"].max())
print("\nSaved audit subset: verified_egfr_exact_ic50_audit.csv")
exact.to_csv("verified_egfr_exact_ic50_audit.csv", index=False)
