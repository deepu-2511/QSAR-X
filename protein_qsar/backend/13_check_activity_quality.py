import pandas as pd
import numpy as np

df = pd.read_csv("verified_egfr_exact_ic50_audit.csv")
df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
df["measured_pIC50"] = pd.to_numeric(df["measured_pIC50"], errors="coerce")

print("=== EXTREME IC50 RECORDS ===")
cols = [
    "molecule_chembl_id",
    "assay_chembl_id",
    "document_chembl_id",
    "standard_value",
    "standard_units",
    "measured_pIC50",
    "pchembl_value",
]
print(df.nlargest(15, "measured_pIC50")[cols].to_string(index=False))

print("\n=== REPEATED MEASUREMENTS PER MOLECULE ===")
counts = df.groupby("molecule_chembl_id").size()
print("Molecules with repeated measurements:", int((counts > 1).sum()))
print("Maximum measurements for one molecule:", int(counts.max()))

print("\n=== WITHIN-MOLECULE ACTIVITY SPREAD ===")
spread = df.groupby("molecule_chembl_id")["measured_pIC50"].agg(
    ["count", "min", "max", "median"]
)
spread["range"] = spread["max"] - spread["min"]
print("Molecules with pIC50 range > 2:", int((spread["range"] > 2).sum()))
print(spread.nlargest(10, "range").to_string())

print("\nSaved: verified_egfr_activity_quality_report.csv")
spread.to_csv("verified_egfr_activity_quality_report.csv")
