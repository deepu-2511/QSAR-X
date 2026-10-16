
from pathlib import Path
import pandas as pd

out = Path("source_validation_output")
data_path = out / "all_egfr_measurements_with_assay_context.csv"
mutant_path = out / "assay_mutant_manual_review.csv"

data = pd.read_csv(data_path, low_memory=False)
mutants = pd.read_csv(mutant_path, low_memory=False)

required = {"assay_chembl_id", "standard_value", "standard_units",
            "standard_type", "molecule_chembl_id", "smiles"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Missing columns in measurement file: {sorted(missing)}")

if "assay_chembl_id" not in mutants.columns:
    raise ValueError("assay_chembl_id missing from mutant review file")

mutant_ids = set(mutants["assay_chembl_id"].dropna().astype(str))
data["is_flagged_mutant_assay"] = (
    data["assay_chembl_id"].astype(str).isin(mutant_ids)
)

data["standard_value"] = pd.to_numeric(data["standard_value"], errors="coerce")
data["pIC50_recalculated"] = None

valid = (
    data["standard_type"].astype(str).str.upper().eq("IC50")
    & data["standard_units"].astype(str).str.lower().eq("nm")
    & data["standard_value"].gt(0)
)

data.loc[valid, "pIC50_recalculated"] = (
    9 - data.loc[valid, "standard_value"].map(__import__("math").log10)
)

data["pIC50_recalculated"] = pd.to_numeric(
    data["pIC50_recalculated"], errors="coerce"
)

data["activity_group"] = data["is_flagged_mutant_assay"].map({
    True: "FLAGGED_MUTANT_ASSAY",
    False: "OTHER_ASSAYS"
})

print("MEASUREMENT COUNTS")
print(data["activity_group"].value_counts().to_string())

print("\nUNIQUE COMPOUNDS")
print(data.groupby("activity_group")["molecule_chembl_id"].nunique().to_string())

print("\nPIC50 DESCRIPTIVE STATISTICS")
stats = data.groupby("activity_group")["pIC50_recalculated"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)
print(stats.to_string())

print("\nASSAY COUNTS")
print(data.groupby("activity_group")["assay_chembl_id"].nunique().to_string())

# Output rows are flagged for review, not excluded from training automatically.
data.to_csv(out / "mutant_context_measurement_analysis.csv", index=False)
stats.to_csv(out / "mutant_context_pIC50_summary.csv")

print("\nSaved:")
print(out / "mutant_context_measurement_analysis.csv")
print(out / "mutant_context_pIC50_summary.csv")
print("\nNo source data deleted. No model retrained.")
