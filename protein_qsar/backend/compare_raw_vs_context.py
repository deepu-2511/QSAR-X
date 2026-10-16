
from pathlib import Path
import pandas as pd

root = Path("..")
out = Path("source_validation_output")
out.mkdir(exist_ok=True)

raw_path = root / "CHEMBL203_IC50_raw.csv"
context_path = out / "all_egfr_measurements_with_assay_context.csv"

raw = pd.read_csv(raw_path, low_memory=False)
ctx = pd.read_csv(context_path, low_memory=False)

key = "activity_id"
if key not in raw.columns or key not in ctx.columns:
    raise ValueError("activity_id column missing from one of the files.")

for df in (raw, ctx):
    df[key] = pd.to_numeric(df[key], errors="coerce")

print("RAW rows:", len(raw))
print("CONTEXT rows:", len(ctx))
print("RAW unique activity IDs:", raw[key].nunique())
print("CONTEXT unique activity IDs:", ctx[key].nunique())

if raw[key].isna().any() or ctx[key].isna().any():
    raise ValueError("Missing activity IDs detected.")

if raw[key].duplicated().any() or ctx[key].duplicated().any():
    raise ValueError("Duplicate activity IDs detected; investigate before merging.")

compare_cols = [
    "assay_chembl_id",
    "molecule_chembl_id",
    "target_chembl_id",
    "standard_type",
    "standard_relation",
    "standard_units",
    "standard_value",
    "smiles",
]

# Context file uses target_chembl_id_raw for the original target column.
if "target_chembl_id" not in ctx.columns and "target_chembl_id_raw" in ctx.columns:
    ctx["target_chembl_id"] = ctx["target_chembl_id_raw"]

compare_cols = [
    c for c in compare_cols if c in raw.columns and c in ctx.columns
]

merged = raw[[key] + compare_cols].merge(
    ctx[[key] + compare_cols],
    on=key,
    how="outer",
    suffixes=("_raw", "_context"),
    indicator=True,
    validate="one_to_one",
)

print("\nRECORD MATCH")
print(merged["_merge"].value_counts(dropna=False).to_string())

mismatch_rows = []
for col in compare_cols:
    a = merged[f"{col}_raw"]
    b = merged[f"{col}_context"]

    if col == "standard_value":
        av = pd.to_numeric(a, errors="coerce")
        bv = pd.to_numeric(b, errors="coerce")
        bad = ~(
            (av == bv) |
            (av.isna() & bv.isna())
        )
    else:
        av = a.fillna("<MISSING>").astype(str).str.strip()
        bv = b.fillna("<MISSING>").astype(str).str.strip()
        bad = av != bv

    count = int(bad.sum())
    mismatch_rows.append({
        "column": col,
        "mismatch_count": count,
    })
    print(f"{col}: {count} mismatches")

mismatch_report = pd.DataFrame(mismatch_rows)
mismatch_report.to_csv(
    out / "raw_vs_context_column_mismatches.csv", index=False
)

merged.loc[merged["_merge"] != "both"].to_csv(
    out / "raw_vs_context_missing_records.csv", index=False
)

summary = {
    "raw_rows": len(raw),
    "context_rows": len(ctx),
    "raw_unique_activity_ids": raw[key].nunique(),
    "context_unique_activity_ids": ctx[key].nunique(),
    "matched_activity_ids": int((merged["_merge"] == "both").sum()),
    "raw_only_records": int((merged["_merge"] == "left_only").sum()),
    "context_only_records": int((merged["_merge"] == "right_only").sum()),
    "total_column_mismatches": int(mismatch_report["mismatch_count"].sum()),
}

pd.DataFrame([summary]).to_csv(
    out / "raw_vs_context_reconciliation_summary.csv", index=False
)

print("\nSUMMARY")
for name, value in summary.items():
    print(f"{name}: {value}")

print("\nSaved reports in:", out.resolve())
print("Original data and model were not modified.")
