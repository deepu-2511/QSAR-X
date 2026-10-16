from pathlib import Path
import pandas as pd

OUT = Path("source_validation_output")

assay_file = OUT / "all_egfr_assay_metadata.csv"
measurement_file = OUT / "all_egfr_measurements_with_assay_context.csv"

if not assay_file.exists() or not measurement_file.exists():
    raise FileNotFoundError("Required assay audit CSV files are missing.")

assays = pd.read_csv(assay_file, low_memory=False)
measurements = pd.read_csv(measurement_file, low_memory=False)

def find_col(df, name):
    return next((c for c in df.columns if c.lower() == name.lower()), None)

def show_count(df, col, title):
    if col and col in df.columns:
        print(f"\n{title}")
        print(df[col].fillna("MISSING").astype(str).value_counts().to_string())

group_col = find_col(assays, "assay_context_group")
id_col = find_col(assays, "assay_chembl_id")
desc_col = find_col(assays, "description")
target_col = find_col(assays, "target_chembl_id")
variant_col = find_col(assays, "variant_id")
doc_col = find_col(assays, "document_chembl_id")
confidence_col = find_col(assays, "confidence_score")
relationship_col = find_col(assays, "relationship_type")

print("=" * 72)
print("EGFR ASSAY CONTEXT AUDIT - REVIEW ONLY")
print("=" * 72)
print(f"Assay metadata rows: {len(assays)}")
print(f"Measurement rows: {len(measurements)}")
print(f"Unique assay IDs: {assays[id_col].nunique() if id_col else 'ID column missing'}")
print("No original data or model will be modified.")

print("\nASSAY METADATA COLUMNS")
print(assays.columns.tolist())
print("\nMEASUREMENT FILE COLUMNS")
print(measurements.columns.tolist())

show_count(assays, group_col, "ASSAYS BY CONTEXT")
show_count(measurements, find_col(measurements, "assay_context_group"),
           "MEASUREMENTS BY CONTEXT")
show_count(assays, target_col, "ASSAYS BY TARGET")
show_count(assays, confidence_col, "ASSAYS BY CONFIDENCE SCORE")
show_count(assays, relationship_col, "ASSAYS BY RELATIONSHIP TYPE")

# Save a compact context count report.
if group_col:
    counts = (
        assays.groupby(group_col, dropna=False)
        .agg(assay_count=(id_col, "nunique") if id_col else (group_col, "size"))
        .reset_index()
    )
    m_group = find_col(measurements, "assay_context_group")
    if m_group and id_col:
        measurement_counts = (
            measurements.groupby(m_group, dropna=False)
            .agg(
                measurement_count=(m_group, "size"),
                unique_compounds=(
                    find_col(measurements, "molecule_chembl_id") or m_group,
                    "nunique"
                )
            ).reset_index()
        )
        counts = counts.merge(
            measurement_counts,
            left_on=group_col, right_on=m_group, how="outer"
        )
    counts.to_csv(OUT / "assay_context_detailed_counts.csv", index=False)

# Flag likely mutant assays for manual review; do not remove them.
if desc_col:
    mutant_pattern = (
        r"\bmutant\b|L858R|T790M|L861Q|G719|exon\s*20|"
        r"deletion|del\d|C797S|G719S"
    )
    mutant_mask = assays[desc_col].fillna("").str.contains(
        mutant_pattern, case=False, regex=True
    )
else:
    mutant_mask = pd.Series(False, index=assays.index)

if group_col:
    mutant_mask |= assays[group_col].fillna("").str.contains(
        "MUTANT", case=False
    )

mutant_review = assays.loc[mutant_mask].copy()
mutant_review.to_csv(OUT / "assay_mutant_manual_review.csv", index=False)

# Flag descriptions needing review, including the broad OTHER category.
if desc_col:
    missing_desc = assays[desc_col].isna() | (
        assays[desc_col].astype(str).str.strip() == ""
    )
else:
    missing_desc = pd.Series(True, index=assays.index)

if group_col:
    other_mask = assays[group_col].fillna("").str.contains(
        "OTHER|UNCLEAR|REVIEW", case=False, regex=True
    )
else:
    other_mask = pd.Series(False, index=assays.index)

review_mask = missing_desc | other_mask
review_cols = [
    c for c in [
        id_col, desc_col, group_col, target_col, variant_col,
        doc_col, confidence_col, relationship_col
    ] if c and c in assays.columns
]
review_queue = assays.loc[review_mask, review_cols].copy()
review_queue.to_csv(OUT / "assay_manual_review_queue.csv", index=False)

# Export measurements linked to the mutant assay IDs when possible.
if id_col:
    measurement_id_col = find_col(measurements, "assay_chembl_id")
    if measurement_id_col:
        mutant_ids = set(mutant_review[id_col].dropna().astype(str))
        linked = measurements[
            measurements[measurement_id_col].astype(str).isin(mutant_ids)
        ].copy()
        linked.to_csv(
            OUT / "mutant_assay_linked_measurements.csv", index=False
        )
        print(f"\nMeasurements linked to flagged mutant assays: {len(linked)}")

# Generate a plain-text audit summary.
summary_path = OUT / "assay_context_audit_summary.txt"
with summary_path.open("w", encoding="utf-8") as f:
    f.write("EGFR ASSAY CONTEXT AUDIT - REVIEW ONLY\n")
    f.write("=" * 48 + "\n")
    f.write(f"Assay metadata rows: {len(assays)}\n")
    f.write(f"Unique assay IDs: {assays[id_col].nunique() if id_col else 'unknown'}\n")
    f.write(f"Measurement rows: {len(measurements)}\n")
    f.write(f"Flagged mutant assays: {len(mutant_review)}\n")
    f.write(f"Manual review queue rows: {len(review_queue)}\n")
    f.write(f"Missing assay descriptions: {int(missing_desc.sum())}\n\n")
    if group_col:
        f.write("ASSAY COUNTS BY CONTEXT\n")
        f.write(assays[group_col].fillna("MISSING").value_counts().to_string())
        f.write("\n\n")
    if target_col:
        f.write("TARGET COUNTS\n")
        f.write(assays[target_col].fillna("MISSING").value_counts().to_string())
        f.write("\n\n")
    f.write("IMPORTANT: Flags are for manual review, not automatic exclusion.\n")
    f.write("No training data or model was changed by this script.\n")

print("\nFILES CREATED")
for name in [
    "assay_context_detailed_counts.csv",
    "assay_mutant_manual_review.csv",
    "assay_manual_review_queue.csv",
    "mutant_assay_linked_measurements.csv",
    "assay_context_audit_summary.txt",
]:
    p = OUT / name
    if p.exists():
        print(f"{name}: {p.stat().st_size:,} bytes")

print("\nAUDIT COMPLETE. No compounds were deleted and no model was retrained.")
