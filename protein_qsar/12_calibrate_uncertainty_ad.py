import os
import warnings
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ============================================================
# STEP 12
# Uncertainty + Applicability Domain Calibration
# ============================================================

RANDOM_STATE = 42

TRAIN_FILE = "CHEMBL203_train.csv"
VALIDATION_FILE = "CHEMBL203_validation.csv"
TEST_FILE = "CHEMBL203_test.csv"

OUTPUT_DIR = "step12_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Final optimized Extra Trees parameters from Step 08
# ------------------------------------------------------------

MODEL_PARAMS = {
    "n_estimators": 400,
    "max_features": 0.3,
    "min_samples_leaf": 2,
    "max_depth": 40,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ------------------------------------------------------------
# Morgan fingerprint
# ------------------------------------------------------------

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return morgan_generator.GetFingerprint(mol)


def fp_matrix(df):
    fps = []
    valid_rows = []

    for i, smiles in enumerate(df["canonical_smiles"]):

        fp = smiles_to_fp(smiles)

        if fp is not None:
            fps.append(fp)
            valid_rows.append(i)

    X = np.array(
        [np.asarray(fp) for fp in fps],
        dtype=np.uint8
    )

    return X, valid_rows


def max_tanimoto_to_reference(query_fp, reference_fps):

    similarities = DataStructs.BulkTanimotoSimilarity(
        query_fp,
        reference_fps
    )

    return max(similarities)


def prediction_statistics(model, X):

    all_predictions = []

    for estimator in model.estimators_:
        all_predictions.append(
            estimator.predict(X)
        )

    all_predictions = np.array(all_predictions)

    mean_prediction = np.mean(all_predictions, axis=0)
    prediction_std = np.std(all_predictions, axis=0)

    return mean_prediction, prediction_std


print("\n" + "=" * 70)
print("STEP 12 - UNCERTAINTY + APPLICABILITY DOMAIN CALIBRATION")
print("=" * 70)

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print("\nDataset sizes:")
print("Training   :", len(train))
print("Validation :", len(validation))
print("Test       :", len(test))

# ------------------------------------------------------------
# Prepare fingerprints
# ------------------------------------------------------------

print("\nGenerating Morgan fingerprints...")

X_train, train_valid = fp_matrix(train)
X_val, val_valid = fp_matrix(validation)
X_test, test_valid = fp_matrix(test)

train = train.iloc[train_valid].reset_index(drop=True)
validation = validation.iloc[val_valid].reset_index(drop=True)
test = test.iloc[test_valid].reset_index(drop=True)

y_train = train["pIC50"].values
y_val = validation["pIC50"].values
y_test = test["pIC50"].values

print("Training matrix   :", X_train.shape)
print("Validation matrix :", X_val.shape)
print("Test matrix       :", X_test.shape)

# ------------------------------------------------------------
# Train model ONLY on training data
# ------------------------------------------------------------

print("\nTraining Extra Trees model...")

model = ExtraTreesRegressor(**MODEL_PARAMS)

model.fit(X_train, y_train)

print("Model training complete.")

# ------------------------------------------------------------
# Validation predictions
# ------------------------------------------------------------

val_pred, val_uncertainty = prediction_statistics(
    model,
    X_val
)

val_error = np.abs(y_val - val_pred)

val_r2 = r2_score(y_val, val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_mae = mean_absolute_error(y_val, val_pred)

print("\nValidation performance:")
print(f"R²   : {val_r2:.4f}")
print(f"RMSE : {val_rmse:.4f}")
print(f"MAE  : {val_mae:.4f}")

# ------------------------------------------------------------
# Build reference fingerprint library
# ------------------------------------------------------------

print("\nBuilding chemical-space reference...")

reference_fps = [
    smiles_to_fp(smiles)
    for smiles in train["canonical_smiles"]
]

reference_fps = [
    fp for fp in reference_fps
    if fp is not None
]

# ------------------------------------------------------------
# Calculate validation chemical similarity
# ------------------------------------------------------------

print("Calculating validation chemical similarity...")

val_similarity = []

for smiles in validation["canonical_smiles"]:

    fp = smiles_to_fp(smiles)

    sim = max_tanimoto_to_reference(
        fp,
        reference_fps
    )

    val_similarity.append(sim)

val_similarity = np.array(val_similarity)

# ------------------------------------------------------------
# AD calibration
#
# Instead of assuming 0.70 / 0.50 are universal,
# find thresholds based on validation prediction error.
# ------------------------------------------------------------

print("\nCalibrating Applicability Domain...")

ad_candidates = np.arange(
    0.30,
    0.91,
    0.05
)

ad_results = []

for threshold in ad_candidates:

    inside = val_similarity >= threshold

    count = np.sum(inside)

    if count == 0:
        continue

    mae = mean_absolute_error(
        y_val[inside],
        val_pred[inside]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val[inside],
            val_pred[inside]
        )
    )

    coverage = count / len(validation)

    ad_results.append({
        "similarity_threshold": threshold,
        "n_compounds": count,
        "coverage": coverage,
        "MAE": mae,
        "RMSE": rmse
    })

ad_results_df = pd.DataFrame(ad_results)

ad_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step12_ad_threshold_calibration.csv"
    ),
    index=False
)

print("\nAD threshold calibration:")
print(ad_results_df.to_string(index=False))

# ------------------------------------------------------------
# Choose calibrated threshold
#
# Requirement:
# retain at least 50% validation coverage,
# then select the threshold with lowest MAE.
# ------------------------------------------------------------

eligible = ad_results_df[
    ad_results_df["coverage"] >= 0.50
]

if len(eligible) > 0:

    best_row = eligible.loc[
        eligible["MAE"].idxmin()
    ]

    calibrated_ad_threshold = float(
        best_row["similarity_threshold"]
    )

else:

    calibrated_ad_threshold = 0.50

print(
    f"\nCalibrated AD threshold: "
    f"{calibrated_ad_threshold:.2f}"
)

# ------------------------------------------------------------
# Uncertainty calibration
# ------------------------------------------------------------

print("\nCalibrating prediction uncertainty...")

uncertainty_percentiles = [
    50,
    60,
    70,
    75,
    80,
    85,
    90,
    95
]

uncertainty_results = []

for percentile in uncertainty_percentiles:

    threshold = np.percentile(
        val_uncertainty,
        percentile
    )

    low_uncertainty = val_uncertainty <= threshold
    high_uncertainty = val_uncertainty > threshold

    low_mae = mean_absolute_error(
        y_val[low_uncertainty],
        val_pred[low_uncertainty]
    )

    high_mae = mean_absolute_error(
        y_val[high_uncertainty],
        val_pred[high_uncertainty]
    )

    uncertainty_results.append({
        "percentile": percentile,
        "uncertainty_threshold": threshold,
        "low_uncertainty_count": np.sum(low_uncertainty),
        "high_uncertainty_count": np.sum(high_uncertainty),
        "low_uncertainty_MAE": low_mae,
        "high_uncertainty_MAE": high_mae
    })

uncertainty_results_df = pd.DataFrame(
    uncertainty_results
)

uncertainty_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step12_uncertainty_calibration.csv"
    ),
    index=False
)

print(
    uncertainty_results_df.to_string(index=False)
)

# ------------------------------------------------------------
# Choose 80th percentile uncertainty threshold
# ------------------------------------------------------------

uncertainty_threshold = np.percentile(
    val_uncertainty,
    80
)

print(
    f"\nCalibrated uncertainty threshold "
    f"(80th percentile): {uncertainty_threshold:.4f}"
)

# ------------------------------------------------------------
# Combined validation analysis
# ------------------------------------------------------------

validation_output = validation.copy()

validation_output["observed_pIC50"] = y_val
validation_output["predicted_pIC50"] = val_pred
validation_output["absolute_error"] = val_error
validation_output["prediction_uncertainty"] = val_uncertainty
validation_output["max_tanimoto_similarity"] = val_similarity

validation_output["AD_status"] = np.where(
    val_similarity >= calibrated_ad_threshold,
    "IN_DOMAIN",
    "OUT_OF_DOMAIN"
)

validation_output["uncertainty_status"] = np.where(
    val_uncertainty <= uncertainty_threshold,
    "LOW_UNCERTAINTY",
    "HIGH_UNCERTAINTY"
)

validation_output["prediction_status"] = np.where(
    (val_similarity >= calibrated_ad_threshold)
    & (val_uncertainty <= uncertainty_threshold),
    "HIGH_CONFIDENCE",
    np.where(
        (val_similarity >= calibrated_ad_threshold)
        & (val_uncertainty > uncertainty_threshold),
        "CAUTION_HIGH_UNCERTAINTY",
        "OUT_OF_DOMAIN"
    )
)

validation_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step12_validation_calibration.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# Evaluate relationship between uncertainty and error
# ------------------------------------------------------------

uncertainty_error_correlation = np.corrcoef(
    val_uncertainty,
    val_error
)[0, 1]

similarity_error_correlation = np.corrcoef(
    val_similarity,
    val_error
)[0, 1]

print("\nCalibration relationships:")
print(
    f"Uncertainty vs absolute error correlation: "
    f"{uncertainty_error_correlation:.4f}"
)

print(
    f"Similarity vs absolute error correlation: "
    f"{similarity_error_correlation:.4f}"
)

# ------------------------------------------------------------
# FINAL TEST EVALUATION
#
# Test set is used ONLY after calibration.
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("FINAL TEST EVALUATION")
print("-" * 70)

test_pred, test_uncertainty = prediction_statistics(
    model,
    X_test
)

test_error = np.abs(
    y_test - test_pred
)

test_similarity = []

for smiles in test["canonical_smiles"]:

    fp = smiles_to_fp(smiles)

    sim = max_tanimoto_to_reference(
        fp,
        reference_fps
    )

    test_similarity.append(sim)

test_similarity = np.array(test_similarity)

test_ad = np.where(
    test_similarity >= calibrated_ad_threshold,
    "IN_DOMAIN",
    "OUT_OF_DOMAIN"
)

test_uncertainty_status = np.where(
    test_uncertainty <= uncertainty_threshold,
    "LOW_UNCERTAINTY",
    "HIGH_UNCERTAINTY"
)

test_status = np.where(
    (test_similarity >= calibrated_ad_threshold)
    & (test_uncertainty <= uncertainty_threshold),
    "HIGH_CONFIDENCE",
    np.where(
        (test_similarity >= calibrated_ad_threshold)
        & (test_uncertainty > uncertainty_threshold),
        "CAUTION_HIGH_UNCERTAINTY",
        "OUT_OF_DOMAIN"
    )
)

test_output = test.copy()

test_output["observed_pIC50"] = y_test
test_output["predicted_pIC50"] = test_pred
test_output["absolute_error"] = test_error
test_output["prediction_uncertainty"] = test_uncertainty
test_output["max_tanimoto_similarity"] = test_similarity
test_output["AD_status"] = test_ad
test_output["uncertainty_status"] = test_uncertainty_status
test_output["prediction_status"] = test_status

test_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step12_test_predictions.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# Test metrics
# ------------------------------------------------------------

test_r2 = r2_score(y_test, test_pred)
test_rmse = np.sqrt(
    mean_squared_error(y_test, test_pred)
)
test_mae = mean_absolute_error(
    y_test,
    test_pred
)

print("\nTest performance:")
print(f"R²   : {test_r2:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"MAE  : {test_mae:.4f}")

print("\nTest prediction categories:")
print(
    pd.Series(test_status)
    .value_counts()
)

# ------------------------------------------------------------
# Error by calibrated AD status
# ------------------------------------------------------------

test_summary = []

for status in [
    "HIGH_CONFIDENCE",
    "CAUTION_HIGH_UNCERTAINTY",
    "OUT_OF_DOMAIN"
]:

    mask = test_status == status

    if np.sum(mask) == 0:
        continue

    test_summary.append({
        "prediction_status": status,
        "n_compounds": int(np.sum(mask)),
        "coverage": float(np.mean(mask)),
        "MAE": mean_absolute_error(
            y_test[mask],
            test_pred[mask]
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_test[mask],
                test_pred[mask]
            )
        ),
        "mean_uncertainty": float(
            np.mean(test_uncertainty[mask])
        ),
        "mean_similarity": float(
            np.mean(test_similarity[mask])
        )
    })

test_summary_df = pd.DataFrame(test_summary)

test_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step12_test_error_by_confidence.csv"
    ),
    index=False
)

print("\nError by prediction confidence:")
print(
    test_summary_df.to_string(index=False)
)

# ------------------------------------------------------------
# Save overall summary
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "validation_R2": val_r2,
    "validation_RMSE": val_rmse,
    "validation_MAE": val_mae,
    "test_R2": test_r2,
    "test_RMSE": test_rmse,
    "test_MAE": test_mae,
    "calibrated_AD_threshold": calibrated_ad_threshold,
    "uncertainty_threshold": uncertainty_threshold,
    "uncertainty_error_correlation": uncertainty_error_correlation,
    "similarity_error_correlation": similarity_error_correlation,
    "test_high_confidence_count": np.sum(
        test_status == "HIGH_CONFIDENCE"
    ),
    "test_caution_count": np.sum(
        test_status == "CAUTION_HIGH_UNCERTAINTY"
    ),
    "test_out_of_domain_count": np.sum(
        test_status == "OUT_OF_DOMAIN"
    )
}])

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step12_summary.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("STEP 12 COMPLETED")
print("=" * 70)

print("\nOutput files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", filename)

print("\nNext step:")
print("Step 13 → Model packaging + prediction engine")