import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# STEP 09 — EXTERNAL / INDEPENDENT-STYLE VALIDATION
# ============================================================

print("=" * 75)
print("STEP 09 — EXTERNAL / INDEPENDENT-STYLE VALIDATION")
print("=" * 75)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "CHEMBL203_train.csv"
VALIDATION_FILE = "CHEMBL203_validation.csv"
TEST_FILE = "CHEMBL203_test.csv"

RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 400,
    "max_features": 0.3,
    "min_samples_leaf": 2,
    "max_depth": 40,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

FP_RADIUS = 2
FP_BITS = 2048


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("\nChecking input files...")

required_files = [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"Required file not found: {file}"
        )

print("All required files found.")


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)


print("\nDataset sizes:")
print(f"Training:   {len(train_df)}")
print(f"Validation: {len(validation_df)}")
print(f"Test:       {len(test_df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "canonical_smiles",
    "pIC50"
]

for name, df in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Test", test_df)
]:
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"{column} column missing from {name} dataset."
            )

print("\nRequired columns verified.")
print("Using: canonical_smiles + pIC50")


# ============================================================
# MORGAN FINGERPRINT FUNCTION
# ============================================================

def smiles_to_morgan(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=FP_RADIUS,
        nBits=FP_BITS
    )

    arr = np.zeros(
        (FP_BITS,),
        dtype=np.uint8
    )

    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, arr)

    return arr


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df, dataset_name):

    print(f"\nGenerating fingerprints for {dataset_name}...")

    fingerprints = []
    valid_rows = []

    invalid_count = 0

    for i, smiles in enumerate(df["canonical_smiles"]):

        fp = smiles_to_morgan(smiles)

        if fp is None:
            invalid_count += 1
            continue

        fingerprints.append(fp)
        valid_rows.append(i)

    clean_df = df.iloc[valid_rows].copy()
    clean_df.reset_index(drop=True, inplace=True)

    X = np.array(fingerprints)

    print(
        f"{dataset_name}: "
        f"{len(clean_df)} valid molecules, "
        f"{invalid_count} invalid molecules"
    )

    print(
        f"{dataset_name} fingerprint shape: {X.shape}"
    )

    return X, clean_df


# ============================================================
# GENERATE FINGERPRINTS
# ============================================================

print("\nGenerating fingerprints...")

X_train, train_clean = prepare_features(
    train_df,
    "Training"
)

X_validation, validation_clean = prepare_features(
    validation_df,
    "Validation"
)

X_test, test_clean = prepare_features(
    test_df,
    "Test"
)

y_train = train_clean["pIC50"].values
y_validation = validation_clean["pIC50"].values
y_test = test_clean["pIC50"].values


# ============================================================
# MODEL 1
# TRAIN ONLY ON TRAINING DATA
# ============================================================

print("\n" + "=" * 75)
print("MODEL 1 — TRAINING DATA ONLY")
print("=" * 75)

model_train_only = ExtraTreesRegressor(
    **MODEL_PARAMS
)

print("\nTraining optimized Extra Trees model...")

model_train_only.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# VALIDATION SET
# ============================================================

print("\nEvaluating on validation set...")

validation_pred = model_train_only.predict(
    X_validation
)

validation_r2 = r2_score(
    y_validation,
    validation_pred
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_pred
    )
)

validation_mae = mean_absolute_error(
    y_validation,
    validation_pred
)


print("\nValidation Results")
print("-" * 40)

print(f"R²   : {validation_r2:.4f}")
print(f"RMSE : {validation_rmse:.4f}")
print(f"MAE  : {validation_mae:.4f}")


# ============================================================
# MODEL 2
# TRAIN + VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("MODEL 2 — TRAIN + VALIDATION")
print("=" * 75)

X_train_validation = np.vstack(
    [
        X_train,
        X_validation
    ]
)

y_train_validation = np.concatenate(
    [
        y_train,
        y_validation
    ]
)

print("\nCombined training dataset:")
print(
    f"Molecules: {len(y_train_validation)}"
)


model_final = ExtraTreesRegressor(
    **MODEL_PARAMS
)

print("\nTraining final optimized model...")

model_final.fit(
    X_train_validation,
    y_train_validation
)

print("Final model training completed.")


# ============================================================
# FINAL TEST SET
# ============================================================

print("\nEvaluating on untouched test set...")

test_pred = model_final.predict(
    X_test
)

test_r2 = r2_score(
    y_test,
    test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred
    )
)

test_mae = mean_absolute_error(
    y_test,
    test_pred
)


print("\n" + "=" * 75)
print("FINAL TEST RESULTS")
print("=" * 75)

print(f"R²   : {test_r2:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"MAE  : {test_mae:.4f}")


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

residuals = y_test - test_pred

absolute_errors = np.abs(residuals)

print("\n" + "=" * 75)
print("RESIDUAL / ERROR ANALYSIS")
print("=" * 75)

print(
    f"Mean residual       : {np.mean(residuals):.4f}"
)

print(
    f"Median residual     : {np.median(residuals):.4f}"
)

print(
    f"Residual std        : {np.std(residuals):.4f}"
)

print(
    f"Maximum absolute error : "
    f"{np.max(absolute_errors):.4f}"
)

print(
    f"Minimum absolute error : "
    f"{np.min(absolute_errors):.4f}"
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

within_05 = np.sum(
    absolute_errors <= 0.5
)

within_10 = np.sum(
    absolute_errors <= 1.0
)

within_15 = np.sum(
    absolute_errors <= 1.5
)

total = len(absolute_errors)


print("\nPrediction error distribution:")

print(
    f"Error <= 0.5 pIC50 : "
    f"{within_05}/{total} "
    f"({within_05 / total * 100:.2f}%)"
)

print(
    f"Error <= 1.0 pIC50 : "
    f"{within_10}/{total} "
    f"({within_10 / total * 100:.2f}%)"
)

print(
    f"Error <= 1.5 pIC50 : "
    f"{within_15}/{total} "
    f"({within_15 / total * 100:.2f}%)"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = test_clean.copy()

prediction_df["observed_pIC50"] = y_test
prediction_df["predicted_pIC50"] = test_pred
prediction_df["residual"] = residuals
prediction_df["absolute_error"] = absolute_errors


prediction_df.to_csv(
    "external_validation_predictions.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "dataset": [
            "validation",
            "test"
        ],

        "n_samples": [
            len(y_validation),
            len(y_test)
        ],

        "R2": [
            validation_r2,
            test_r2
        ],

        "RMSE": [
            validation_rmse,
            test_rmse
        ],

        "MAE": [
            validation_mae,
            test_mae
        ]
    }
)


summary.to_csv(
    "external_validation_summary.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 09 COMPLETED")
print("=" * 75)

print("\nValidation:")
print(f"  R²   = {validation_r2:.4f}")
print(f"  RMSE = {validation_rmse:.4f}")
print(f"  MAE  = {validation_mae:.4f}")

print("\nUntouched Test:")
print(f"  R²   = {test_r2:.4f}")
print(f"  RMSE = {test_rmse:.4f}")
print(f"  MAE  = {test_mae:.4f}")

print("\nFiles saved:")
print("  external_validation_predictions.csv")
print("  external_validation_summary.csv")

print("\nNOTE:")
print(
    "The test set is a hold-out set from the same "
    "ChEMBL-derived dataset. It is not a truly independent "
    "external experimental dataset."
)

print("=" * 75)