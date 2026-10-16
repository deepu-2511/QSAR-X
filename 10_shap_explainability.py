import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import shap


# ============================================================
# STEP 10 — SHAP EXPLAINABILITY + PREDICTION ANALYSIS
# ============================================================

print("=" * 75)
print("STEP 10 — SHAP EXPLAINABILITY + PREDICTION ANALYSIS")
print("=" * 75)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "CHEMBL203_train.csv"
VALIDATION_FILE = "CHEMBL203_validation.csv"
TEST_FILE = "CHEMBL203_test.csv"

OUTPUT_DIR = "step10_results"

RANDOM_STATE = 42

FP_RADIUS = 2
FP_BITS = 2048

MODEL_PARAMS = {
    "n_estimators": 400,
    "max_features": 0.3,
    "min_samples_leaf": 2,
    "max_depth": 40,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print(f"\nOutput directory: {OUTPUT_DIR}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training:   {len(train_df)}")
print(f"Validation: {len(validation_df)}")
print(f"Test:       {len(test_df)}")


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
        FP_BITS,
        dtype=np.uint8
    )

    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(
        fp,
        arr
    )

    return arr


# ============================================================
# FEATURE GENERATION
# ============================================================

def prepare_features(df, dataset_name):

    fingerprints = []
    valid_indices = []

    invalid = 0

    print(
        f"\nGenerating Morgan fingerprints: "
        f"{dataset_name}"
    )

    for i, smiles in enumerate(
        df["canonical_smiles"]
    ):

        fp = smiles_to_morgan(smiles)

        if fp is None:
            invalid += 1
            continue

        fingerprints.append(fp)
        valid_indices.append(i)

    clean_df = df.iloc[
        valid_indices
    ].copy()

    clean_df.reset_index(
        drop=True,
        inplace=True
    )

    X = np.asarray(
        fingerprints,
        dtype=np.float32
    )

    print(
        f"{dataset_name}: "
        f"{len(clean_df)} valid, "
        f"{invalid} invalid"
    )

    print(
        f"Matrix shape: {X.shape}"
    )

    return X, clean_df


# ============================================================
# PREPARE ALL DATA
# ============================================================

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
# COMBINE TRAIN + VALIDATION
# ============================================================

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

print("\nFinal training matrix:")
print(
    X_train_validation.shape
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FINAL EXTRA TREES MODEL")
print("=" * 75)

model = ExtraTreesRegressor(
    **MODEL_PARAMS
)

model.fit(
    X_train_validation,
    y_train_validation
)

print("Final model training complete.")


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

test_predictions = model.predict(
    X_test
)

r2 = r2_score(
    y_test,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

mae = mean_absolute_error(
    y_test,
    test_predictions
)

residuals = (
    y_test -
    test_predictions
)

absolute_errors = np.abs(
    residuals
)


print("\nTest performance:")
print(
    f"R²   : {r2:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"MAE  : {mae:.4f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_clean.copy()

prediction_df[
    "observed_pIC50"
] = y_test

prediction_df[
    "predicted_pIC50"
] = test_predictions

prediction_df[
    "residual"
] = residuals

prediction_df[
    "absolute_error"
] = absolute_errors

prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step10_test_predictions.csv"
    ),
    index=False
)

print(
    "\nSaved: "
    "step10_test_predictions.csv"
)


# ============================================================
# MODEL FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 75)
print("EXTRA TREES FEATURE IMPORTANCE")
print("=" * 75)

feature_importance = (
    model.feature_importances_
)

importance_df = pd.DataFrame(
    {
        "feature": [
            f"ECFP4_bit_{i}"
            for i in range(FP_BITS)
        ],
        "importance": feature_importance
    }
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step10_feature_importance.csv"
    ),
    index=False
)

print("\nTop 20 fingerprint bits:")

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# PLOT 1 — PREDICTED VS OBSERVED
# ============================================================

print("\nCreating predicted vs observed plot...")

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.65,
    s=30
)

min_value = min(
    y_test.min(),
    test_predictions.min()
)

max_value = max(
    y_test.max(),
    test_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Observed pIC50"
)

plt.ylabel(
    "Predicted pIC50"
)

plt.title(
    f"Observed vs Predicted pIC50\n"
    f"R² = {r2:.4f}, RMSE = {rmse:.4f}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_observed_vs_predicted.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2 — RESIDUALS
# ============================================================

print("Creating residual plot...")

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    test_predictions,
    residuals,
    alpha=0.65,
    s=30
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted pIC50"
)

plt.ylabel(
    "Residual (Observed - Predicted)"
)

plt.title(
    "Residual Analysis"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_residual_plot.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 3 — ERROR DISTRIBUTION
# ============================================================

print("Creating error distribution plot...")

plt.figure(
    figsize=(8, 7)
)

plt.hist(
    absolute_errors,
    bins=30
)

plt.xlabel(
    "Absolute Prediction Error (pIC50)"
)

plt.ylabel(
    "Number of Compounds"
)

plt.title(
    "Prediction Error Distribution"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_error_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SHAP ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("SHAP EXPLAINABILITY")
print("=" * 75)

print(
    "\nCreating SHAP TreeExplainer..."
)

explainer = shap.TreeExplainer(
    model
)


# ============================================================
# USE A REPRESENTATIVE TEST SUBSET
# ============================================================

SHAP_SAMPLE_SIZE = min(
    200,
    len(X_test)
)

rng = np.random.default_rng(
    RANDOM_STATE
)

sample_indices = rng.choice(
    len(X_test),
    size=SHAP_SAMPLE_SIZE,
    replace=False
)

X_shap = X_test[
    sample_indices
]

print(
    f"SHAP sample size: "
    f"{SHAP_SAMPLE_SIZE}"
)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print(
    "\nCalculating SHAP values..."
)

shap_values = explainer.shap_values(
    X_shap
)

shap_values = np.asarray(
    shap_values
)

print(
    f"SHAP matrix shape: "
    f"{shap_values.shape}"
)


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

shap_importance_df = pd.DataFrame(
    {
        "feature": [
            f"ECFP4_bit_{i}"
            for i in range(FP_BITS)
        ],
        "mean_abs_shap": mean_abs_shap
    }
)

shap_importance_df = (
    shap_importance_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)

shap_importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step10_shap_global_importance.csv"
    ),
    index=False
)

print(
    "\nTop 20 SHAP features:"
)

print(
    shap_importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# SHAP BAR PLOT
# ============================================================

print(
    "\nCreating SHAP importance plot..."
)

plt.figure(
    figsize=(10, 8)
)

top_shap = (
    shap_importance_df
    .head(20)
    .sort_values(
        "mean_abs_shap"
    )
)

plt.barh(
    top_shap["feature"],
    top_shap["mean_abs_shap"]
)

plt.xlabel(
    "Mean |SHAP value|"
)

plt.ylabel(
    "Morgan fingerprint bit"
)

plt.title(
    "Top 20 SHAP Features"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_shap_global_importance.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SHAP BEESWARM
# ============================================================

print(
    "\nCreating SHAP summary plot..."
)

feature_names = [
    f"ECFP4_bit_{i}"
    for i in range(FP_BITS)
]

X_shap_df = pd.DataFrame(
    X_shap,
    columns=feature_names
)

shap.summary_plot(
    shap_values,
    X_shap_df,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_shap_summary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP SAMPLE COMPOUND ANALYSIS
# ============================================================

print(
    "\nCreating individual compound SHAP data..."
)

individual_records = []

for local_index in range(
    len(sample_indices)
):

    original_index = (
        sample_indices[
            local_index
        ]
    )

    compound_shap = (
        shap_values[
            local_index
        ]
    )

    top_positive = np.argsort(
        compound_shap
    )[-10:][::-1]

    top_negative = np.argsort(
        compound_shap
    )[:10]

    for rank, feature_index in enumerate(
        top_positive,
        start=1
    ):

        individual_records.append(
            {
                "test_index": int(
                    original_index
                ),
                "smiles": test_clean.iloc[
                    original_index
                ]["canonical_smiles"],
                "observed_pIC50": y_test[
                    original_index
                ],
                "predicted_pIC50": test_predictions[
                    original_index
                ],
                "feature": (
                    f"ECFP4_bit_{feature_index}"
                ),
                "shap_value": compound_shap[
                    feature_index
                ],
                "direction": "positive",
                "rank": rank
            }
        )

    for rank, feature_index in enumerate(
        top_negative,
        start=1
    ):

        individual_records.append(
            {
                "test_index": int(
                    original_index
                ),
                "smiles": test_clean.iloc[
                    original_index
                ]["canonical_smiles"],
                "observed_pIC50": y_test[
                    original_index
                ],
                "predicted_pIC50": test_predictions[
                    original_index
                ],
                "feature": (
                    f"ECFP4_bit_{feature_index}"
                ),
                "shap_value": compound_shap[
                    feature_index
                ],
                "direction": "negative",
                "rank": rank
            }
        )


individual_shap_df = pd.DataFrame(
    individual_records
)

individual_shap_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step10_individual_shap_features.csv"
    ),
    index=False
)


# ============================================================
# ERROR-BASED ANALYSIS
# ============================================================

print(
    "\nCreating error analysis table..."
)

error_analysis = prediction_df.copy()

error_analysis[
    "error_category"
] = pd.cut(
    error_analysis[
        "absolute_error"
    ],
    bins=[
        -np.inf,
        0.5,
        1.0,
        1.5,
        np.inf
    ],
    labels=[
        "<=0.5",
        "0.5-1.0",
        "1.0-1.5",
        ">1.5"
    ]
)

error_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step10_error_analysis.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "Test_R2",
            "Test_RMSE",
            "Test_MAE",
            "Mean_Residual",
            "Median_Residual",
            "Residual_SD",
            "Maximum_Absolute_Error",
            "Error_<=_0.5_percent",
            "Error_<=_1.0_percent",
            "Error_<=_1.5_percent",
            "Error_>_1.5_percent",
            "SHAP_Sample_Size",
            "Fingerprint_Bits",
            "Fingerprint_Radius"
        ],

        "value": [
            r2,
            rmse,
            mae,
            np.mean(residuals),
            np.median(residuals),
            np.std(residuals),
            np.max(absolute_errors),
            np.mean(
                absolute_errors <= 0.5
            ) * 100,
            np.mean(
                absolute_errors <= 1.0
            ) * 100,
            np.mean(
                absolute_errors <= 1.5
            ) * 100,
            np.mean(
                absolute_errors > 1.5
            ) * 100,
            SHAP_SAMPLE_SIZE,
            FP_BITS,
            FP_RADIUS
        ]
    }
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step10_summary.csv"
    ),
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("STEP 10 COMPLETE")
print("=" * 75)

print("\nTest performance:")
print(
    f"R²   = {r2:.4f}"
)

print(
    f"RMSE = {rmse:.4f}"
)

print(
    f"MAE  = {mae:.4f}"
)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        f"  {filename}"
    )

print("\n" + "=" * 75)