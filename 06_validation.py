import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "CHEMBL203_train.csv"

RANDOM_SEED = 42
N_SPLITS = 5

FP_SIZE = 2048
RADIUS = 2

N_PERMUTATIONS = 20


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 75)
print("RIGOROUS QSAR VALIDATION")
print("=" * 75)

print()
print(f"Training molecules: {len(df)}")


# ============================================================
# MORGAN FINGERPRINTS
# ============================================================

generator = (
    rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE
    )
)

fingerprints = []
valid_indices = []

for i, smiles in enumerate(
    df["canonical_smiles"]
):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    fp = generator.GetFingerprint(mol)

    fingerprints.append(
        np.asarray(fp)
    )

    valid_indices.append(i)


X = np.array(
    fingerprints,
    dtype=np.uint8
)

y = df.iloc[
    valid_indices
]["pIC50"].values


print()
print(f"Fingerprint matrix: {X.shape}")


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================

print()
print("=" * 75)
print("5-FOLD CROSS-VALIDATION")
print("=" * 75)

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)

model = ExtraTreesRegressor(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

cv_predictions = cross_val_predict(
    model,
    X,
    y,
    cv=kf,
    n_jobs=1
)

cv_r2 = r2_score(
    y,
    cv_predictions
)

cv_rmse = np.sqrt(
    mean_squared_error(
        y,
        cv_predictions
    )
)

cv_mae = mean_absolute_error(
    y,
    cv_predictions
)

print()
print(f"CV R²   : {cv_r2:.4f}")
print(f"CV RMSE : {cv_rmse:.4f}")
print(f"CV MAE  : {cv_mae:.4f}")


# ============================================================
# Y-RANDOMIZATION
# ============================================================

print()
print("=" * 75)
print("Y-RANDOMIZATION TEST")
print("=" * 75)

rng = np.random.default_rng(
    RANDOM_SEED
)

permutation_results = []

for i in range(
    N_PERMUTATIONS
):

    print(
        f"Permutation "
        f"{i + 1}/{N_PERMUTATIONS}"
    )

    y_random = rng.permutation(y)

    random_model = ExtraTreesRegressor(
        n_estimators=150,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=RANDOM_SEED + i,
        n_jobs=-1
    )

    random_predictions = cross_val_predict(
        random_model,
        X,
        y_random,
        cv=kf,
        n_jobs=1
    )

    random_r2 = r2_score(
        y_random,
        random_predictions
    )

    random_rmse = np.sqrt(
        mean_squared_error(
            y_random,
            random_predictions
        )
    )

    permutation_results.append({

        "permutation": i + 1,
        "R2": random_r2,
        "RMSE": random_rmse

    })


permutation_df = pd.DataFrame(
    permutation_results
)


# ============================================================
# PERMUTATION SUMMARY
# ============================================================

print()
print("=" * 75)
print("Y-RANDOMIZATION SUMMARY")
print("=" * 75)

print(
    permutation_df.to_string(
        index=False
    )
)

print()

print(
    f"Mean randomized R²: "
    f"{permutation_df['R2'].mean():.4f}"
)

print(
    f"Maximum randomized R²: "
    f"{permutation_df['R2'].max():.4f}"
)

print(
    f"Mean randomized RMSE: "
    f"{permutation_df['RMSE'].mean():.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

cv_results = pd.DataFrame([{

    "model": "Extra Trees",
    "folds": N_SPLITS,
    "CV_R2": cv_r2,
    "CV_RMSE": cv_rmse,
    "CV_MAE": cv_mae,
    "fingerprint": "Morgan",
    "radius": RADIUS,
    "fp_size": FP_SIZE,
    "random_seed": RANDOM_SEED

}])

cv_results.to_csv(
    "cross_validation_results.csv",
    index=False
)

permutation_df.to_csv(
    "y_randomization_results.csv",
    index=False
)


print()
print("=" * 75)
print("VALIDATION COMPLETE")
print("=" * 75)

print()
print("Saved:")
print("  cross_validation_results.csv")
print("  y_randomization_results.csv")