import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "CHEMBL203_train.csv"
VALIDATION_FILE = "CHEMBL203_validation.csv"
TEST_FILE = "CHEMBL203_test.csv"

RANDOM_SEED = 42

FP_SIZE = 2048
RADIUS = 2

N_ESTIMATORS = 300

# Applicability-domain thresholds
# These are initial operational thresholds.
# They should be calibrated further using validation data.
IN_DOMAIN_THRESHOLD = 0.70
BORDERLINE_THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("APPLICABILITY DOMAIN + UNCERTAINTY ANALYSIS")
print("=" * 75)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print()
print("Dataset sizes:")
print(f"Training:   {len(train_df)}")
print(f"Validation: {len(validation_df)}")
print(f"Test:       {len(test_df)}")


# ============================================================
# MORGAN FINGERPRINT GENERATOR
# ============================================================

generator = (
    rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE
    )
)


# ============================================================
# FINGERPRINT FUNCTION
# ============================================================

def generate_fingerprints(df):

    fingerprints = []
    valid_indices = []

    for i, smiles in enumerate(
        df["canonical_smiles"]
    ):

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        fp = generator.GetFingerprint(mol)

        fingerprints.append(fp)

        valid_indices.append(i)

    return fingerprints, valid_indices


# ============================================================
# GENERATE FINGERPRINTS
# ============================================================

print()
print("Generating fingerprints...")

train_fps, train_valid_indices = (
    generate_fingerprints(train_df)
)

validation_fps, validation_valid_indices = (
    generate_fingerprints(validation_df)
)

test_fps, test_valid_indices = (
    generate_fingerprints(test_df)
)

print(
    f"Training fingerprints:   {len(train_fps)}"
)

print(
    f"Validation fingerprints: {len(validation_fps)}"
)

print(
    f"Test fingerprints:       {len(test_fps)}"
)


# ============================================================
# NUMPY MATRICES FOR MODEL
# ============================================================

def fingerprints_to_numpy(fps):

    matrix = []

    for fp in fps:

        arr = np.zeros(
            (FP_SIZE,),
            dtype=np.uint8
        )

        DataStructs.ConvertToNumpyArray(
            fp,
            arr
        )

        matrix.append(arr)

    return np.array(
        matrix,
        dtype=np.uint8
    )


X_train = fingerprints_to_numpy(
    train_fps
)

X_validation = fingerprints_to_numpy(
    validation_fps
)

X_test = fingerprints_to_numpy(
    test_fps
)

y_train = train_df.iloc[
    train_valid_indices
]["pIC50"].values

y_validation = validation_df.iloc[
    validation_valid_indices
]["pIC50"].values

y_test = test_df.iloc[
    test_valid_indices
]["pIC50"].values


# ============================================================
# TRAIN EXTRA TREES ENSEMBLE
# ============================================================

print()
print("=" * 75)
print("TRAINING EXTRA TREES ENSEMBLE")
print("=" * 75)

model = ExtraTreesRegressor(
    n_estimators=N_ESTIMATORS,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print()
print(
    f"Trees trained: {N_ESTIMATORS}"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_predictions = model.predict(
    X_test
)


# ============================================================
# MODEL METRICS
# ============================================================

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

print()
print("=" * 75)
print("TEST PERFORMANCE")
print("=" * 75)

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
# ENSEMBLE UNCERTAINTY
# ============================================================

print()
print("Calculating prediction uncertainty...")

tree_predictions = np.array([

    tree.predict(X_test)

    for tree in model.estimators_

])

prediction_mean = np.mean(
    tree_predictions,
    axis=0
)

prediction_std = np.std(
    tree_predictions,
    axis=0
)


# ============================================================
# TANI­MOTO SIMILARITY
# ============================================================

print()
print("Calculating applicability-domain similarity...")


def calculate_max_similarity(
    query_fp,
    reference_fps
):

    similarities = (
        DataStructs.BulkTanimotoSimilarity(
            query_fp,
            reference_fps
        )
    )

    return max(similarities)


max_similarities = []

mean_top5_similarities = []


for i, query_fp in enumerate(
    test_fps
):

    similarities = (
        DataStructs.BulkTanimotoSimilarity(
            query_fp,
            train_fps
        )
    )

    similarities = np.array(
        similarities
    )

    similarities.sort()

    similarities = similarities[::-1]

    max_similarity = similarities[0]

    top5_count = min(
        5,
        len(similarities)
    )

    mean_top5 = np.mean(
        similarities[:top5_count]
    )

    max_similarities.append(
        max_similarity
    )

    mean_top5_similarities.append(
        mean_top5
    )


max_similarities = np.array(
    max_similarities
)

mean_top5_similarities = np.array(
    mean_top5_similarities
)


# ============================================================
# APPLICABILITY-DOMAIN CLASSIFICATION
# ============================================================

def classify_domain(
    similarity
):

    if similarity >= IN_DOMAIN_THRESHOLD:

        return "IN_DOMAIN"

    elif similarity >= BORDERLINE_THRESHOLD:

        return "BORDERLINE"

    else:

        return "OUT_OF_DOMAIN"


domain_status = [

    classify_domain(similarity)

    for similarity in max_similarities

]


# ============================================================
# ABSOLUTE ERRORS
# ============================================================

absolute_error = np.abs(
    y_test - prediction_mean
)


# ============================================================
# RESULTS TABLE
# ============================================================

results = test_df.iloc[
    test_valid_indices
].copy()

results.reset_index(
    drop=True,
    inplace=True
)

results["observed_pIC50"] = y_test

results["predicted_pIC50"] = (
    prediction_mean
)

results["prediction_std"] = (
    prediction_std
)

results["absolute_error"] = (
    absolute_error
)

results["max_tanimoto_similarity"] = (
    max_similarities
)

results["mean_top5_similarity"] = (
    mean_top5_similarities
)

results["applicability_domain"] = (
    domain_status
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 75)
print("APPLICABILITY DOMAIN SUMMARY")
print("=" * 75)

domain_counts = (
    results[
        "applicability_domain"
    ]
    .value_counts()
)

print()

print(
    domain_counts.to_string()
)


print()
print(
    f"Mean maximum similarity: "
    f"{max_similarities.mean():.4f}"
)

print(
    f"Minimum maximum similarity: "
    f"{max_similarities.min():.4f}"
)

print(
    f"Maximum maximum similarity: "
    f"{max_similarities.max():.4f}"
)


print()
print(
    f"Mean prediction uncertainty: "
    f"{prediction_std.mean():.4f}"
)

print(
    f"Maximum prediction uncertainty: "
    f"{prediction_std.max():.4f}"
)


# ============================================================
# ERROR BY DOMAIN
# ============================================================

print()
print("=" * 75)
print("ERROR BY APPLICABILITY DOMAIN")
print("=" * 75)

for domain in [
    "IN_DOMAIN",
    "BORDERLINE",
    "OUT_OF_DOMAIN"
]:

    subset = results[
        results[
            "applicability_domain"
        ] == domain
    ]

    if len(subset) == 0:

        print()
        print(
            f"{domain}: no molecules"
        )

        continue

    domain_mae = subset[
        "absolute_error"
    ].mean()

    domain_rmse = np.sqrt(
        np.mean(
            subset[
                "absolute_error"
            ] ** 2
        )
    )

    print()

    print(
        f"{domain}:"
    )

    print(
        f"  Molecules: {len(subset)}"
    )

    print(
        f"  MAE: {domain_mae:.4f}"
    )

    print(
        f"  RMSE: {domain_rmse:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    "applicability_domain_results.csv",
    index=False
)


summary = pd.DataFrame([{

    "test_R2": r2,
    "test_RMSE": rmse,
    "test_MAE": mae,

    "mean_max_similarity":
        max_similarities.mean(),

    "minimum_max_similarity":
        max_similarities.min(),

    "maximum_max_similarity":
        max_similarities.max(),

    "mean_prediction_std":
        prediction_std.mean(),

    "maximum_prediction_std":
        prediction_std.max(),

    "in_domain":
        domain_counts.get(
            "IN_DOMAIN",
            0
        ),

    "borderline":
        domain_counts.get(
            "BORDERLINE",
            0
        ),

    "out_of_domain":
        domain_counts.get(
            "OUT_OF_DOMAIN",
            0
        )

}])


summary.to_csv(
    "applicability_domain_summary.csv",
    index=False
)


print()
print("=" * 75)
print("APPLICABILITY DOMAIN ANALYSIS COMPLETE")
print("=" * 75)

print()
print("Saved:")
print(
    "  applicability_domain_results.csv"
)

print(
    "  applicability_domain_summary.csv"
)