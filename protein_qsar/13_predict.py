import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")

# ============================================================
# STEP 13
# QSAR COMMAND-LINE PREDICTION ENGINE
# ============================================================

RANDOM_STATE = 42

# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

TRAIN_FILE = "CHEMBL203_train.csv"
VALIDATION_FILE = "CHEMBL203_validation.csv"

MODEL_FILE = "step13_qsar_model.pkl"

# Calibrated in Step 12
AD_THRESHOLD = 0.60
UNCERTAINTY_THRESHOLD = 0.8988

# ------------------------------------------------------------
# Optimized Extra Trees parameters from Step 08
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

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS
)


# ============================================================
# MOLECULE FUNCTIONS
# ============================================================

def validate_smiles(smiles):
    """Validate SMILES and return RDKit molecule."""

    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)

    return mol


def canonicalize_smiles(mol):
    """Return canonical SMILES."""

    return Chem.MolToSmiles(
        mol,
        canonical=True
    )


def generate_fingerprint(mol):
    """Generate Morgan fingerprint."""

    return morgan_generator.GetFingerprint(mol)


def fingerprint_to_numpy(fp):
    """Convert RDKit fingerprint to numpy array."""

    arr = np.zeros(
        MORGAN_BITS,
        dtype=np.uint8
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr
    )

    return arr


# ============================================================
# MODEL
# ============================================================

def train_model():
    """Train optimized model on train + validation."""

    print("\nLoading training data...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)

    combined = pd.concat(
        [train, validation],
        ignore_index=True
    )

    X = []
    y = []

    invalid = 0

    for _, row in combined.iterrows():

        mol = validate_smiles(
            row["canonical_smiles"]
        )

        if mol is None:
            invalid += 1
            continue

        fp = generate_fingerprint(mol)

        X.append(
            fingerprint_to_numpy(fp)
        )

        y.append(
            float(row["pIC50"])
        )

    X = np.asarray(X)
    y = np.asarray(y)

    print(f"Training compounds : {len(y)}")
    print(f"Invalid molecules  : {invalid}")
    print(f"Fingerprint size   : {X.shape[1]}")

    print("\nTraining optimized Extra Trees model...")

    model = ExtraTreesRegressor(
        **MODEL_PARAMS
    )

    model.fit(X, y)

    print("Model training complete.")

    return model


def load_or_train_model():

    # --------------------------------------------------------
    # If saved model exists, load it
    # --------------------------------------------------------

    if os.path.exists(MODEL_FILE):

        import joblib

        print(
            f"\nLoading saved model: {MODEL_FILE}"
        )

        model = joblib.load(
            MODEL_FILE
        )

        return model

    # --------------------------------------------------------
    # Otherwise train and save
    # --------------------------------------------------------

    model = train_model()

    import joblib

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel saved to: {MODEL_FILE}"
    )

    return model


# ============================================================
# CHEMICAL SPACE
# ============================================================

def load_reference_fingerprints():

    print("\nLoading chemical-space reference...")

    train = pd.read_csv(
        TRAIN_FILE
    )

    validation = pd.read_csv(
        VALIDATION_FILE
    )

    combined = pd.concat(
        [train, validation],
        ignore_index=True
    )

    reference_fps = []

    for smiles in combined[
        "canonical_smiles"
    ]:

        mol = validate_smiles(smiles)

        if mol is None:
            continue

        reference_fps.append(
            generate_fingerprint(mol)
        )

    print(
        f"Reference compounds: "
        f"{len(reference_fps)}"
    )

    return reference_fps


def max_tanimoto_similarity(
    query_fp,
    reference_fps
):

    similarities = DataStructs.BulkTanimotoSimilarity(
        query_fp,
        reference_fps
    )

    return max(similarities)


# ============================================================
# UNCERTAINTY
# ============================================================

def predict_with_uncertainty(
    model,
    X
):

    predictions = []

    for estimator in model.estimators_:

        predictions.append(
            estimator.predict(X)
        )

    predictions = np.asarray(
        predictions
    )

    mean_prediction = np.mean(
        predictions,
        axis=0
    )

    prediction_std = np.std(
        predictions,
        axis=0
    )

    return (
        mean_prediction,
        prediction_std
    )


# ============================================================
# IC50 CONVERSION
# ============================================================

def pic50_to_ic50_nm(pic50):

    # pIC50 = -log10(IC50[M])
    #
    # IC50[M] = 10^(-pIC50)
    #
    # nM = M * 1e9

    return (
        10 ** (-pic50)
    ) * 1e9


# ============================================================
# CONFIDENCE
# ============================================================

def classify_prediction(
    similarity,
    uncertainty
):

    in_domain = (
        similarity >= AD_THRESHOLD
    )

    low_uncertainty = (
        uncertainty <= UNCERTAINTY_THRESHOLD
    )

    if in_domain and low_uncertainty:

        return (
            "IN_DOMAIN",
            "LOW_UNCERTAINTY",
            "HIGH_CONFIDENCE"
        )

    elif in_domain and not low_uncertainty:

        return (
            "IN_DOMAIN",
            "HIGH_UNCERTAINTY",
            "CAUTION_HIGH_UNCERTAINTY"
        )

    else:

        return (
            "OUT_OF_DOMAIN",
            (
                "LOW_UNCERTAINTY"
                if low_uncertainty
                else "HIGH_UNCERTAINTY"
            ),
            "OUT_OF_DOMAIN"
        )


# ============================================================
# SINGLE PREDICTION
# ============================================================

def predict_smiles(
    smiles,
    model,
    reference_fps
):

    mol = validate_smiles(
        smiles
    )

    if mol is None:

        raise ValueError(
            "Invalid SMILES."
        )

    canonical = canonicalize_smiles(
        mol
    )

    fp = generate_fingerprint(
        mol
    )

    X = fingerprint_to_numpy(
        fp
    ).reshape(1, -1)

    prediction, uncertainty = (
        predict_with_uncertainty(
            model,
            X
        )
    )

    predicted_pic50 = float(
        prediction[0]
    )

    uncertainty_value = float(
        uncertainty[0]
    )

    predicted_ic50_nm = (
        pic50_to_ic50_nm(
            predicted_pic50
        )
    )

    similarity = max_tanimoto_similarity(
        fp,
        reference_fps
    )

    ad_status, uncertainty_status, confidence = (
        classify_prediction(
            similarity,
            uncertainty_value
        )
    )

    return {
        "input_smiles": smiles,
        "canonical_smiles": canonical,
        "predicted_pIC50": predicted_pic50,
        "predicted_IC50_nM": predicted_ic50_nm,
        "uncertainty": uncertainty_value,
        "max_tanimoto_similarity": similarity,
        "AD_status": ad_status,
        "uncertainty_status": uncertainty_status,
        "confidence": confidence
    }


# ============================================================
# DISPLAY
# ============================================================

def display_result(result):

    print("\n")
    print("=" * 70)
    print("QSAR PREDICTION RESULT")
    print("=" * 70)

    print(
        f"\nInput SMILES:"
        f"\n  {result['input_smiles']}"
    )

    print(
        f"\nCanonical SMILES:"
        f"\n  {result['canonical_smiles']}"
    )

    print("\nPrediction:")
    print(
        f"  pIC50              : "
        f"{result['predicted_pIC50']:.4f}"
    )

    print(
        f"  IC50               : "
        f"{result['predicted_IC50_nM']:.4f} nM"
    )

    print("\nModel uncertainty:")
    print(
        f"  Prediction SD      : "
        f"{result['uncertainty']:.4f}"
    )

    print("\nChemical-space assessment:")
    print(
        f"  Max Tanimoto       : "
        f"{result['max_tanimoto_similarity']:.4f}"
    )

    print(
        f"  AD status          : "
        f"{result['AD_status']}"
    )

    print(
        f"  Uncertainty status : "
        f"{result['uncertainty_status']}"
    )

    print(
        f"  Final confidence   : "
        f"{result['confidence']}"
    )

    print("\nCalibration thresholds:")
    print(
        f"  AD threshold       : "
        f"{AD_THRESHOLD:.4f}"
    )

    print(
        f"  Uncertainty limit  : "
        f"{UNCERTAINTY_THRESHOLD:.4f}"
    )

    print("\n" + "=" * 70)


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode(
    model,
    reference_fps
):

    print("\n")
    print("=" * 70)
    print("QSAR INTERACTIVE PREDICTION MODE")
    print("=" * 70)

    print(
        "\nEnter a SMILES string."
        "\nType 'exit' to quit."
    )

    while True:

        smiles = input(
            "\nSMILES > "
        ).strip()

        if smiles.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nExiting prediction mode."
            )

            break

        if not smiles:

            continue

        try:

            result = predict_smiles(
                smiles,
                model,
                reference_fps
            )

            display_result(
                result
            )

        except Exception as e:

            print(
                f"\nPrediction error: {e}"
            )


# ============================================================
# CSV BATCH MODE
# ============================================================

def batch_prediction(
    input_file,
    output_file,
    model,
    reference_fps
):

    print(
        f"\nReading input file: "
        f"{input_file}"
    )

    df = pd.read_csv(
        input_file
    )

    if "smiles" not in df.columns:

        raise ValueError(
            "CSV must contain a column named 'smiles'."
        )

    results = []

    print(
        f"Compounds found: "
        f"{len(df)}"
    )

    for index, row in df.iterrows():

        smiles = str(
            row["smiles"]
        ).strip()

        try:

            result = predict_smiles(
                smiles,
                model,
                reference_fps
            )

            result["input_id"] = index + 1

            results.append(
                result
            )

            print(
                f"[{index + 1}/{len(df)}] "
                f"Predicted successfully"
            )

        except Exception as e:

            results.append({
                "input_id": index + 1,
                "input_smiles": smiles,
                "canonical_smiles": "",
                "predicted_pIC50": np.nan,
                "predicted_IC50_nM": np.nan,
                "uncertainty": np.nan,
                "max_tanimoto_similarity": np.nan,
                "AD_status": "INVALID",
                "uncertainty_status": "INVALID",
                "confidence": "INVALID",
                "error": str(e)
            })

            print(
                f"[{index + 1}/{len(df)}] "
                f"FAILED: {e}"
            )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nBatch prediction saved to:"
        f"\n  {output_file}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 13 - EGFR QSAR "
            "Command-Line Prediction Engine"
        )
    )

    parser.add_argument(
        "--smiles",
        type=str,
        help="SMILES string to predict"
    )

    parser.add_argument(
        "--input",
        type=str,
        help=(
            "CSV input file containing "
            "a 'smiles' column"
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="step13_predictions.csv",
        help="Output CSV for batch prediction"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive prediction mode"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_or_train_model()

    # --------------------------------------------------------
    # Load reference chemical space
    # --------------------------------------------------------

    reference_fps = (
        load_reference_fingerprints()
    )

    # --------------------------------------------------------
    # Single SMILES
    # --------------------------------------------------------

    if args.smiles:

        try:

            result = predict_smiles(
                args.smiles,
                model,
                reference_fps
            )

            display_result(
                result
            )

        except Exception as e:

            print(
                f"\nPrediction failed: {e}"
            )

            sys.exit(1)

    # --------------------------------------------------------
    # Batch CSV
    # --------------------------------------------------------

    elif args.input:

        batch_prediction(
            args.input,
            args.output,
            model,
            reference_fps
        )

    # --------------------------------------------------------
    # Interactive
    # --------------------------------------------------------

    else:

        interactive_mode(
            model,
            reference_fps
        )


if __name__ == "__main__":
    main()