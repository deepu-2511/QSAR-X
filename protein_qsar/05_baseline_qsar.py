from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "CHEMBL203_train.csv"
VALIDATION_FILE = "CHEMBL203_validation.csv"
TEST_FILE = "CHEMBL203_test.csv"

RANDOM_SEED = 42

FP_SIZE = 2048
RADIUS = 2


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(filename):

    if not Path(filename).exists():
        raise FileNotFoundError(
            f"File not found: {filename}"
        )

    df = pd.read_csv(filename)

    required_columns = [
        "canonical_smiles",
        "pIC50"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    return df


# ============================================================
# MORGAN FINGERPRINTS
# ============================================================

def generate_fingerprints(df, generator):

    fingerprints = []

    valid_indices = []

    for index, smiles in enumerate(
        df["canonical_smiles"]
    ):

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        fp = generator.GetFingerprint(mol)

        fingerprints.append(
            np.asarray(fp)
        )

        valid_indices.append(index)

    X = np.array(
        fingerprints,
        dtype=np.uint8
    )

    y = df.iloc[
        valid_indices
    ]["pIC50"].values

    return X, y


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    r2 = r2_score(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return r2, rmse, mae


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("BASELINE QSAR BENCHMARK")
    print("=" * 75)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = load_dataset(
        TRAIN_FILE
    )

    validation_df = load_dataset(
        VALIDATION_FILE
    )

    test_df = load_dataset(
        TEST_FILE
    )

    print()
    print("Dataset sizes:")
    print(
        f"Training:   {len(train_df)}"
    )
    print(
        f"Validation: {len(validation_df)}"
    )
    print(
        f"Test:       {len(test_df)}"
    )

    # --------------------------------------------------------
    # Fingerprint generator
    # --------------------------------------------------------

    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=RADIUS,
            fpSize=FP_SIZE
        )
    )

    print()
    print(
        f"Morgan fingerprint: "
        f"radius={RADIUS}, "
        f"bits={FP_SIZE}"
    )

    # --------------------------------------------------------
    # Generate fingerprints
    # --------------------------------------------------------

    print()
    print("Generating training fingerprints...")

    X_train, y_train = generate_fingerprints(
        train_df,
        generator
    )

    print(
        f"Training matrix: "
        f"{X_train.shape}"
    )

    print()
    print("Generating validation fingerprints...")

    X_validation, y_validation = (
        generate_fingerprints(
            validation_df,
            generator
        )
    )

    print(
        f"Validation matrix: "
        f"{X_validation.shape}"
    )

    print()
    print("Generating test fingerprints...")

    X_test, y_test = generate_fingerprints(
        test_df,
        generator
    )

    print(
        f"Test matrix: "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

        "SVR": SVR(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            epsilon=0.1
        )
    }

    # --------------------------------------------------------
    # Validation benchmark
    # --------------------------------------------------------

    validation_results = []

    print()
    print("=" * 75)
    print("VALIDATION BENCHMARK")
    print("=" * 75)

    for name, model in models.items():

        print()
        print(
            f"Training: {name}"
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_validation
        )

        r2, rmse, mae = calculate_metrics(
            y_validation,
            prediction
        )

        print(
            f"R²   : {r2:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"MAE  : {mae:.4f}"
        )

        validation_results.append({

            "model": name,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae
        })

    validation_results_df = pd.DataFrame(
        validation_results
    )

    validation_results_df = (
        validation_results_df
        .sort_values(
            "RMSE",
            ascending=True
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 75)
    print("VALIDATION RESULTS")
    print("=" * 75)

    print(
        validation_results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select model using validation RMSE
    # --------------------------------------------------------

    selected_model_name = (
        validation_results_df.iloc[0]["model"]
    )

    print()
    print(
        f"Selected model based on "
        f"validation RMSE: "
        f"{selected_model_name}"
    )

    selected_model = models[
        selected_model_name
    ]

    # --------------------------------------------------------
    # Refit selected model on TRAIN + VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "Refitting selected model on "
        "training + validation data..."
    )

    X_train_validation = np.vstack([
        X_train,
        X_validation
    ])

    y_train_validation = np.concatenate([
        y_train,
        y_validation
    ])

    selected_model.fit(
        X_train_validation,
        y_train_validation
    )

    # --------------------------------------------------------
    # Final TEST evaluation
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL UNTOUCHED TEST EVALUATION")
    print("=" * 75)

    test_prediction = (
        selected_model.predict(
            X_test
        )
    )

    test_r2, test_rmse, test_mae = (
        calculate_metrics(
            y_test,
            test_prediction
        )
    )

    print()
    print(
        f"Model: {selected_model_name}"
    )

    print(
        f"Test R²   : {test_r2:.4f}"
    )

    print(
        f"Test RMSE : {test_rmse:.4f}"
    )

    print(
        f"Test MAE  : {test_mae:.4f}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    validation_results_df.to_csv(
        "baseline_validation_results.csv",
        index=False
    )

    final_results = pd.DataFrame([{

        "model": selected_model_name,
        "test_R2": test_r2,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "fingerprint": "Morgan",
        "radius": RADIUS,
        "fp_size": FP_SIZE,
        "random_seed": RANDOM_SEED

    }])

    final_results.to_csv(
        "baseline_final_test_results.csv",
        index=False
    )

    print()
    print(
        "Saved:"
    )

    print(
        "  baseline_validation_results.csv"
    )

    print(
        "  baseline_final_test_results.csv"
    )

    print()
    print("=" * 75)
    print("BASELINE QSAR COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()