import numpy as np
import pandas as pd

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "CHEMBL203_train.csv"
VALIDATION_FILE = "CHEMBL203_validation.csv"
TEST_FILE = "CHEMBL203_test.csv"

RANDOM_SEED = 42

N_SPLITS = 5

FP_SIZE = 2048
RADIUS = 2


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

PARAM_GRID = {

    "n_estimators": [
        200,
        400,
        600
    ],

    "max_features": [
        "sqrt",
        "log2",
        0.3,
        0.5
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_depth": [
        None,
        20,
        40
    ]
}


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
# GENERATE MORGAN FINGERPRINTS
# ============================================================

def generate_fingerprints(
    df,
    generator
):

    fingerprints = []
    valid_indices = []

    for index, smiles in enumerate(
        df["canonical_smiles"]
    ):

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            continue

        fp = generator.GetFingerprint(
            mol
        )

        arr = np.zeros(
            (FP_SIZE,),
            dtype=np.uint8
        )

        # RDKit ExplicitBitVect -> NumPy
        from rdkit import DataStructs

        DataStructs.ConvertToNumpyArray(
            fp,
            arr
        )

        fingerprints.append(
            arr
        )

        valid_indices.append(
            index
        )

    X = np.asarray(
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

def calculate_metrics(
    y_true,
    y_pred
):

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
    print("EXTRA TREES HYPERPARAMETER OPTIMIZATION")
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
        f"Morgan fingerprint:"
        f" radius={RADIUS},"
        f" bits={FP_SIZE}"
    )

    # --------------------------------------------------------
    # Generate fingerprints
    # --------------------------------------------------------

    print()
    print(
        "Generating training fingerprints..."
    )

    X_train, y_train = generate_fingerprints(
        train_df,
        generator
    )

    print(
        f"Training matrix: {X_train.shape}"
    )

    print()
    print(
        "Generating validation fingerprints..."
    )

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
    print(
        "Generating test fingerprints..."
    )

    X_test, y_test = generate_fingerprints(
        test_df,
        generator
    )

    print(
        f"Test matrix: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Cross-validation setup
    # --------------------------------------------------------

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    parameter_combinations = list(
        ParameterGrid(
            PARAM_GRID
        )
    )

    print()
    print("=" * 75)
    print("HYPERPARAMETER SEARCH")
    print("=" * 75)

    print()
    print(
        f"Parameter combinations: "
        f"{len(parameter_combinations)}"
    )

    print(
        f"Cross-validation folds: "
        f"{N_SPLITS}"
    )

    print(
        f"Total model fits: "
        f"{len(parameter_combinations) * N_SPLITS}"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search_results = []

    best_rmse = float("inf")
    best_params = None

    for combination_number, params in enumerate(
        parameter_combinations,
        start=1
    ):

        print()
        print(
            f"[{combination_number}/"
            f"{len(parameter_combinations)}]"
        )

        print(
            f"Parameters: {params}"
        )

        fold_r2 = []
        fold_rmse = []
        fold_mae = []

        for fold_number, (
            train_indices,
            validation_indices
        ) in enumerate(
            kfold.split(X_train),
            start=1
        ):

            model = ExtraTreesRegressor(
                n_estimators=params[
                    "n_estimators"
                ],

                max_features=params[
                    "max_features"
                ],

                min_samples_leaf=params[
                    "min_samples_leaf"
                ],

                max_depth=params[
                    "max_depth"
                ],

                random_state=(
                    RANDOM_SEED
                    + combination_number
                    + fold_number
                ),

                n_jobs=-1
            )

            model.fit(
                X_train[
                    train_indices
                ],
                y_train[
                    train_indices
                ]
            )

            predictions = model.predict(
                X_train[
                    validation_indices
                ]
            )

            r2, rmse, mae = (
                calculate_metrics(
                    y_train[
                        validation_indices
                    ],
                    predictions
                )
            )

            fold_r2.append(
                r2
            )

            fold_rmse.append(
                rmse
            )

            fold_mae.append(
                mae
            )

        mean_r2 = np.mean(
            fold_r2
        )

        std_r2 = np.std(
            fold_r2
        )

        mean_rmse = np.mean(
            fold_rmse
        )

        std_rmse = np.std(
            fold_rmse
        )

        mean_mae = np.mean(
            fold_mae
        )

        std_mae = np.std(
            fold_mae
        )

        result = {

            "combination": combination_number,

            "n_estimators":
                params["n_estimators"],

            "max_features":
                str(params["max_features"]),

            "min_samples_leaf":
                params["min_samples_leaf"],

            "max_depth":
                str(params["max_depth"]),

            "mean_R2":
                mean_r2,

            "std_R2":
                std_r2,

            "mean_RMSE":
                mean_rmse,

            "std_RMSE":
                std_rmse,

            "mean_MAE":
                mean_mae,

            "std_MAE":
                std_mae
        }

        search_results.append(
            result
        )

        print(
            f"Mean CV R²: "
            f"{mean_r2:.4f}"
        )

        print(
            f"Mean CV RMSE: "
            f"{mean_rmse:.4f}"
        )

        print(
            f"Mean CV MAE: "
            f"{mean_mae:.4f}"
        )

        # Lower RMSE is better
        if mean_rmse < best_rmse:

            best_rmse = mean_rmse

            best_params = params.copy()

    # --------------------------------------------------------
    # Search results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        search_results
    )

    results_df = results_df.sort_values(
        by="mean_RMSE",
        ascending=True
    ).reset_index(
        drop=True
    )

    print()
    print("=" * 75)
    print("TOP HYPERPARAMETER CONFIGURATIONS")
    print("=" * 75)

    print(
        results_df.head(
            10
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best parameters
    # --------------------------------------------------------

    best_row = results_df.iloc[
        0
    ]

    best_params = {

        "n_estimators":
            int(best_row["n_estimators"]),

        "max_features":
            (
                float(best_row["max_features"])
                if best_row["max_features"]
                not in ["sqrt", "log2"]
                else best_row["max_features"]
            ),

        "min_samples_leaf":
            int(best_row["min_samples_leaf"]),

        "max_depth":
            (
                None
                if best_row["max_depth"] == "None"
                else int(best_row["max_depth"])
            )
    }

    print()
    print("=" * 75)
    print("BEST PARAMETERS")
    print("=" * 75)

    for key, value in best_params.items():

        print(
            f"{key}: {value}"
        )

    print()
    print(
        f"Best CV RMSE: "
        f"{best_row['mean_RMSE']:.4f}"
    )

    print(
        f"Best CV R²: "
        f"{best_row['mean_R2']:.4f}"
    )

    print(
        f"Best CV MAE: "
        f"{best_row['mean_MAE']:.4f}"
    )

    # --------------------------------------------------------
    # Train optimized model on training set
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("OPTIMIZED MODEL — VALIDATION")
    print("=" * 75)

    optimized_model = ExtraTreesRegressor(
        n_estimators=best_params[
            "n_estimators"
        ],

        max_features=best_params[
            "max_features"
        ],

        min_samples_leaf=best_params[
            "min_samples_leaf"
        ],

        max_depth=best_params[
            "max_depth"
        ],

        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    optimized_model.fit(
        X_train,
        y_train
    )

    validation_predictions = (
        optimized_model.predict(
            X_validation
        )
    )

    validation_r2, validation_rmse, validation_mae = (
        calculate_metrics(
            y_validation,
            validation_predictions
        )
    )

    print()
    print(
        f"Validation R²   : "
        f"{validation_r2:.4f}"
    )

    print(
        f"Validation RMSE : "
        f"{validation_rmse:.4f}"
    )

    print(
        f"Validation MAE  : "
        f"{validation_mae:.4f}"
    )

    # --------------------------------------------------------
    # Refit on training + validation
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL MODEL — TRAIN + VALIDATION")
    print("=" * 75)

    X_train_validation = np.vstack([
        X_train,
        X_validation
    ])

    y_train_validation = np.concatenate([
        y_train,
        y_validation
    ])

    final_model = ExtraTreesRegressor(
        n_estimators=best_params[
            "n_estimators"
        ],

        max_features=best_params[
            "max_features"
        ],

        min_samples_leaf=best_params[
            "min_samples_leaf"
        ],

        max_depth=best_params[
            "max_depth"
        ],

        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    final_model.fit(
        X_train_validation,
        y_train_validation
    )

    # --------------------------------------------------------
    # Final untouched test
    # --------------------------------------------------------

    test_predictions = (
        final_model.predict(
            X_test
        )
    )

    test_r2, test_rmse, test_mae = (
        calculate_metrics(
            y_test,
            test_predictions
        )
    )

    print()
    print("=" * 75)
    print("FINAL UNTOUCHED TEST PERFORMANCE")
    print("=" * 75)

    print()
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
    # Compare with previous baseline
    # --------------------------------------------------------

    baseline_test_r2 = 0.6622
    baseline_test_rmse = 0.7925
    baseline_test_mae = 0.6106

    print()
    print("=" * 75)
    print("BASELINE vs OPTIMIZED")
    print("=" * 75)

    print()
    print(
        f"{'Metric':<12}"
        f"{'Baseline':>15}"
        f"{'Optimized':>15}"
    )

    print(
        "-" * 42
    )

    print(
        f"{'R²':<12}"
        f"{baseline_test_r2:>15.4f}"
        f"{test_r2:>15.4f}"
    )

    print(
        f"{'RMSE':<12}"
        f"{baseline_test_rmse:>15.4f}"
        f"{test_rmse:>15.4f}"
    )

    print(
        f"{'MAE':<12}"
        f"{baseline_test_mae:>15.4f}"
        f"{test_mae:>15.4f}"
    )

    # --------------------------------------------------------
    # Save search results
    # --------------------------------------------------------

    results_df.to_csv(
        "hyperparameter_search_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save final summary
    # --------------------------------------------------------

    summary = pd.DataFrame([{

        "model":
            "Extra Trees Optimized",

        "fingerprint":
            "Morgan",

        "radius":
            RADIUS,

        "fp_size":
            FP_SIZE,

        "CV_R2":
            best_row["mean_R2"],

        "CV_R2_STD":
            best_row["std_R2"],

        "CV_RMSE":
            best_row["mean_RMSE"],

        "CV_RMSE_STD":
            best_row["std_RMSE"],

        "CV_MAE":
            best_row["mean_MAE"],

        "CV_MAE_STD":
            best_row["std_MAE"],

        "validation_R2":
            validation_r2,

        "validation_RMSE":
            validation_rmse,

        "validation_MAE":
            validation_mae,

        "test_R2":
            test_r2,

        "test_RMSE":
            test_rmse,

        "test_MAE":
            test_mae,

        "n_estimators":
            best_params["n_estimators"],

        "max_features":
            str(best_params["max_features"]),

        "min_samples_leaf":
            best_params["min_samples_leaf"],

        "max_depth":
            str(best_params["max_depth"]),

        "random_seed":
            RANDOM_SEED

    }])

    summary.to_csv(
        "optimized_qsar_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    test_output = test_df.iloc[
        :len(test_predictions)
    ].copy()

    test_output[
        "predicted_pIC50"
    ] = test_predictions

    test_output[
        "prediction_error"
    ] = (
        y_test -
        test_predictions
    )

    test_output[
        "absolute_error"
    ] = np.abs(
        y_test -
        test_predictions
    )

    test_output.to_csv(
        "optimized_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 75)

    print()
    print("Saved:")

    print(
        "  hyperparameter_search_results.csv"
    )

    print(
        "  optimized_qsar_summary.csv"
    )

    print(
        "  optimized_test_predictions.csv"
    )


if __name__ == "__main__":
    main()