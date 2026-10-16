from pathlib import Path
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D


ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = ROOT_DIR / "models_deploy" / "legacy" / "step13_qsar_model.pkl"
if not MODEL_FILE.exists():
    MODEL_FILE = ROOT_DIR / "step13_qsar_model.pkl"
TARGET_REGISTRY_FILE = ROOT_DIR / "models" / "target_registry.json"
MULTITARGET_MANIFEST_FILE = ROOT_DIR / "models" / "multitarget" / "active_candidate.json"
TRAIN_FILE = ROOT_DIR / "CHEMBL203_train.csv"
VALIDATION_FILE = ROOT_DIR / "CHEMBL203_validation.csv"

AD_THRESHOLD = 0.60
UNCERTAINTY_THRESHOLD = 0.8988

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# ============================================================
# MORGAN FINGERPRINT GENERATOR
# ============================================================

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}\n"
        "Run 'python 13_predict.py' once first."
    )

model = joblib.load(MODEL_FILE)


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

reference_data = pd.concat(
    [train, validation],
    ignore_index=True
)


# ============================================================
# BUILD REFERENCE FINGERPRINTS
# ============================================================

reference_fps = []

for smiles in reference_data["canonical_smiles"]:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    fp = morgan_generator.GetFingerprint(mol)

    reference_fps.append(fp)


_candidate_cache = {}
_multitarget_cache = None

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = [first + second for first in AMINO_ACIDS for second in AMINO_ACIDS]
HYDROPHOBIC = set("AILMFWVY")


def load_registered_candidate(target: dict):
    """Load a versioned single-target candidate and its development references."""
    target_id = target["target_id"].upper()
    model_path_value = target.get("model_path")
    if not model_path_value:
        raise FileNotFoundError(f"No model artifact is registered for {target_id}.")
    model_path = (ROOT_DIR / model_path_value).resolve()
    cache_key = (target_id, target.get("model_version"), str(model_path))
    if cache_key in _candidate_cache:
        return _candidate_cache[cache_key]
    if ROOT_DIR.resolve() not in model_path.parents:
        raise ValueError("Registered model path is outside the project directory.")
    model_dir = model_path.parent
    split_path = model_dir / "split_assignments.csv"
    report_value = target.get("evaluation_report")
    report_path = (ROOT_DIR / report_value).resolve() if report_value else model_dir / "evaluation_report.json"
    required_files = (model_path, split_path, report_path)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Registered candidate files are missing: " + ", ".join(missing))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_target = report.get("target_chembl_id", report.get("target_id"))
    if report_target and report_target.upper() != target_id:
        raise ValueError(f"Registered model report does not match {target_id}.")
    if report.get("fingerprint") != {"type": "Morgan", "radius": 2, "bits": 2048}:
        raise ValueError(f"{target_id} candidate fingerprint configuration is unsupported.")
    model = joblib.load(model_path)
    development = pd.read_csv(split_path, usecols=["canonical_smiles", "split"])
    development = development[development["split"].isin(["train", "validation"])]

    fingerprints = []
    for smiles in development["canonical_smiles"].dropna().astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fingerprints.append(morgan_generator.GetFingerprint(mol))
    if not fingerprints:
        raise ValueError(f"{target_id} development-set reference fingerprints are empty.")
    external_path_value = target.get("external_validation_report")
    external_path = (ROOT_DIR / external_path_value).resolve() if external_path_value else model_dir / "external_validation_report.json"
    external_report = json.loads(external_path.read_text(encoding="utf-8")) if external_path.is_file() else None
    result = (model, fingerprints, report, external_report)
    _candidate_cache[cache_key] = result
    return result


# ============================================================
# SMILES VALIDATION
# ============================================================

def validate_smiles(smiles: str):

    if not smiles or not isinstance(smiles, str):
        return None

    return Chem.MolFromSmiles(smiles)


# ============================================================
# FINGERPRINT → NUMPY
# ============================================================

def fingerprint_to_numpy(fp):

    arr = np.zeros(
        MORGAN_BITS,
        dtype=np.uint8
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr
    )

    return arr


def protein_sequence_features(sequence: str):
    """Match the protein feature encoding used by build_multitarget_features.py."""
    normalized = "".join(sequence.split()).upper()
    if len(normalized) < 50 or any(letter not in AMINO_ACIDS for letter in normalized):
        raise ValueError("Protein sequence must contain at least 50 standard amino-acid residues.")
    length = len(normalized)
    amino_acid = [normalized.count(residue) / length for residue in AMINO_ACIDS]
    pair_counts = Counter(normalized[index:index + 2] for index in range(length - 1))
    pair_denominator = max(length - 1, 1)
    dipeptide = [pair_counts.get(pair, 0) / pair_denominator for pair in DIPEPTIDES]
    hydrophobic = sum(residue in HYDROPHOBIC for residue in normalized) / length
    return np.asarray(amino_acid + dipeptide + [length / 1000.0, hydrophobic], dtype=np.float32)


def load_multitarget_candidate():
    global _multitarget_cache
    if _multitarget_cache is not None:
        return _multitarget_cache
    if not MULTITARGET_MANIFEST_FILE.is_file():
        raise FileNotFoundError("No multi-target candidate has been trained and registered yet.")
    manifest = json.loads(MULTITARGET_MANIFEST_FILE.read_text(encoding="utf-8"))
    if not manifest.get("research_prediction_enabled"):
        raise ValueError("The multi-target candidate is registered but research predictions are disabled.")
    model_path = (ROOT_DIR / manifest["model_path"]).resolve()
    if ROOT_DIR.resolve() not in model_path.parents or not model_path.is_file():
        raise FileNotFoundError("Registered multi-target model artifact is unavailable.")
    report_path = (ROOT_DIR / manifest["evaluation_report"]).resolve()
    if ROOT_DIR.resolve() not in report_path.parents or not report_path.is_file():
        raise FileNotFoundError("Registered multi-target evaluation report is unavailable.")
    model = joblib.load(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _multitarget_cache = (model, report, manifest)
    return _multitarget_cache


# ============================================================
# MODEL UNCERTAINTY
# ============================================================

def predict_with_uncertainty(X):

    predictions = []

    for estimator in model.estimators_:

        predictions.append(
            estimator.predict(X)
        )

    predictions = np.asarray(predictions)

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
# TANIMOTO SIMILARITY
# ============================================================

def max_tanimoto_similarity(fp, fingerprints=None):

    comparison_fps = reference_fps if fingerprints is None else fingerprints
    if not comparison_fps:
        return 0.0

    similarities = DataStructs.BulkTanimotoSimilarity(
        fp,
        comparison_fps
    )

    return float(max(similarities))


# ============================================================
# pIC50 → IC50 nM
# ============================================================

def pic50_to_ic50_nm(pic50):

    return float(
        (10 ** (-pic50)) * 1e9
    )


# ============================================================
# APPLICABILITY + CONFIDENCE
# ============================================================

def classify_prediction(
    similarity,
    uncertainty
):

    in_domain = similarity >= AD_THRESHOLD

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

        uncertainty_status = (
            "LOW_UNCERTAINTY"
            if low_uncertainty
            else "HIGH_UNCERTAINTY"
        )

        return (
            "OUT_OF_DOMAIN",
            uncertainty_status,
            "OUT_OF_DOMAIN"
        )


# ============================================================
# MOLECULAR STRUCTURE → SVG
# ============================================================

def generate_structure_svg(mol):

    drawer = rdMolDraw2D.MolDraw2DSVG(
        500,
        350
    )

    drawer.DrawMolecule(mol)

    drawer.FinishDrawing()

    return drawer.GetDrawingText()


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict(smiles: str, target_id: str = "CHEMBL203"):

    normalized_target_id = target_id.upper()
    registry = json.loads(TARGET_REGISTRY_FILE.read_text(encoding="utf-8"))
    target = next(
        (row for row in registry.get("targets", []) if row["target_id"].upper() == normalized_target_id),
        None,
    )
    if target is None:
        raise ValueError("No prediction model is registered for this target.")
    if not target.get("prediction_available", False):
        raise ValueError(f"No prediction is enabled for {normalized_target_id}; its model is not ready.")

    # --------------------------------------------------------
    # Validate SMILES
    # --------------------------------------------------------

    mol = validate_smiles(smiles)

    if mol is None:

        raise ValueError(
            "Invalid SMILES structure."
        )


    # --------------------------------------------------------
    # Canonical SMILES
    # --------------------------------------------------------

    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True
    )


    # --------------------------------------------------------
    # Morgan fingerprint
    # --------------------------------------------------------

    fp = morgan_generator.GetFingerprint(
        mol
    )


    X = fingerprint_to_numpy(
        fp
    ).reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if target.get("model_path"):
        candidate_model, candidate_reference_fps, candidate_report, external_report = load_registered_candidate(target)
        predicted_pic50 = float(candidate_model.predict(X)[0])
        comparisons = candidate_report.get("cv_model_comparison", candidate_report.get("model_comparison_cv", []))
        selected_cv = next(
            (row for row in comparisons if row.get("model") == candidate_report.get("selected_model_by_train_cv_rmse")),
            {},
        )
        external_metrics = (
            external_report.get("metrics_on_external_source_nonoverlap", {})
            if external_report
            else {}
        )
        test_metrics = candidate_report.get("scaffold_test_metrics", {})
        test_rmse = test_metrics.get("RMSE_pIC50", test_metrics.get("RMSE"))
        expected_error = external_metrics.get("RMSE_pIC50") or test_rmse or selected_cv.get("CV_RMSE_mean")
        prediction_uncertainty = float(expected_error) if expected_error is not None else 0.0
        similarity = max_tanimoto_similarity(fp, candidate_reference_fps)
        ad_status = "NOT_CALIBRATED"
        uncertainty_status = "NOT_CALIBRATED"
        confidence = "VALIDATED_REVIEW_REQUIRED" if target.get("quality_gates_passed") else "SCREENING_CANDIDATE"
        ad_threshold = None
        uncertainty_threshold = None
        uncertainty_method = (
            "External-source RMSE; dataset-level error estimate, not per-compound SD"
            if external_metrics.get("RMSE_pIC50") is not None
            else "Scaffold-held-out test/CV RMSE; dataset-level error estimate, not per-compound SD"
        )
        selected_name = candidate_report.get("selected_model_by_train_cv_rmse", "registered candidate")
        parameters = candidate_report.get("selected_model_parameters", {})
        algorithm = f"{selected_name} candidate"
        warning = f"{target.get('gene', normalized_target_id)} model is not a universal protein model. "
        if test_rmse is not None:
            test_r2 = test_metrics.get("R2")
            warning += "Scaffold-held-out internal test: "
            if test_r2 is not None:
                warning += f"R2={test_r2:.2f}, "
            warning += f"RMSE={test_rmse:.2f} pIC50. "
        if external_metrics:
            warning += (
                "External-source non-overlap check (n="
                f"{external_metrics['n_compounds']}) had RMSE="
                f"{external_metrics['RMSE_pIC50']:.2f} pIC50. "
            )
        warning += (
            "Applicability domain and per-compound uncertainty are not calibrated for this candidate. "
            "Do not treat this as experimental evidence."
        )
    else:
        prediction, uncertainty = predict_with_uncertainty(X)
        predicted_pic50 = float(prediction[0])
        prediction_uncertainty = float(uncertainty[0])
        similarity = max_tanimoto_similarity(fp)
        (
            ad_status,
            uncertainty_status,
            confidence,
        ) = classify_prediction(similarity, prediction_uncertainty)
        ad_threshold = AD_THRESHOLD
        uncertainty_threshold = UNCERTAINTY_THRESHOLD
        uncertainty_method = "Extra Trees ensemble prediction SD"
        algorithm = "Extra Trees Regressor"
        warning = None


    # --------------------------------------------------------
    # Convert pIC50 → IC50
    # --------------------------------------------------------

    predicted_ic50_nm = pic50_to_ic50_nm(
        predicted_pic50
    )


    # --------------------------------------------------------
    # Applicability Domain
    # --------------------------------------------------------

    if not target.get("model_path"):
        if ad_status == "OUT_OF_DOMAIN":
            warning = (
                "Compound is outside the calibrated applicability domain. "
                "Quantitative prediction should be treated with caution."
            )
        elif uncertainty_status == "HIGH_UNCERTAINTY":
            warning = (
                "Prediction uncertainty is above the calibrated threshold. "
                "Interpret the quantitative prediction with caution."
            )


    # --------------------------------------------------------
    # Molecular structure SVG
    # --------------------------------------------------------

    structure_svg = generate_structure_svg(
        mol
    )


    # ========================================================
    # FINAL API RESPONSE
    # ========================================================

    return {

        "target": {
            key: target.get(key)
            for key in [
                "target_id", "name", "gene", "organism", "endpoint",
                "activity_unit", "model_version", "prediction_mode", "model_status",
            ]
        },


        "compound": {

            "input_smiles": smiles,

            "canonical_smiles": canonical_smiles,

            "structure_svg": structure_svg
        },


        "prediction": {

            "pIC50": round(
                predicted_pic50,
                4
            ),

            "IC50_nM": round(
                predicted_ic50_nm,
                4
            )
        },


        "uncertainty": {

            "prediction_sd": round(
                prediction_uncertainty,
                4
            ),

            "threshold": (
                uncertainty_threshold
            ),

            "status": uncertainty_status,
            "method": uncertainty_method,
        },


        "applicability_domain": {

            "max_tanimoto_similarity": round(
                similarity,
                4
            ),

            "threshold": ad_threshold,

            "status": ad_status
        },


        "confidence": confidence,


        "warning": warning,


        "model": {

            "algorithm": algorithm,

            "version": target.get("model_version"),

            "fingerprint": "Morgan",

            "radius": MORGAN_RADIUS,

            "bits": MORGAN_BITS
        }
    }


def predict_general(smiles: str, protein_sequence: str):
    """Research-only prediction for an arbitrary standard amino-acid sequence.

    The response deliberately carries no calibrated applicability-domain or
    per-compound uncertainty claim. Target-specific registered models remain
    the preferred prediction route.
    """
    molecule = validate_smiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES structure.")
    model, report, manifest = load_multitarget_candidate()
    fp = morgan_generator.GetFingerprint(molecule)
    chemical = fingerprint_to_numpy(fp).astype(np.float32)
    descriptors = np.asarray(
        [
            Descriptors.MolWt(molecule),
            Descriptors.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumHAcceptors(molecule),
            Descriptors.NumRotatableBonds(molecule),
            Chem.GetFormalCharge(molecule),
        ],
        dtype=np.float32,
    )
    sequence_vector = protein_sequence_features(protein_sequence)
    X = np.concatenate([chemical, descriptors, sequence_vector]).reshape(1, -1)
    expected_features = getattr(model, "n_features_in_", X.shape[1])
    if X.shape[1] != expected_features:
        raise ValueError("Multi-target model feature schema does not match this predictor version.")
    predicted_pic50 = float(model.predict(X)[0])
    holdout = report.get("leave_target_out_macro_metrics") or {}
    expected_error = holdout.get("RMSE_pIC50")
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
    return {
        "target": {
            "target_id": "CUSTOM_PROTEIN_SEQUENCE",
            "name": "Unregistered protein sequence",
            "gene": None,
            "organism": None,
            "endpoint": "IC50",
            "activity_unit": "nM",
            "model_version": manifest.get("model_version"),
            "prediction_mode": "research",
            "model_status": manifest.get("status"),
            "training_target_count": len(report.get("target_ids", [])),
        },
        "compound": {
            "input_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "structure_svg": generate_structure_svg(molecule),
        },
        "prediction": {
            "pIC50": round(predicted_pic50, 4),
            "IC50_nM": round(pic50_to_ic50_nm(predicted_pic50), 4),
        },
        "uncertainty": {
            "prediction_sd": round(float(expected_error), 4) if expected_error is not None else None,
            "threshold": None,
            "status": "NOT_CALIBRATED",
            "method": "Leave-target-out macro RMSE; dataset-level error estimate, not per-compound SD",
        },
        "applicability_domain": {
            "max_tanimoto_similarity": None,
            "threshold": None,
            "status": "NOT_CALIBRATED_FOR_UNSEEN_PROTEIN",
        },
        "confidence": "RESEARCH_ONLY",
        "warning": (
            f"Unregistered protein sequence prediction uses a {len(report.get('target_ids', []))}-target research model. "
            "It is not a validated target-specific assay result and must be confirmed experimentally."
        ),
        "model": {
            "algorithm": report.get("selected_model_by_train_cv_rmse"),
            "version": manifest.get("model_version"),
            "fingerprint": "Morgan + RDKit descriptors + amino-acid/dipeptide composition",
            "radius": MORGAN_RADIUS,
            "bits": MORGAN_BITS,
        },
    }
# ============================================================
# BATCH PREDICTION
# ============================================================

import io
import pandas as pd
from rdkit import Chem


def predict_batch(file_bytes, target_id: str = "CHEMBL203"):
    """
    Run QSAR predictions for a CSV containing a 'smiles' column.

    Returns a dictionary containing:
        - total compounds
        - successful predictions
        - failed predictions
        - prediction results
    """

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(
            f"Unable to read CSV file: {str(e)}"
        )

    if df.empty:
        raise ValueError("CSV file contains no rows.")
    if len(df) > 1000:
        raise ValueError("CSV is limited to 1,000 compounds per request.")

    # --------------------------------------------------------
    # Find SMILES column
    # --------------------------------------------------------

    smiles_column = None

    for column in df.columns:
        if column.strip().lower() in {
            "smiles",
            "canonical_smiles",
            "structure",
            "molecule_smiles"
        }:
            smiles_column = column
            break

    if smiles_column is None:
        raise ValueError(
            "CSV must contain a 'smiles' column."
        )

    results = []

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process compounds
    # --------------------------------------------------------

    for index, value in enumerate(df[smiles_column]):

        row_number = index + 1

        if pd.isna(value):
            results.append({
                "row": row_number,
                "smiles": "",
                "status": "FAILED",
                "error": "Missing SMILES"
            })

            failed += 1
            continue

        smiles = str(value).strip()

        if not smiles:
            results.append({
                "row": row_number,
                "smiles": "",
                "status": "FAILED",
                "error": "Empty SMILES"
            })

            failed += 1
            continue

        # Validate SMILES before prediction
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            results.append({
                "row": row_number,
                "smiles": smiles,
                "status": "FAILED",
                "error": "Invalid SMILES"
            })

            failed += 1
            continue

        try:
            prediction = predict(smiles, target_id=target_id)

            results.append({
                "row": row_number,
                "target_id": target_id,
                "smiles": smiles,
                "canonical_smiles":
                    prediction["compound"]["canonical_smiles"],

                "pIC50":
                    prediction["prediction"]["pIC50"],

                "IC50_nM":
                    prediction["prediction"]["IC50_nM"],

                "uncertainty":
                    prediction["uncertainty"]["prediction_sd"],

                "uncertainty_status":
                    prediction["uncertainty"]["status"],

                "uncertainty_method":
                    prediction["uncertainty"]["method"],

                "max_tanimoto_similarity":
                    prediction[
                        "applicability_domain"
                    ]["max_tanimoto_similarity"],

                "applicability_domain":
                    prediction[
                        "applicability_domain"
                    ]["status"],

                "confidence":
                    prediction["confidence"],

                "status": "SUCCESS",
                "error": ""
            })

            successful += 1

        except Exception as e:

            results.append({
                "row": row_number,
                "smiles": smiles,
                "status": "FAILED",
                "error": str(e)
            })

            failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    return {
        "total_compounds": len(df),
        "successful_predictions": successful,
        "failed_predictions": failed,
        "results": results
    }
