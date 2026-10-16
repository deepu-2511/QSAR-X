# ============================================================
# QSAR-X Professional
# Dataset Curation Engine
# Curator v2.3.0
# ============================================================
#
# Publication-oriented QSAR dataset curation engine
#
# Supports:
#   - pChEMBL
#   - pIC50
#   - Raw IC50 concentrations
#   - IC50 endpoint filtering
#   - Relation filtering
#   - SMILES validation
#   - Salt standardization
#   - Canonical SMILES
#   - Duplicate aggregation
#   - Molecular descriptors
#   - Quality flags
#   - Rejected-record tracking
#   - CSV/JSON/TXT report generation
#
# IMPORTANT:
# pChEMBL is an activity representation, NOT an endpoint.
#
# ============================================================

import io
import json
import math
from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import SaltRemover


VERSION = "2.3.0"


# ============================================================
# COLUMN ALIASES
# ============================================================

SMILES_ALIASES = [
    "smiles",
    "canonical_smiles",
    "structure",
    "molecule_smiles",
    "canonical smiles",
    "canonical_smiles_string",
]

ACTIVITY_ALIASES = [
    "activity",
    "ic50",
    "ic50_nm",
    "ic50 (nm)",
    "pic50",
    "pIC50",
    "value",
    "standard_value",
    "activity_value",
    "pchembl_value",
    "pchembl value",
    "pchembl",
]

UNIT_ALIASES = [
    "unit",
    "units",
    "activity_unit",
    "standard_units",
]

ENDPOINT_ALIASES = [
    "endpoint",
    "standard_type",
    "activity_type",
    "type",
]

RELATION_ALIASES = [
    "relation",
    "standard_relation",
    "activity_relation",
]


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(name):

    if name is None:
        return ""

    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def find_column(df, aliases):

    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for alias in aliases:

        key = normalize_column_name(alias)

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# SAFE VALUE
# ============================================================

def safe_json_value(value):

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

    return value


# ============================================================
# ACTIVITY MODE
# ============================================================

def detect_activity_mode(activity_column):

    if activity_column is None:
        return "UNKNOWN"

    normalized = normalize_column_name(
        activity_column
    )

    if "pchembl" in normalized:
        return "pCHEMBL"

    if normalized in {
        "pic50",
        "pic50_value",
        "p_ic50",
    }:
        return "pIC50"

    return "CONCENTRATION"


# ============================================================
# UNIT CONVERSION
# ============================================================

def convert_to_pic50(value, unit):

    try:
        value = float(value)

    except Exception:
        return None

    if not math.isfinite(value):
        return None

    if value <= 0:
        return None

    if unit is None:
        return None

    unit = (
        str(unit)
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
    )

    conversion = {

        "m": 1.0,
        "mol": 1.0,
        "mol/l": 1.0,

        "mm": 1e-3,
        "mmol": 1e-3,
        "mmol/l": 1e-3,

        "um": 1e-6,
        "umol": 1e-6,
        "umol/l": 1e-6,

        "nm": 1e-9,
        "nmol": 1e-9,
        "nmol/l": 1e-9,

        "pm": 1e-12,
        "pmol": 1e-12,
        "pmol/l": 1e-12,
    }

    factor = conversion.get(unit)

    if factor is None:
        return None

    molar = value * factor

    if molar <= 0:
        return None

    return -math.log10(molar)


# ============================================================
# SALT REMOVAL
# ============================================================

def remove_salts(mol):

    if mol is None:
        return None, False

    try:

        remover = SaltRemover.SaltRemover()

        cleaned = remover.StripMol(
            mol,
            dontRemoveEverything=True
        )

        original = Chem.MolToSmiles(mol)

        standardized = Chem.MolToSmiles(
            cleaned
        )

        changed = (
            original != standardized
        )

        return cleaned, changed

    except Exception:

        return mol, False


# ============================================================
# SMILES STANDARDIZATION
# ============================================================

def canonicalize_smiles(
    smiles,
    remove_salt=True
):

    if smiles is None:

        return {
            "canonical_smiles": None,
            "status": "MISSING",
            "message": "Missing SMILES",
            "salt_standardized": False,
        }

    try:

        if pd.isna(smiles):

            return {
                "canonical_smiles": None,
                "status": "MISSING",
                "message": "Missing SMILES",
                "salt_standardized": False,
            }

    except Exception:
        pass

    smiles = str(smiles).strip()

    if not smiles:

        return {
            "canonical_smiles": None,
            "status": "MISSING",
            "message": "Missing SMILES",
            "salt_standardized": False,
        }

    try:

        mol = Chem.MolFromSmiles(smiles)

    except Exception:

        mol = None

    if mol is None:

        return {
            "canonical_smiles": None,
            "status": "INVALID",
            "message": "Invalid SMILES",
            "salt_standardized": False,
        }

    salt_changed = False

    if remove_salt:

        mol, salt_changed = remove_salts(
            mol
        )

    if mol is None:

        return {
            "canonical_smiles": None,
            "status": "INVALID",
            "message": "Salt standardization failed",
            "salt_standardized": False,
        }

    try:

        Chem.SanitizeMol(mol)

        canonical = Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except Exception:

        return {
            "canonical_smiles": None,
            "status": "INVALID",
            "message": "RDKit sanitization failed",
            "salt_standardized": salt_changed,
        }

    return {
        "canonical_smiles": canonical,
        "status": "VALID_STANDARDIZED",
        "message": "SMILES standardized successfully",
        "salt_standardized": salt_changed,
    }


# ============================================================
# DESCRIPTORS
# ============================================================

def calculate_basic_descriptors(
    canonical_smiles
):

    if not canonical_smiles:

        return {
            "molecular_weight": None,
            "logp": None,
            "tpsa": None,
            "hbd": None,
            "hba": None,
            "rotatable_bonds": None,
            "rings": None,
        }

    mol = Chem.MolFromSmiles(
        canonical_smiles
    )

    if mol is None:

        return {
            "molecular_weight": None,
            "logp": None,
            "tpsa": None,
            "hbd": None,
            "hba": None,
            "rotatable_bonds": None,
            "rings": None,
        }

    return {

        "molecular_weight": float(
            Descriptors.MolWt(mol)
        ),

        "logp": float(
            Descriptors.MolLogP(mol)
        ),

        "tpsa": float(
            rdMolDescriptors.CalcTPSA(mol)
        ),

        "hbd": int(
            rdMolDescriptors.CalcNumHBD(mol)
        ),

        "hba": int(
            rdMolDescriptors.CalcNumHBA(mol)
        ),

        "rotatable_bonds": int(
            rdMolDescriptors.CalcNumRotatableBonds(mol)
        ),

        "rings": int(
            rdMolDescriptors.CalcNumRings(mol)
        ),
    }


# ============================================================
# ACTIVITY PARSER
# ============================================================

def parse_activity(
    value,
    activity_mode,
    unit=None
):
    """
    Return:
        pIC50,
        status,
        message

    Status:
        VALID
        MISSING_ACTIVITY
        INVALID_ACTIVITY
    """

    # --------------------------------------------------------
    # Missing activity
    # --------------------------------------------------------

    if value is None:

        return (
            None,
            "MISSING_ACTIVITY",
            "Missing activity value"
        )

    try:

        if pd.isna(value):

            return (
                None,
                "MISSING_ACTIVITY",
                "Missing activity value"
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Empty string
    # --------------------------------------------------------

    if isinstance(value, str):

        if not value.strip():

            return (
                None,
                "MISSING_ACTIVITY",
                "Empty activity value"
            )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    try:

        numeric = float(value)

    except Exception:

        return (
            None,
            "INVALID_ACTIVITY",
            f"Non-numeric activity value: {value}"
        )

    if not math.isfinite(numeric):

        return (
            None,
            "INVALID_ACTIVITY",
            "Non-finite activity value"
        )

    # ========================================================
    # pChEMBL
    # ========================================================

    if activity_mode == "pCHEMBL":

        return (
            numeric,
            "VALID",
            "pChEMBL used directly"
        )

    # ========================================================
    # pIC50
    # ========================================================

    if activity_mode == "pIC50":

        return (
            numeric,
            "VALID",
            "pIC50 used directly"
        )

    # ========================================================
    # Raw concentration
    # ========================================================

    if activity_mode == "CONCENTRATION":

        if unit is None:

            pic50 = convert_to_pic50(
                numeric,
                "nM"
            )

            if pic50 is None:

                return (
                    None,
                    "INVALID_ACTIVITY",
                    "Invalid concentration; "
                    "assumed nM conversion failed"
                )

            return (
                pic50,
                "VALID",
                "Missing unit; assumed nM"
            )

        if str(unit).strip() == "":

            pic50 = convert_to_pic50(
                numeric,
                "nM"
            )

            if pic50 is None:

                return (
                    None,
                    "INVALID_ACTIVITY",
                    "Invalid concentration; "
                    "assumed nM conversion failed"
                )

            return (
                pic50,
                "VALID",
                "Missing unit; assumed nM"
            )

        pic50 = convert_to_pic50(
            numeric,
            unit
        )

        if pic50 is None:

            return (
                None,
                "INVALID_ACTIVITY",
                f"Unsupported concentration/unit: "
                f"{unit}"
            )

        return (
            pic50,
            "VALID",
            f"Converted from {unit}"
        )

    return (
        None,
        "INVALID_ACTIVITY",
        f"Unsupported activity mode: "
        f"{activity_mode}"
    )


# ============================================================
# QUALITY STATISTICS
# ============================================================

def calculate_statistics(
    df
):

    result = {}

    # --------------------------------------------------------
    # pIC50
    # --------------------------------------------------------

    if "pIC50" in df.columns:

        values = pd.to_numeric(
            df["pIC50"],
            errors="coerce"
        ).dropna()

        if len(values):

            result[
                "pIC50_statistics"
            ] = {

                "count": int(
                    values.count()
                ),

                "mean": float(
                    values.mean()
                ),

                "median": float(
                    values.median()
                ),

                "std": float(
                    values.std()
                ),

                "min": float(
                    values.min()
                ),

                "max": float(
                    values.max()
                ),

                "q01": float(
                    values.quantile(.01)
                ),

                "q05": float(
                    values.quantile(.05)
                ),

                "q25": float(
                    values.quantile(.25)
                ),

                "q75": float(
                    values.quantile(.75)
                ),

                "q95": float(
                    values.quantile(.95)
                ),

                "q99": float(
                    values.quantile(.99)
                ),
            }

    # --------------------------------------------------------
    # Molecular weight
    # --------------------------------------------------------

    if "molecular_weight" in df.columns:

        values = pd.to_numeric(
            df["molecular_weight"],
            errors="coerce"
        ).dropna()

        if len(values):

            result[
                "molecular_weight_statistics"
            ] = {

                "count": int(
                    values.count()
                ),

                "mean": float(
                    values.mean()
                ),

                "median": float(
                    values.median()
                ),

                "std": float(
                    values.std()
                ),

                "min": float(
                    values.min()
                ),

                "max": float(
                    values.max()
                ),

                "q01": float(
                    values.quantile(.01)
                ),

                "q05": float(
                    values.quantile(.05)
                ),

                "q25": float(
                    values.quantile(.25)
                ),

                "q75": float(
                    values.quantile(.75)
                ),

                "q95": float(
                    values.quantile(.95)
                ),

                "q99": float(
                    values.quantile(.99)
                ),
            }

    return result


# ============================================================
# MAIN CURATION ENGINE
# ============================================================

def curate_dataset(
    file_bytes,
    strict_endpoint=True,
    strict_relation=True,
    remove_salt=True,
    aggregate_duplicates=True,
):
    """
    Main QSAR dataset curation engine.
    """

    # ========================================================
    # READ
    # ========================================================

    try:

        df = pd.read_csv(
            io.BytesIO(file_bytes)
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to read CSV: {exc}"
        )

    if df.empty:

        raise ValueError(
            "CSV file is empty"
        )

    # ========================================================
    # DETECT COLUMNS
    # ========================================================

    smiles_col = find_column(
        df,
        SMILES_ALIASES
    )

    activity_col = find_column(
        df,
        ACTIVITY_ALIASES
    )

    unit_col = find_column(
        df,
        UNIT_ALIASES
    )

    endpoint_col = find_column(
        df,
        ENDPOINT_ALIASES
    )

    relation_col = find_column(
        df,
        RELATION_ALIASES
    )

    if smiles_col is None:

        raise ValueError(
            "No SMILES column detected"
        )

    if activity_col is None:

        raise ValueError(
            "No activity column detected"
        )

    activity_mode = detect_activity_mode(
        activity_col
    )

    # ========================================================
    # WORKING DATAFRAME
    # ========================================================

    work = df.copy()

    work["canonical_smiles"] = None
    work["pIC50"] = None

    work["measurement_count"] = 1

    work["pIC50_mean"] = None
    work["pIC50_std"] = None
    work["pIC50_min"] = None
    work["pIC50_max"] = None

    work["molecular_weight"] = None
    work["logp"] = None
    work["tpsa"] = None
    work["hbd"] = None
    work["hba"] = None
    work["rotatable_bonds"] = None
    work["rings"] = None

    work["curation_status"] = None
    work["curation_message"] = None
    work["quality_flag"] = None

    work["salt_standardized"] = False

    work["activity_mode"] = (
        activity_mode
    )

    # ========================================================
    # NORMALIZED ENDPOINT
    # ========================================================

    if endpoint_col is not None:

        work["endpoint"] = (
            work[endpoint_col]
            .apply(safe_json_value)
        )

    else:

        work["endpoint"] = None

    # ========================================================
    # NORMALIZED RELATION
    # ========================================================

    if relation_col is not None:

        work["relation"] = (
            work[relation_col]
            .apply(safe_json_value)
        )

    else:

        work["relation"] = None

    # ========================================================
    # NORMALIZED UNITS
    # ========================================================

    if unit_col is not None:

        work["units"] = (
            work[unit_col]
            .apply(safe_json_value)
        )

    else:

        work["units"] = None

    # ========================================================
    # COUNTERS
    # ========================================================

    total_rows = len(work)

    invalid_smiles = 0
    missing_smiles = 0

    endpoint_rejected = 0
    relation_rejected = 0

    missing_activity = 0
    invalid_activity = 0

    salt_standardized = 0

    duplicate_measurement_rows = 0

    # ========================================================
    # PROCESS ROWS
    # ========================================================

    for idx in work.index:

        # ----------------------------------------------------
        # SMILES
        # ----------------------------------------------------

        smiles = work.at[
            idx,
            smiles_col
        ]

        smiles_result = canonicalize_smiles(
            smiles,
            remove_salt=remove_salt
        )

        work.at[
            idx,
            "canonical_smiles"
        ] = smiles_result[
            "canonical_smiles"
        ]

        work.at[
            idx,
            "salt_standardized"
        ] = smiles_result[
            "salt_standardized"
        ]

        if smiles_result["status"] == "MISSING":

            missing_smiles += 1

            work.at[
                idx,
                "curation_status"
            ] = "MISSING"

            work.at[
                idx,
                "curation_message"
            ] = "Missing SMILES"

            continue

        if smiles_result["status"] == "INVALID":

            invalid_smiles += 1

            work.at[
                idx,
                "curation_status"
            ] = "INVALID"

            work.at[
                idx,
                "curation_message"
            ] = "Invalid SMILES"

            continue

        if smiles_result[
            "salt_standardized"
        ]:

            salt_standardized += 1

        # ----------------------------------------------------
        # ENDPOINT
        # ----------------------------------------------------

        endpoint_value = work.at[
            idx,
            "endpoint"
        ]

        if (
            strict_endpoint
            and endpoint_col is not None
        ):

            if endpoint_value is not None:

                normalized_endpoint = (
                    str(endpoint_value)
                    .strip()
                    .upper()
                )

                # pChEMBL is representation.
                if normalized_endpoint in {
                    "PCHEMBL",
                    "P-CHEMBL",
                    "P_CHEMBL",
                }:

                    pass

                # IC50 is accepted.
                elif normalized_endpoint == "IC50":

                    pass

                # Other endpoints rejected.
                else:

                    endpoint_rejected += 1

                    work.at[
                        idx,
                        "curation_status"
                    ] = "ENDPOINT_REJECTED"

                    work.at[
                        idx,
                        "curation_message"
                    ] = (
                        f"Endpoint "
                        f"'{endpoint_value}' "
                        f"is not IC50"
                    )

                    continue

        # ----------------------------------------------------
        # RELATION
        # ----------------------------------------------------

        relation_value = work.at[
            idx,
            "relation"
        ]

        if (
            strict_relation
            and relation_col is not None
        ):

            if relation_value is not None:

                normalized_relation = (
                    str(relation_value)
                    .strip()
                )

                if normalized_relation != "=":

                    relation_rejected += 1

                    work.at[
                        idx,
                        "curation_status"
                    ] = "RELATION_REJECTED"

                    work.at[
                        idx,
                        "curation_message"
                    ] = (
                        f"Relation "
                        f"'{relation_value}' "
                        f"is not '='"
                    )

                    continue

        # ----------------------------------------------------
        # ACTIVITY
        # ----------------------------------------------------

        value = work.at[
            idx,
            activity_col
        ]

        unit = work.at[
            idx,
            "units"
        ]

        (
            pic50,
            activity_status,
            activity_message
        ) = parse_activity(
            value,
            activity_mode,
            unit
        )

        # Missing activity
        if activity_status == "MISSING_ACTIVITY":

            missing_activity += 1

            work.at[
                idx,
                "curation_status"
            ] = "MISSING_ACTIVITY"

            work.at[
                idx,
                "curation_message"
            ] = activity_message

            continue

        # Invalid activity
        if activity_status == "INVALID_ACTIVITY":

            invalid_activity += 1

            work.at[
                idx,
                "curation_status"
            ] = "INVALID_ACTIVITY"

            work.at[
                idx,
                "curation_message"
            ] = activity_message

            continue

        # ----------------------------------------------------
        # VALID ACTIVITY
        # ----------------------------------------------------

        work.at[
            idx,
            "pIC50"
        ] = float(pic50)

        work.at[
            idx,
            "curation_status"
        ] = "VALID"

        work.at[
            idx,
            "curation_message"
        ] = activity_message

        # ----------------------------------------------------
        # QUALITY FLAGS
        # ----------------------------------------------------

        flags = []

        if pic50 < 1:

            flags.append(
                "EXTREME_LOW_PIC50"
            )

        if pic50 > 12:

            flags.append(
                "EXTREME_HIGH_PIC50"
            )

        if (
            activity_mode == "CONCENTRATION"
            and (
                unit is None
                or str(unit).strip() == ""
            )
        ):

            flags.append(
                "MISSING_UNIT_ASSUMED_NM"
            )

        if flags:

            work.at[
                idx,
                "quality_flag"
            ] = ";".join(flags)

    # ========================================================
    # VALID RECORDS
    # ========================================================

    valid = work[
        work[
            "curation_status"
        ] == "VALID"
    ].copy()

    unique_compounds_before_aggregation = (
        valid[
            "canonical_smiles"
        ].nunique()
    )

    # ========================================================
    # DESCRIPTORS
    # ========================================================

    for idx in valid.index:

        descriptors = (
            calculate_basic_descriptors(
                valid.at[
                    idx,
                    "canonical_smiles"
                ]
            )
        )

        for key, value in descriptors.items():

            work.at[
                idx,
                key
            ] = value

    valid = work[
        work[
            "curation_status"
        ] == "VALID"
    ].copy()

    # ========================================================
    # DESCRIPTOR FLAGS
    # ========================================================

    for idx in valid.index:

        flags = []

        mw = valid.at[
            idx,
            "molecular_weight"
        ]

        pic50 = valid.at[
            idx,
            "pIC50"
        ]

        if mw is not None:

            if mw > 1000:
                flags.append(
                    "HIGH_MOLECULAR_WEIGHT"
                )

            if mw < 50:
                flags.append(
                    "LOW_MOLECULAR_WEIGHT"
                )

        if pic50 is not None:

            if pic50 < 1:
                flags.append(
                    "EXTREME_LOW_PIC50"
                )

            if pic50 > 12:
                flags.append(
                    "EXTREME_HIGH_PIC50"
                )

        if flags:

            existing = work.at[
                idx,
                "quality_flag"
            ]

            if existing:

                work.at[
                    idx,
                    "quality_flag"
                ] = (
                    str(existing)
                    + ";"
                    + ";".join(flags)
                )

            else:

                work.at[
                    idx,
                    "quality_flag"
                ] = ";".join(flags)

    # ========================================================
    # DUPLICATE AGGREGATION
    # ========================================================

    if (
        aggregate_duplicates
        and not valid.empty
    ):

        grouped_rows = []

        for canonical_smiles, group in (
            valid.groupby(
                "canonical_smiles",
                sort=False
            )
        ):

            measurements = (
                pd.to_numeric(
                    group["pIC50"],
                    errors="coerce"
                )
                .dropna()
            )

            count = len(
                measurements
            )

            if count > 1:

                duplicate_measurement_rows += (
                    count - 1
                )

            representative = (
                group.iloc[0].copy()
            )

            representative[
                "measurement_count"
            ] = count

            representative[
                "pIC50_mean"
            ] = float(
                measurements.mean()
            )

            representative[
                "pIC50_std"
            ] = (
                float(
                    measurements.std()
                )
                if count > 1
                else 0.0
            )

            representative[
                "pIC50_min"
            ] = float(
                measurements.min()
            )

            representative[
                "pIC50_max"
            ] = float(
                measurements.max()
            )

            # Median activity
            representative[
                "pIC50"
            ] = float(
                measurements.median()
            )

            grouped_rows.append(
                representative
            )

        clean = pd.DataFrame(
            grouped_rows
        ).reset_index(
            drop=True
        )

    else:

        clean = valid.copy()

        if not clean.empty:

            clean[
                "measurement_count"
            ] = 1

            clean[
                "pIC50_mean"
            ] = clean[
                "pIC50"
            ]

            clean[
                "pIC50_std"
            ] = 0.0

            clean[
                "pIC50_min"
            ] = clean[
                "pIC50"
            ]

            clean[
                "pIC50_max"
            ] = clean[
                "pIC50"
            ]

    # ========================================================
    # REJECTED DATASET
    # ========================================================

    rejected = work[
        work[
            "curation_status"
        ] != "VALID"
    ].copy()

    # ========================================================
    # STATUS COUNTS
    # ========================================================

    status_counts = {
        str(k): int(v)
        for k, v in
        work[
            "curation_status"
        ]
        .value_counts(
            dropna=False
        )
        .items()
    }

    # ========================================================
    # QUALITY
    # ========================================================

    quality = calculate_statistics(
        clean
    )

    quality.update({

        "activity_mode":
            activity_mode,

        "activity_column":
            activity_col,

        "smiles_column":
            smiles_col,

        "unit_column":
            unit_col,

        "endpoint_column":
            endpoint_col,

        "relation_column":
            relation_col,

        "curator_version":
            VERSION,

        "strict_endpoint":
            strict_endpoint,

        "strict_relation":
            strict_relation,

        "remove_salt":
            remove_salt,

        "aggregate_duplicates":
            aggregate_duplicates,

        "status_counts":
            status_counts,

    })

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "curator_version":
            VERSION,

        "total_rows":
            int(total_rows),

        "valid_smiles":
            int(
                total_rows
                - invalid_smiles
                - missing_smiles
            ),

        "invalid_smiles":
            int(invalid_smiles),

        "missing_smiles":
            int(missing_smiles),

        "endpoint_rejected":
            int(endpoint_rejected),

        "relation_rejected":
            int(relation_rejected),

        "missing_activity":
            int(missing_activity),

        "invalid_activity":
            int(invalid_activity),

        "salt_standardized":
            int(salt_standardized),

        "duplicate_measurement_rows":
            int(duplicate_measurement_rows),

        "unique_compounds_before_aggregation":
            int(
                unique_compounds_before_aggregation
            ),

        "clean_compounds":
            int(len(clean)),

        "activity_valid":
            int(len(valid)),

        "rejected_records":
            int(len(rejected)),

        "quality_flags":
            int(
                clean[
                    "quality_flag"
                ]
                .notna()
                .sum()
            )
            if not clean.empty
            else 0,
    }

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "version":
            VERSION,

        "raw_dataset":
            df,

        "curated_dataset":
            work,

        "clean_dataset":
            clean,

        "rejected_records":
            rejected,

        "summary":
            summary,

        "quality":
            quality,

        "detected_columns": {

            "smiles":
                smiles_col,

            "activity":
                activity_col,

            "units":
                unit_col,

            "endpoint":
                endpoint_col,

            "relation":
                relation_col,
        },
    }


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def make_json_safe(value):

    if isinstance(value, dict):

        return {
            str(k):
                make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):

        return [
            make_json_safe(v)
            for v in value
        ]

    if isinstance(
        value,
        (
            pd.Timestamp,
            pd.Timedelta
        )
    ):

        return str(value)

    if hasattr(value, "item"):

        try:
            return value.item()
        except Exception:
            pass

    if value is None:
        return None

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

    return value


# ============================================================
# SAVE CURATION OUTPUTS
# ============================================================

def save_curation_outputs(
    result,
    output_dir="curation_output"
):
    """
    Save:

        clean_dataset.csv
        rejected_records.csv
        curated_dataset.csv
        curation_report.json
        curation_report.txt
    """

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Clean dataset
    # --------------------------------------------------------

    clean_file = (
        output_path
        / "clean_dataset.csv"
    )

    result[
        "clean_dataset"
    ].to_csv(
        clean_file,
        index=False
    )

    # --------------------------------------------------------
    # Rejected dataset
    # --------------------------------------------------------

    rejected_file = (
        output_path
        / "rejected_records.csv"
    )

    result[
        "rejected_records"
    ].to_csv(
        rejected_file,
        index=False
    )

    # --------------------------------------------------------
    # Full curated dataset
    # --------------------------------------------------------

    curated_file = (
        output_path
        / "curated_dataset.csv"
    )

    result[
        "curated_dataset"
    ].to_csv(
        curated_file,
        index=False
    )

    # --------------------------------------------------------
    # JSON report
    # --------------------------------------------------------

    report = {

        "curator_version":
            result["version"],

        "summary":
            result["summary"],

        "quality":
            result["quality"],

        "detected_columns":
            result[
                "detected_columns"
            ],
    }

    json_file = (
        output_path
        / "curation_report.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            make_json_safe(report),
            handle,
            indent=2
        )

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    txt_file = (
        output_path
        / "curation_report.txt"
    )

    with open(
        txt_file,
        "w",
        encoding="utf-8"
    ) as handle:

        handle.write(
            "=" * 70
            + "\n"
        )

        handle.write(
            "QSAR-X PROFESSIONAL\n"
        )

        handle.write(
            "DATASET CURATION REPORT\n"
        )

        handle.write(
            "=" * 70
            + "\n\n"
        )

        handle.write(
            f"Curator version: "
            f"{result['version']}\n\n"
        )

        handle.write(
            "SUMMARY\n"
        )

        handle.write(
            "-" * 70
            + "\n"
        )

        for key, value in (
            result[
                "summary"
            ].items()
        ):

            handle.write(
                f"{key}: {value}\n"
            )

        handle.write(
            "\nQUALITY\n"
        )

        handle.write(
            "-" * 70
            + "\n"
        )

        for key, value in (
            result[
                "quality"
            ].items()
        ):

            handle.write(
                f"{key}: {value}\n"
            )

        handle.write(
            "\nDETECTED COLUMNS\n"
        )

        handle.write(
            "-" * 70
            + "\n"
        )

        for key, value in (
            result[
                "detected_columns"
            ].items()
        ):

            handle.write(
                f"{key}: {value}\n"
            )

    return {

        "clean_dataset":
            str(clean_file),

        "rejected_records":
            str(rejected_file),

        "curated_dataset":
            str(curated_file),

        "json_report":
            str(json_file),

        "text_report":
            str(txt_file),
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "QSAR-X Professional "
            "Dataset Curator"
        )
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        help="Input CSV file"
    )

    parser.add_argument(
        "--output",
        default="curation_output",
        help="Output directory"
    )

    parser.add_argument(
        "--no-salt-removal",
        action="store_true",
        help="Disable salt removal"
    )

    parser.add_argument(
        "--no-aggregation",
        action="store_true",
        help="Disable duplicate aggregation"
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("QSAR-X PROFESSIONAL DATASET CURATOR")
    print(f"Version: {VERSION}")
    print("=" * 70)
    print()

    if not args.input_csv:

        print(
            "Curator module loaded successfully."
        )

        print()
        print(
            "Example:"
        )

        print(
            "python curator.py compound.csv"
        )

        raise SystemExit(0)

    input_path = Path(
        args.input_csv
    )

    if not input_path.exists():

        raise SystemExit(
            f"File not found: {input_path}"
        )

    print(
        f"Input: {input_path}"
    )

    print()

    result = curate_dataset(
        input_path.read_bytes(),

        strict_endpoint=True,

        strict_relation=True,

        remove_salt=(
            not args.no_salt_removal
        ),

        aggregate_duplicates=(
            not args.no_aggregation
        ),
    )

    print(
        "SUMMARY"
    )

    print(
        "-" * 70
    )

    for key, value in (
        result[
            "summary"
        ].items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "STATUS COUNTS"
    )

    print(
        "-" * 70
    )

    print(
        result[
            "quality"
        ].get(
            "status_counts",
            {}
        )
    )

    print()

    print(
        "Generating output files..."
    )

    outputs = save_curation_outputs(
        result,
        args.output
    )

    print()

    print(
        "OUTPUT FILES"
    )

    print(
        "-" * 70
    )

    for key, value in outputs.items():

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "Curation completed successfully."
    )