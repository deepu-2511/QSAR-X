from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field
import json
from pathlib import Path
import re

from predictor import predict, predict_batch, predict_general
from curator import curate_dataset

REGISTRY_FILE = Path(__file__).resolve().parent.parent / "models" / "target_registry.json"
TARGET_DATA_DIR = REGISTRY_FILE.parent.parent / "data" / "targets"
MULTITARGET_MANIFEST_FILE = REGISTRY_FILE.parent / "multitarget" / "active_candidate.json"
FRONTEND_DIST = REGISTRY_FILE.parent.parent / "frontend" / "dist"


def load_target_registry():
    with REGISTRY_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_target(target_id: str):
    for target in load_target_registry().get("targets", []):
        if target["target_id"].casefold() == target_id.casefold():
            return target
    return None


def normalize_protein_sequence(value: str) -> str:
    sequence = value.strip().upper()
    if sequence.startswith(">"):
        sequence = "".join(sequence.splitlines()[1:])
    return re.sub(r"\s+", "", sequence)


def _sequence_from_fasta(path: Path) -> str:
    return "".join(line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith(">"))


def resolve_collected_target_query(query: str):
    """Resolve a locally collected panel protein."""
    if not query.strip() or not TARGET_DATA_DIR.is_dir():
        return None
    folded = query.strip().casefold()
    for metadata_path in TARGET_DATA_DIR.glob("*/target_metadata.json"):
        fasta_path = metadata_path.parent / "protein.fasta"
        if not fasta_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        aliases = {str(metadata.get("target_chembl_id", "")), str(metadata.get("pref_name", "")), str(metadata.get("protein_accession", ""))}
        for component in metadata.get("target_components", []):
            for synonym in component.get("target_component_synonyms", []):
                aliases.add(str(synonym.get("component_synonym", "")))
        if any(alias and alias.casefold() == folded for alias in aliases):
            gene = next((str(item.get("component_synonym")) for component in metadata.get("target_components", []) for item in component.get("target_component_synonyms", []) if item.get("syn_type") == "GENE_SYMBOL"), metadata.get("pref_name"))
            return {"target_id": metadata.get("target_chembl_id"), "gene": gene, "name": metadata.get("pref_name"), "uniprot_accession": metadata.get("protein_accession"), "organism": metadata.get("organism"), "sequence": _sequence_from_fasta(fasta_path)}
    return None


def list_research_panel_targets():
    panel = []
    for metadata_path in sorted(TARGET_DATA_DIR.glob("CHEMBL*/target_metadata.json")):
        target_id = metadata_path.parent.name
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gene = next((str(item.get("component_synonym")) for component in metadata.get("target_components", []) for item in component.get("target_component_synonyms", []) if item.get("syn_type") == "GENE_SYMBOL"), metadata.get("pref_name"))
        model_root = REGISTRY_FILE.parent / target_id
        versions = sorted(model_root.glob("baseline_v*"), key=lambda path: path.name)
        latest = versions[-1] if versions else None
        report_path = latest / "evaluation_report.json" if latest else None
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path and report_path.is_file() else {}
        external_path = latest / "external_validation_report.json" if latest else None
        external = json.loads(external_path.read_text(encoding="utf-8")) if external_path and external_path.is_file() else {}
        stage = "training"
        if report:
            stage = "validated" if external.get("quality_gates_passed") else ("external_evidence_recorded" if external else "ready_for_external_validation")
        panel.append({
            "target_id": target_id,
            "gene": gene,
            "name": metadata.get("pref_name"),
            "uniprot_accession": metadata.get("protein_accession"),
            "organism": metadata.get("organism"),
            "prediction_mode": "target_specific_candidate",
            "training_stage": stage,
            "model_version": latest.name if latest else None,
            "scaffold_test_metrics": report.get("scaffold_test_metrics"),
            "external_validation": external.get("metrics_on_external_source_nonoverlap"),
        })
    return panel


def resolve_target_query(query: str):
    """Resolve ChEMBL IDs, gene symbols, UniProt accessions, names or exact FASTA sequences."""
    normalized = query.strip()
    if not normalized:
        return None
    targets = load_target_registry().get("targets", [])
    folded = normalized.casefold()
    for target in targets:
        aliases = {
            target.get("target_id", ""),
            target.get("gene", ""),
            target.get("name", ""),
            target.get("uniprot_accession", ""),
        }
        if any(alias and alias.casefold() == folded for alias in aliases):
            return target

    collected = resolve_collected_target_query(normalized)
    if collected:
        return {**collected, "prediction_available": False, "prediction_mode": "target_specific", "model_status": "target_specific_training_or_validation_pending"}

    sequence = normalize_protein_sequence(normalized)
    if len(sequence) < 50 or re.search(r"[^ACDEFGHIKLMNPQRSTVWY]", sequence):
        return None
    for target in targets:
        fasta_path = REGISTRY_FILE.parent.parent / "data" / "targets" / target["target_id"] / "protein.fasta"
        if not fasta_path.exists():
            continue
        lines = fasta_path.read_text(encoding="utf-8").splitlines()
        registered_sequence = "".join(line.strip().upper() for line in lines if line and not line.startswith(">"))
        if sequence == registered_sequence:
            return target
    return None


def require_predictable_target(target_id: str):
    target = get_target(target_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "UNKNOWN_TARGET", "message": f"Target '{target_id}' is not in the registry."},
        )
    if not target.get("prediction_available", False):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "MODEL_NOT_APPROVED",
                "message": f"{target['gene']} has no available target-specific model.",
                "target_id": target["target_id"],
                "model_status": target["model_status"],
            },
        )
    return target


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="QSAR-X Professional API",

    description=(
        "Target-aware QSAR API backed by a versioned model registry. "
        "Target-specific screening models can predict. "
        "Unregistered targets require target-specific assay data, training, and validation first."
    ),

    version="1.3.0"
)

if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Chemical SMILES string"
    )

    target_id: str | None = Field(
        default=None,
        min_length=1,
        description="Registered target identifier; may be omitted if protein_target is supplied",
    )
    protein_target: str | None = Field(
        default=None,
        min_length=1,
        max_length=100000,
        description="Registered gene/name/UniProt/ChEMBL ID or exact protein sequence",
    )


class TargetResolutionRequest(BaseModel):
    protein_target: str = Field(..., min_length=1, max_length=100000)


class GeneralPredictionRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=20000)
    protein_sequence: str | None = Field(default=None, min_length=50, max_length=100000)
    protein_target: str | None = Field(default=None, min_length=1, max_length=100000)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    if (FRONTEND_DIST / "index.html").is_file():
        return FileResponse(FRONTEND_DIST / "index.html")

    return {

        "name": "QSAR-X Professional API",

        "version": "1.3.0",

        "status": "running",

        "targets": [
            {
                "target_id": target["target_id"],
                "gene": target["gene"],
                "name": target["name"],
                "organism": target["organism"],
                "model_status": target["model_status"],
                "prediction_available": target["prediction_available"],
                "endpoint": target["endpoint"],
            }
            for target in load_target_registry()["targets"]
        ],

        "features": [

            "Protein target resolution",

            "Model status and validation reports",

            "Single compound prediction",

            "Batch CSV prediction",

            "Dataset upload and curation",

            "pIC50 prediction",

            "IC50 conversion",

            "Uncertainty estimation",

            "Applicability domain",

            "RDKit structure rendering"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    active_models = sum(
        bool(target.get("prediction_available"))
        for target in load_target_registry()["targets"]
    )

    return {

        "status": "ok",

        "service": "QSAR-X Professional API",

        "model": "versioned target registry",

        "prediction_enabled_target_models": active_models,

        "version": "1.3.0"
    }


@app.get("/targets")
def list_targets():
    return {"targets": load_target_registry()["targets"]}


@app.get("/research-targets")
def research_targets():
    return {"targets": list_research_panel_targets(), "model_status": "target_specific_models"}


@app.get("/targets/{target_id}")
def target_details(target_id: str):
    target = get_target(target_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Target not found.")
    return target


@app.post("/resolve-target")
def resolve_target_endpoint(request: TargetResolutionRequest):
    target = resolve_target_query(request.protein_target)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "TARGET_NOT_RESOLVED",
                "message": "Target is not in the local model registry. For a new protein, first resolve its ChEMBL single-protein target and collect target-specific IC50 assay data, then curate, train, and validate a model; protein sequence plus SMILES alone is not enough to produce a reliable QSAR value.",
            },
        )
    return {
        "success": True,
        "target": target,
        "prediction_available": bool(target.get("prediction_available")),
        "prediction_mode": target.get("prediction_mode", "legacy"),
    }


def _registered_report_path(target: dict, key: str) -> Path | None:
    """Resolve only report paths recorded in the registry and confined to this project."""
    relative = target.get(key)
    if not relative:
        return None
    root = REGISTRY_FILE.parent.parent.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise HTTPException(status_code=500, detail="Registered report path is outside the project.")
    return path


@app.get("/models/{target_id}/status")
def model_status(target_id: str):
    target = get_target(target_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Target not found.")
    model_path_value = target.get("model_path")
    model_path = (REGISTRY_FILE.parent.parent / model_path_value) if model_path_value else None
    return {
        "target_id": target["target_id"],
        "gene": target.get("gene"),
        "model_version": target.get("model_version"),
        "model_status": target.get("model_status"),
        "prediction_mode": target.get("prediction_mode", "legacy"),
        "prediction_available": bool(target.get("prediction_available")),
        "model_file_exists": bool(model_path and model_path.is_file()),
        "quality_gates_passed": target.get("quality_gates_passed"),
        "validation": target.get("validation", {}),
    }


@app.get("/reports/{target_id}")
def target_reports(target_id: str):
    target = get_target(target_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Target not found.")
    reports = {}
    for key in ("evaluation_report", "external_validation_report", "quality_gate_report"):
        path = _registered_report_path(target, key)
        if path is None:
            continue
        if not path.is_file():
            reports[key] = {"available": False}
            continue
        try:
            reports[key] = {"available": True, "report": json.loads(path.read_text(encoding="utf-8"))}
        except (OSError, json.JSONDecodeError) as exc:
            reports[key] = {"available": False, "error": str(exc)}
    return {"target_id": target["target_id"], "model_version": target.get("model_version"), "reports": reports}


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/predict")
def predict_endpoint(
    request: PredictionRequest
):

    try:

        resolved_target = (
            resolve_target_query(request.protein_target)
            if request.protein_target
            else get_target(request.target_id or "")
        )
        if resolved_target is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "TARGET_NOT_RESOLVED", "message": "Provide a registered protein target."},
            )
        if request.protein_target and request.target_id:
            target_id_from_id = get_target(request.target_id)
            if target_id_from_id is None or target_id_from_id["target_id"] != resolved_target["target_id"]:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "TARGET_INPUT_MISMATCH", "message": "target_id and protein_target resolve to different targets."},
                )
        target = require_predictable_target(resolved_target["target_id"])

        result = predict(
            request.smiles,
            target_id=target["target_id"],
        )

        return {

            "success": True,

            "result": result
        }

    except HTTPException:

        raise

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction error: {str(e)}"
            )
        )


@app.post("/predict-general")
def predict_general_endpoint(request: GeneralPredictionRequest):
    """Research-only multi-target prediction for an unregistered protein sequence."""
    try:
        sequence = request.protein_sequence
        if not sequence and request.protein_target:
            collected = resolve_collected_target_query(request.protein_target)
            if collected:
                sequence = collected["sequence"]
        if not sequence:
            raise ValueError("Provide a protein sequence or a protein from the locally collected target panel.")
        return {"success": True, "result": predict_general(request.smiles, sequence)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail={"code": "MULTITARGET_MODEL_NOT_READY", "message": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"General prediction error: {exc}")


# ============================================================
# BATCH CSV PREDICTION
# ============================================================

@app.post("/predict-batch")
async def predict_batch_endpoint(
    file: UploadFile = File(...),
    target_id: str = Form(default="CHEMBL203"),
):

    try:

        target = require_predictable_target(target_id)

        # ----------------------------------------------------
        # FILE NAME
        # ----------------------------------------------------

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="No file provided."
            )

        # ----------------------------------------------------
        # FILE TYPE
        # ----------------------------------------------------

        if not file.filename.lower().endswith(".csv"):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Only CSV files are supported."
                )
            )

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        contents = await file.read()

        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Batch CSV must be 5 MB or smaller.")

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty."
            )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        result = predict_batch(
            contents,
            target_id=target["target_id"],
        )

        if result["total_compounds"] > 1000:
            raise HTTPException(status_code=413, detail="Batch CSV is limited to 1,000 rows per request.")

        return {

            "success": True,

            "filename": file.filename,

            "result": result
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Batch prediction error: {str(e)}"
            )
        )


# ============================================================
# DATASET UPLOAD & CURATION
# ============================================================

@app.post("/curate-dataset")
async def curate_dataset_endpoint(
    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # FILE NAME
        # ----------------------------------------------------

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="No file provided."
            )

        # ----------------------------------------------------
        # FILE TYPE
        # ----------------------------------------------------

        if not file.filename.lower().endswith(".csv"):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Only CSV files are supported."
                )
            )

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty."
            )

        # ----------------------------------------------------
        # CURATE DATASET
        # ----------------------------------------------------

        result = curate_dataset(
            contents
        )

        return {

            "success": True,

            "filename": file.filename,

            "result": result
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Dataset curation error: {str(e)}"
            )
        )
