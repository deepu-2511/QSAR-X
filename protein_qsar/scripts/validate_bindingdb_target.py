"""Run a reproducible external BindingDB IC50 check for one target candidate.

BindingDB REST provides affinity type/value and structures, but does not supply
all provenance required for production promotion. The resulting report is an
external-source validation record and deliberately remains non-promotional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NUMBER = re.compile(r"^[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(value: object) -> str | None:
    molecule = Chem.MolFromSmiles(str(value).strip()) if value else None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_id")
    parser.add_argument("model_version")
    parser.add_argument("--cutoff-nm", type=int, default=100000)
    args = parser.parse_args()
    target_id = args.target_id.upper()
    root = ROOT / "data" / "targets" / target_id
    metadata = json.loads((root / "target_metadata.json").read_text(encoding="utf-8"))
    accession = metadata["protein_accession"]
    model_dir = ROOT / "models" / target_id / args.model_version
    model_path = model_dir / "candidate_model.joblib"
    training_path = root / "IC50_clean.csv"
    if not model_path.is_file() or not training_path.is_file():
        raise FileNotFoundError("Candidate model or curated target data is missing.")

    url = f"https://bindingdb.org/rest/getLigandsByUniprot?uniprot={accession};{args.cutoff_nm}&response=application/json"
    request = Request(url, headers={"User-Agent": "QSAR-X/1.0 external validation"})
    payload = None
    last_error = None
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8").strip()
            if not body:
                raise ValueError("BindingDB returned an empty response")
            payload = json.loads(body)
            break
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                print(f"BindingDB attempt {attempt}/3 failed for {target_id}: {exc}; retrying...", flush=True)
                time.sleep(3 * attempt)
    if payload is None:
        report = {
            "target_chembl_id": target_id,
            "target_uniprot_accession": accession,
            "model_version": args.model_version,
            "source_database": "BindingDB REST",
            "source_url": url,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "quality_gates_passed": False,
            "status": "external_validation_source_unavailable",
            "error": str(last_error),
        }
        (model_dir / "external_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    (root / "BindingDB_IC50_external_raw.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    response_data = payload.get("getLindsByUniprotResponse", {})
    rows = []
    for record in response_data.get("bdb.affinities", []):
        value = str(record.get("bdb.affinity", "")).strip()
        smiles = canonicalize(record.get("bdb.smile"))
        if str(record.get("bdb.affinity_type", "")).upper() != "IC50" or not smiles or not NUMBER.fullmatch(value):
            continue
        activity = float(value)
        if activity > 0 and math.isfinite(activity):
            rows.append({"canonical_smiles": smiles, "activity_nM": activity})
    external = pd.DataFrame(rows)
    if not external.empty:
        external = external.groupby("canonical_smiles", as_index=False)["activity_nM"].median()
    training = pd.read_csv(training_path)
    train_smiles = set(training["canonical_smiles"].dropna().astype(str))
    before_overlap = len(external)
    external = external.loc[~external["canonical_smiles"].isin(train_smiles)].copy()
    external["pIC50"] = 9.0 - np.log10(external["activity_nM"])
    features = []
    for smiles in external["canonical_smiles"]:
        vector = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles)), vector)
        features.append(vector)
    model = joblib.load(model_path)
    predictions = model.predict(np.stack(features)) if features else np.asarray([])
    if len(external) >= 2:
        observed = external["pIC50"].to_numpy(float)
        external["predicted_pIC50"] = predictions
        external["absolute_error_pIC50"] = np.abs(predictions - observed)
        metrics = {"n_compounds": int(len(external)), "R2": float(r2_score(observed, predictions)), "RMSE_pIC50": float(np.sqrt(mean_squared_error(observed, predictions))), "MAE_pIC50": float(mean_absolute_error(observed, predictions)), "Spearman": float(pd.Series(observed).corr(pd.Series(predictions), method="spearman"))}
    else:
        metrics = {"n_compounds": int(len(external)), "R2": None, "RMSE_pIC50": None, "MAE_pIC50": None, "Spearman": None}
    external.to_csv(model_dir / "external_validation_predictions.csv", index=False)
    gates = {"accession_matches": response_data.get("bdb.primary") == accession, "exact_structure_overlap_removed": not bool(set(external["canonical_smiles"]) & train_smiles), "minimum_30_external_compounds": len(external) >= 30, "relation_and_assay_context_documented": False}
    report = {"target_chembl_id": target_id, "target_uniprot_accession": accession, "model_version": args.model_version, "model_sha256": sha256(model_path), "source_database": "BindingDB REST", "source_url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "exact_structure_overlaps_removed": int(before_overlap - len(external)), "metrics_on_external_source_nonoverlap": metrics, "quality_gates": gates, "quality_gates_passed": False, "status": "external_source_exploratory_only", "limitations": ["BindingDB REST omits relation operators, detailed assay context and auditable publication provenance.", "This report cannot promote the model to commercial/validated status."]}
    (model_dir / "external_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
