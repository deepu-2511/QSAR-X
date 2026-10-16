"""Finish the JAK deep-dive and reproducibility manifest after model-wide evaluation."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn

from research_evaluation import ROOT, RESULTS, jak_deep_dive, mkdirs


def main() -> None:
    paths = mkdirs()
    registry=json.loads((ROOT/"models"/"target_registry.json").read_text())["targets"]
    target=next(row for row in registry if row["target_id"]=="JAK_FAMILY")
    base=ROOT/target["evaluation_report"]
    split=pd.read_csv(base.parent/"split_assignments.csv")
    model=joblib.load(ROOT/target["model_path"])
    context=jak_deep_dive(ROOT/"data"/"targets"/"JAK_FAMILY"/"JAK_family_pChEMBL_clean.csv",split,model,paths)
    metrics=pd.read_csv(RESULTS/"metrics"/"registered_target_scaffold_test_metrics.csv")
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"random_seed":42,"fingerprint":{"type":"Morgan","radius":2,"bits":2048},"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,"rdkit":rdkit.__version__,"scipy":scipy.__version__,"registered_models_evaluated":len(registry),"JAK_deep_dive":context}
    (paths["validation"] / "reproducibility_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    summary={"models":len(metrics),"mean_internal_scaffold_R2":float(metrics.R2.mean()),"mean_internal_scaffold_RMSE":float(metrics.RMSE_pIC50.mean()),"all_external_gates_passed":int(metrics.external_quality_gates_passed.sum()),"limitations":["All recomputed performance values are internal scaffold-held-out measurements.","No registered model currently has strict external validation gates passed.","JAK_FAMILY is pooled pChEMBL data with no per-row isoform or assay metadata."]}
    (paths["metrics"] / "evaluation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("JAK deep dive and reproducibility manifest complete.")

if __name__ == "__main__": main()
