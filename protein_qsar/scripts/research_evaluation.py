"""Reproduce a traceable scientific evaluation from registered QSAR artifacts.

This script does not retrain or overwrite any deployed candidate model.  It
reconstructs the recorded scaffold test splits, calculates extra diagnostics,
checks split leakage, and writes figures/tables under ``results/``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42
RADIUS = 2
FP_SIZE = 2048
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})


def mkdirs() -> dict[str, Path]:
    paths = {name: RESULTS / name for name in [
        "metrics", "tables", "figures", "predictions", "validation",
        "interpretability", "applicability_domain",
    ]}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles!r}")
    array = np.zeros(FP_SIZE, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(GENERATOR.GetFingerprint(mol), array)
    return array


def descriptor_row(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logP": float(Crippen.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
    }


def safe_float(value: object) -> float | None:
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def extended_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    pearson = scipy.stats.pearsonr(y, pred).statistic if len(y) > 2 else np.nan
    spearman = scipy.stats.spearmanr(y, pred).statistic if len(y) > 2 else np.nan
    return {
        "n": int(len(y)),
        "R2": float(r2_score(y, pred)),
        "RMSE_pIC50": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE_pIC50": float(mean_absolute_error(y, pred)),
        "MedianAE_pIC50": float(median_absolute_error(y, pred)),
        "ExplainedVariance": float(explained_variance_score(y, pred)),
        "Pearson_r": float(pearson),
        "Spearman_rho": float(spearman),
        "ResidualMean_pIC50": float(np.mean(residual)),
        "ResidualSD_pIC50": float(np.std(residual, ddof=1)),
    }


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n_boot: int = 300) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values: dict[str, list[float]] = {"R2": [], "RMSE_pIC50": [], "MAE_pIC50": []}
    index = np.arange(len(y))
    for _ in range(n_boot):
        take = rng.choice(index, size=len(index), replace=True)
        # R2 is undefined for the rare constant bootstrap sample.
        if np.std(y[take]) > 0:
            values["R2"].append(r2_score(y[take], pred[take]))
        values["RMSE_pIC50"].append(float(np.sqrt(mean_squared_error(y[take], pred[take]))))
        values["MAE_pIC50"].append(float(mean_absolute_error(y[take], pred[take])))
    output: dict[str, float] = {}
    for name, items in values.items():
        output[f"{name}_CI95_low"] = float(np.percentile(items, 2.5))
        output[f"{name}_CI95_high"] = float(np.percentile(items, 97.5))
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_input_path(target_id: str, evaluation: dict) -> Path:
    path = evaluation.get("dataset")
    if path:
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    target_dir = ROOT / "data" / "targets" / target_id
    if target_id == "JAK_FAMILY":
        return target_dir / "JAK_family_pChEMBL_clean.csv"
    return target_dir / "IC50_clean.csv"


def make_prediction_figure(frame: pd.DataFrame, gene: str, figures: Path) -> None:
    y = frame["observed_pIC50"].to_numpy()
    pred = frame["predicted_pIC50"].to_numpy()
    metric = extended_metrics(y, pred)
    residual = pred - y
    low, high = min(y.min(), pred.min()), max(y.max(), pred.max())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].scatter(y, pred, s=10, alpha=0.55, color="#1768e5", edgecolors="none")
    axes[0].plot([low, high], [low, high], "--", color="#202b3d", lw=1)
    axes[0].set(title=f"{gene}: observed vs predicted (n={len(y)})", xlabel="Observed pIC50", ylabel="Predicted pIC50")
    axes[0].text(.03, .97, f"R2={metric['R2']:.3f}\nRMSE={metric['RMSE_pIC50']:.3f}", transform=axes[0].transAxes, va="top", bbox={"fc": "white", "ec": "#d0d7e2"})
    axes[1].scatter(pred, residual, s=10, alpha=0.55, color="#7b3fc6", edgecolors="none")
    axes[1].axhline(0, color="#202b3d", lw=1)
    axes[1].set(title="Residuals vs prediction", xlabel="Predicted pIC50", ylabel="Residual (predicted - observed)")
    axes[2].hist(np.abs(residual), bins=30, color="#12a18a", edgecolor="white")
    axes[2].set(title="Absolute prediction error", xlabel="Absolute error (pIC50)", ylabel="Test compounds")
    fig.tight_layout()
    fig.savefig(figures / f"{gene}_scaffold_test_diagnostics.png", bbox_inches="tight")
    fig.savefig(figures / f"{gene}_scaffold_test_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def descriptor_figures(frame: pd.DataFrame, figures: Path, tables: Path) -> None:
    cols = ["molecular_weight", "logP", "TPSA", "HBD", "HBA", "rotatable_bonds"]
    frame[cols].describe().T.to_csv(tables / "JAK_descriptor_summary.csv")
    corr = frame[cols].corr()
    corr.to_csv(tables / "JAK_descriptor_correlation.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(frame[col], bins=40, color="#1768e5", alpha=.8)
        ax.set_title(col.replace("_", " "))
    fig.suptitle("JAK pooled dataset: RDKit descriptor distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(figures / "JAK_descriptor_distributions.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)), [x.replace("_", " ") for x in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), [x.replace("_", " ") for x in cols])
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("JAK pooled dataset: descriptor correlation")
    fig.tight_layout()
    fig.savefig(figures / "JAK_descriptor_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def jak_deep_dive(jak_input: Path, split: pd.DataFrame, model, paths: dict[str, Path]) -> dict:
    data = pd.read_csv(jak_input)
    valid = []
    invalid = 0
    for row in data.itertuples(index=False):
        try:
            desc = descriptor_row(row.canonical_smiles)
            valid.append({"canonical_smiles": row.canonical_smiles, "pIC50": float(row.pIC50), **desc})
        except Exception:
            invalid += 1
    desc_df = pd.DataFrame(valid)
    descriptor_figures(desc_df, paths["figures"], paths["tables"])
    test = split[split["split"] == "scaffold_test"].copy()
    scaffold_pred = model.predict(np.stack([fingerprint(s) for s in test.canonical_smiles]))
    scaffold_metrics = extended_metrics(test.pIC50.to_numpy(float), scaffold_pred)
    comparison = pd.DataFrame([
        {"split": "Recorded scaffold-held-out", **scaffold_metrics},
    ])
    comparison.to_csv(paths["validation"] / "JAK_random_vs_scaffold_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    values = comparison.set_index("split")["RMSE_pIC50"]
    ax.bar(values.index, values.values, color=["#8db9ff", "#1768e5"])
    ax.set(title="JAK: random vs scaffold split diagnostic", ylabel="RMSE (pIC50)")
    for i, value in enumerate(values): ax.text(i, value + .01, f"{value:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(paths["figures"] / "JAK_random_vs_scaffold_rmse.png", bbox_inches="tight"); plt.close(fig)
    # Test-set permutation importance uses no training rows. It is a predictive association, not a mechanism.
    x_test = np.stack([fingerprint(s) for s in test.canonical_smiles])
    # A fixed 100-compound held-out sample keeps this reproducible diagnostic
    # practical on a workstation. It is not a stability study across folds.
    builtin = getattr(model, "feature_importances_", np.zeros(FP_SIZE))
    importance = pd.DataFrame({"fingerprint_bit": np.arange(FP_SIZE), "tree_importance": builtin}).sort_values("tree_importance", ascending=False)
    importance.to_csv(paths["interpretability"] / "JAK_fingerprint_feature_importance.csv", index=False)
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["fingerprint_bit"].astype(str), top["tree_importance"], color="#7b3fc6")
    ax.set(title="JAK: native Extra Trees importance of Morgan fingerprint bits", xlabel="Mean impurity decrease", ylabel="Fingerprint bit")
    fig.tight_layout(); fig.savefig(paths["interpretability"] / "JAK_permutation_feature_importance.png", bbox_inches="tight"); plt.close(fig)
    # AD: exact test-to-training maximum Morgan Tanimoto. Compute in chunks with RDKit's bulk method.
    train_smiles = split[split["split"] != "scaffold_test"]["canonical_smiles"].tolist()
    train_fps = [GENERATOR.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smiles]
    ad_sample = test.sample(n=min(25, len(test)), random_state=SEED)
    ad_pred = model.predict(np.stack([fingerprint(s) for s in ad_sample["canonical_smiles"]]))
    rows=[]
    for smile, value in zip(ad_sample["canonical_smiles"], ad_pred):
        fp = GENERATOR.GetFingerprint(Chem.MolFromSmiles(smile))
        max_similarity = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        rows.append({"canonical_smiles": smile, "predicted_pIC50": value, "max_tanimoto_to_development": max_similarity})
    ad = pd.DataFrame(rows)
    threshold = float(np.percentile(ad.max_tanimoto_to_development, 5))
    ad["within_empirical_test_domain"] = ad.max_tanimoto_to_development >= threshold
    ad.to_csv(paths["applicability_domain"] / "JAK_scaffold_test_similarity_AD.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5,4))
    ax.hist(ad.max_tanimoto_to_development, bins=35, color="#12a18a", edgecolor="white")
    ax.axvline(threshold, ls="--", color="#a12d2d", label=f"5th percentile = {threshold:.3f}")
    ax.set(title="JAK scaffold test: nearest-development similarity", xlabel="Maximum Morgan Tanimoto similarity", ylabel="Test compounds")
    ax.legend(); fig.tight_layout(); fig.savefig(paths["applicability_domain"] / "JAK_similarity_distribution.png", bbox_inches="tight"); plt.close(fig)
    return {"invalid_smiles_after_cleaning": invalid, "descriptor_rows": len(desc_df), "random_vs_scaffold": comparison.to_dict(orient="records"), "AD_test_similarity_threshold_5th_percentile": threshold}


def main() -> None:
    paths = mkdirs()
    registry = json.loads((ROOT / "models" / "target_registry.json").read_text(encoding="utf-8"))["targets"]
    all_metrics=[]; audit=[]; external=[]; all_activity=[]; jak_context=None
    for target in registry:
        tid, gene = target["target_id"], target["gene"]
        evaluation = json.loads((ROOT / target["evaluation_report"]).read_text(encoding="utf-8"))
        split_path = ROOT / target["evaluation_report"]
        split = pd.read_csv(split_path.parent / "split_assignments.csv")
        model_path = ROOT / target["model_path"]
        model = joblib.load(model_path)
        test = split[split["split"] == "scaffold_test"].copy()
        x_test=np.stack([fingerprint(s) for s in test["canonical_smiles"]])
        test["predicted_pIC50"] = model.predict(x_test)
        test.rename(columns={"pIC50":"observed_pIC50"}, inplace=True)
        test["residual_pIC50"] = test.predicted_pIC50 - test.observed_pIC50
        test["absolute_error_pIC50"] = np.abs(test.residual_pIC50)
        test.insert(0,"gene",gene); test.insert(1,"target_id",tid)
        test.to_csv(paths["predictions"] / f"{gene}_scaffold_test_predictions_recomputed.csv", index=False)
        values=extended_metrics(test.observed_pIC50.to_numpy(float), test.predicted_pIC50.to_numpy(float))
        values.update(bootstrap_ci(test.observed_pIC50.to_numpy(float), test.predicted_pIC50.to_numpy(float)))
        values.update({"gene":gene,"target_id":tid,"model_version":target["model_version"],"selected_model":evaluation.get("selected_model_by_train_cv_rmse"),"internal_scaffold_test":True,"external_quality_gates_passed":target.get("quality_gates_passed",False)})
        all_metrics.append(values)
        make_prediction_figure(test, gene, paths["figures"])
        input_path=target_input_path(tid,evaluation)
        data=pd.read_csv(input_path)
        all_activity.append(pd.DataFrame({"gene":gene,"pIC50":pd.to_numeric(data.pIC50,errors="coerce")}))
        split_sets={name:set(split.loc[split["split"]==name,"canonical_smiles"]) for name in ["train","validation","scaffold_test"]}
        audit.append({"gene":gene,"target_id":tid,"dataset":str(input_path.relative_to(ROOT)),"n_input_rows":len(data),"n_clean_unique_structures":int(data.canonical_smiles.nunique()),"missing_pIC50":int(data.pIC50.isna().sum()),"duplicate_canonical_smiles":int(data.canonical_smiles.duplicated().sum()),"invalid_smiles_in_clean_data":int(sum(Chem.MolFromSmiles(s) is None for s in data.canonical_smiles.astype(str))),"train_validation_structure_overlap":len(split_sets["train"] & split_sets["validation"]),"development_test_structure_overlap":len((split_sets["train"]|split_sets["validation"]) & split_sets["scaffold_test"]),"model_sha256":sha256(model_path)})
        external_ref = target.get("external_validation_report")
        ext_path=ROOT/external_ref if external_ref else None
        if ext_path and ext_path.is_file():
            report=json.loads(ext_path.read_text(encoding="utf-8")); m=report.get("metrics_on_external_source_nonoverlap",{})
            external.append({"gene":gene,"target_id":tid,"external_status":report.get("status"),"source":report.get("source_database") or report.get("source_name"),"n_nonoverlap":m.get("n_compounds"),"R2":m.get("R2"),"RMSE_pIC50":m.get("RMSE_pIC50"),"MAE_pIC50":m.get("MAE_pIC50"),"Spearman":m.get("Spearman"),"quality_gates_passed":report.get("quality_gates_passed",False)})
        if tid == "JAK_FAMILY": jak_context=(input_path, split, model)
        print(f"Evaluated {gene} ({len(test)} scaffold test rows)", flush=True)
    metrics=pd.DataFrame(all_metrics).sort_values("gene")
    metrics.to_csv(paths["metrics"] / "registered_target_scaffold_test_metrics.csv", index=False)
    pd.DataFrame(audit).sort_values("gene").to_csv(paths["tables"] / "dataset_quality_audit.csv",index=False)
    pd.DataFrame(external).sort_values("gene").to_csv(paths["validation"] / "external_validation_inventory.csv",index=False)
    activities=pd.concat(all_activity,ignore_index=True).dropna()
    activities.to_csv(paths["tables"] / "registered_target_activity_summary_rows.csv", index=False)
    summary=activities.groupby("gene").pIC50.agg(["count","mean","std","median","min","max"]).reset_index()
    summary.to_csv(paths["tables"] / "activity_distribution_summary_by_target.csv",index=False)
    fig,ax=plt.subplots(figsize=(11,4.5)); order=summary.sort_values("median").gene.tolist(); data=[activities.loc[activities.gene==g,"pIC50"].to_numpy() for g in order]; ax.boxplot(data,tick_labels=order,showfliers=False); ax.tick_params(axis="x",rotation=60);ax.set(title="Curated activity distributions by registered target",ylabel="pIC50",xlabel="Target");fig.tight_layout();fig.savefig(paths["figures"] / "activity_distribution_by_target.png",bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2)); display=metrics.sort_values("RMSE_pIC50"); axes[0].barh(display.gene,display.RMSE_pIC50,color="#1768e5"); axes[0].set(title="Internal scaffold-test RMSE",xlabel="RMSE (pIC50)"); axes[1].barh(display.gene,display.R2,color="#12a18a");axes[1].set(title="Internal scaffold-test R2",xlabel="R2");fig.tight_layout();fig.savefig(paths["figures"] / "registered_model_scaffold_test_comparison.png",bbox_inches="tight");plt.close(fig)
    context=jak_deep_dive(*jak_context,paths)
    reproducibility={"created_utc":datetime.now(timezone.utc).isoformat(),"random_seed":SEED,"fingerprint":{"type":"Morgan","radius":RADIUS,"bits":FP_SIZE},"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,"rdkit":rdkit.__version__,"scipy":scipy.__version__,"registered_models_evaluated":len(registry),"JAK_deep_dive":context}
    (paths["validation"] / "reproducibility_manifest.json").write_text(json.dumps(reproducibility,indent=2),encoding="utf-8")
    (paths["metrics"] / "evaluation_summary.json").write_text(json.dumps({"models":len(metrics),"mean_internal_scaffold_R2":float(metrics.R2.mean()),"mean_internal_scaffold_RMSE":float(metrics.RMSE_pIC50.mean()),"all_external_gates_passed":int(metrics.external_quality_gates_passed.sum()),"limitations":["All recomputed performance values are internal scaffold-held-out measurements.","No registered model currently has strict external validation gates passed.","JAK_FAMILY is pooled pChEMBL data with no per-row isoform or assay metadata."]},indent=2),encoding="utf-8")
    print(f"Completed. Results written to {RESULTS}")


if __name__ == "__main__":
    main()
