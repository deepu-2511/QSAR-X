# QSAR-X: protein-targeted QSAR workflow

QSAR-X accepts a protein target and compound SMILES, resolves supported targets, checks model availability, predicts target-specific IC50/pIC50, and returns applicability/uncertainty status plus model warnings. It includes dataset collection and curation, scaffold-based model development, external-validation gates, versioned model reports, a FastAPI service, and a React dashboard.

## What is currently supported

| Target/model | Prediction | Evidence/status |
| --- | --- | --- |
| EGFR (`CHEMBL203`) | Enabled | Existing Extra Trees model preserved. Historical held-out set was reused during development (R² 0.670, RMSE 0.783 pIC50), so this is legacy validation, not an independent estimate. |
| BRAF (`CHEMBL5145`) | Enabled in research mode | Scaffold-held-out test R² 0.720, RMSE 0.656, MAE 0.483 pIC50. Exploratory BindingDB non-overlap check (n=203) R² 0.153, RMSE 1.169, MAE 0.967; assay relation/context are missing and gates fail. Treat each result as exploratory. |
| 10 additional kinase targets (ABL1, JAK2, CDK2, MAP2K1, MAPK14, AKT1, KDR, ERBB2, SRC, MET) | Research-only sequence model | The 12-target panel contains 60,032 curated target-compound records. Its leave-one-target-out RMSE is 1.261 pIC50 and Spearman is 0.295, so it fails promotion gates and cannot be called commercial-grade. |
| Other proteins | Research sequence input | A raw protein sequence plus SMILES can use the 12-target candidate through `/predict-general`. Results are explicitly research-only. Target-specific assay data and independent validation are required before a model is enabled as validated. |

There is no scientifically valid way to guarantee a highly accurate QSAR prediction for every arbitrary protein from only a protein sequence and a SMILES. The target needs compatible experimental activity data, endpoint/unit and assay-context curation, target-specific or sufficiently broad training, and independent evaluation. For an unregistered or untrained protein, QSAR-X rejects prediction instead of returning a misleading EGFR/BRAF result.

## Start the web application

Use a Python 3.10–3.12 environment with the packages in `backend/requirements.txt` and Node.js installed. From `protein_qsar/`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

Terminal 1, backend:

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2, frontend:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`. The frontend contacts the API at `http://127.0.0.1:8000`; keep both servers running. API status and schema are at `http://127.0.0.1:8000/health` and `/docs`.

## Hosted deployment

The repository includes a Docker deployment definition. It builds the React frontend, serves it through the FastAPI service, and contains compressed runtime copies of every registered model under `models_deploy/`. The source artifacts under `models/` are intentionally excluded from the image and Git repository because several exceed GitHub's individual-file limit.

Render can deploy the repository directly from `render.yaml`:

1. In Render, choose **New > Blueprint** and select the GitHub repository.
2. Confirm the `qsar-x` web service and deploy it.
3. Open the generated `https://qsar-x-...onrender.com` service URL. The dashboard and API are served from that same URL.

Docker can also run locally:

```powershell
cd C:\Users\MANEESTHILAGAR\OneDrive\Desktop\QSAR\protein_qsar
docker build -t qsar-x .
docker run --rm -p 8000:8000 qsar-x
```

Then open `http://127.0.0.1:8000`.

Input accepts a registered gene, ChEMBL ID, UniProt accession, target name, or protein sequence, plus a valid SMILES. EGFR and BRAF use their registered target-specific models. A raw sequence uses the 12-target research candidate, whose output must not be treated as a validated assay result.

## Build the 12-target kinase research panel

Commands below run from `protein_qsar/`. The collected originals and reports are preserved by target under `data/targets/`.

```powershell
python scripts\build_kinase_panel.py
python scripts\build_multitarget_features.py --input data\common\kinase_panel_ic50.csv --output data\common\kinase_panel_features.npz
python scripts\train_multitarget.py --features data\common\kinase_panel_features.npz --max-rows-per-target 1500
python scripts\register_multitarget_candidate.py baseline_v2 --allow-research-predictions
```

This creates a scaffold-grouped multi-target candidate under `models/multitarget/baseline_vN/` and writes a feature manifest, split predictions, model comparison, leave-one-target-out metrics, and model hash. `baseline_v2` is activated only for research sequence predictions because both unseen-target quality gates fail.

## Add a ChEMBL single-protein target

Collection supports exact ChEMBL IC50 measurements in nM. The baseline requires at least 100 curated compounds and 10 scaffolds.

```powershell
python scripts\target_pipeline.py collect CHEMBL1234
python scripts\target_pipeline.py curate CHEMBL1234
python scripts\train_target_baseline.py --target-id CHEMBL1234
```

Training chooses the next `baseline_vN` without overwriting older versions. Register that exact version as a candidate (the registry keeps predictions disabled until evaluation data is attached):

```powershell
python scripts\register_target.py CHEMBL1234 baseline_v1
```

Replace `baseline_v1` with the version printed by training. Prepare an independent CSV with one record per external measurement and these columns:

`target_chembl_id,target_uniprot_accession,canonical_smiles,activity_nM,standard_type,standard_units,relation,assay_context,publication_id`

Include the source assay conditions and publication identifiers. The validator canonicalizes structures, removes exact structure overlaps with training, computes metrics, and fails closed if provenance, relation, assay context, sample size, metrics, or model hash do not pass.

```powershell
python scripts\validate_external_target.py CHEMBL1234 baseline_v1 `
  --external-csv .\external_ic50.csv `
  --source-name "Independent assay collection" `
  --source-url "https://example.org/source"
python scripts\quality_gate.py CHEMBL1234 `
  --model-version baseline_v1 `
  --external-report models\CHEMBL1234\baseline_v1\external_validation_report.json `
  --apply-registry
```

When automated gates fail, the model can only be marked as a warned research candidate; the quality-gate command exits with code 2. Automated success still leaves the model in research mode. After an authorized reviewer has checked the assay provenance and quality report, add `--approve-reviewed-model --reviewed-by "reviewer name or ID"` to `quality_gate.py` to record that review and enable validated mode. Never apply that option to a failed gate report.

For BRAF, `scripts/validate_external_braf.py` records the current BindingDB REST comparison. The endpoint supplies no relation operator or sufficient assay context; its metrics also miss the configured screening floor. So BRAF correctly remains runnable in research mode with warnings, and its current model does not pass the promotion gate.

## API and reports

- `GET /targets` — registered targets and model status.
- `POST /resolve-target` — resolve a registered gene/name/accession/ChEMBL ID or exact stored sequence.
- `POST /predict` — `{"protein_target":"BRAF","smiles":"CCO"}` or registered `target_id` plus `smiles`.
- `POST /predict-general` — `{"protein_sequence":"...","smiles":"CCO"}` for the explicitly marked research sequence model.
- `POST /predict-batch` — multipart CSV with a SMILES column and `target_id` form value; up to 1,000 rows / 5 MB.
- `GET /models/{target_id}/status` — artifact and validation status.
- `GET /reports/{target_id}` — evaluation, external-validation, and gate reports.
- `POST /curate-dataset` — validate/curate an uploaded activity CSV.

The dashboard exposes model status and JSON reports for the selected target. Candidate predictions return a research warning, dataset-level error estimate, and similarity; the external/test RMSE is not a per-compound uncertainty interval. Applicability thresholds are not calibrated for BRAF or future candidate targets.

## Reproducibility and interpretation

- `models/target_registry.json` identifies the target and current model version; each trained version stores its model artifact, data hash, scaffold assignments, comparison metrics, and evaluation report.
- The EGFR model and its historical input splits remain preserved. New target models are versioned and do not overwrite them.
- ChEMBL activity curation retains exact positive IC50 nM records, standardizes pIC50, canonicalizes structures, and aggregates repeated canonical structures by median. Assay-condition harmonization remains a manual review task.
- Scaffold test performance is internal validation. It is not a substitute for an independently sourced, context-matched external set or prospective experiments.
- pIC50/IC50 values are model estimates, not experimental measurements or clinical guidance.

## Recompute the scientific evaluation and reports

The evaluator preserves every registered candidate model. It reloads each saved model and its saved scaffold-test assignment, then writes recomputed predictions, residuals, extended regression metrics, bootstrap intervals, quality-control audit tables, figures, external-evidence inventory, and a reproducibility manifest under `results/`.

```powershell
cd C:\Users\MANEESTHILAGAR\OneDrive\Desktop\QSAR\protein_qsar
python scripts\research_evaluation.py
python scripts\generate_research_report.py
python scripts\generate_research_pdf.py
```

The evaluator includes a focused JAK pooled-model analysis with RDKit descriptor distributions, descriptor correlation, native tree feature importance, and an empirical nearest-neighbour Morgan-similarity applicability-domain sample. It does not create an isoform-specific JAK1/JAK2/JAK3 claim and does not treat the pooled source as an independent IC50 validation set.

Main machine-readable outputs:

- `results/metrics/registered_target_scaffold_test_metrics.csv`
- `results/tables/dataset_quality_audit.csv`
- `results/validation/external_validation_inventory.csv`
- `results/predictions/*_scaffold_test_predictions_recomputed.csv`
- `results/validation/reproducibility_manifest.json`
- `reports/QSAR_Research_Report.md`, `.docx`, and `.pdf`
