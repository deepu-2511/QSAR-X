# QSAR Lab — MVP

The target-aware research platform and its current EGFR/BRAF model status are documented in [protein_qsar/README.md](protein_qsar/README.md). This Streamlit MVP remains a separate starter app.

A starter QSAR web application built with Streamlit.

## Features
- CSV/XLSX upload
- SMILES validation
- 10 RDKit molecular descriptors
- Random Forest, Extra Trees, Gradient Boosting, and SVR
- Held-out test-set R²/RMSE/MAE
- 5-fold cross-validation
- Single-SMILES activity prediction
- Descriptor CSV export

## Run
1. Install Python 3.10–3.12.
2. Create/activate a virtual environment.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start:
   `streamlit run app.py`

## Dataset format
At minimum:
- one SMILES column
- one numeric activity column

Example:
SMILES,Activity
CCO,5.2
CCCO,5.8
c1ccccc1,4.1

## Important
This is an MVP, not a validated scientific production platform. For publication-grade QSAR, add:
- endpoint/unit standardization (e.g. pIC50)
- duplicate/assay-conflict handling
- scaffold-aware splitting
- Y-randomization/permutation tests
- applicability-domain methods
- uncertainty estimation
- external validation set
- descriptor selection done inside CV to avoid leakage
- richer 2D/3D descriptors and fingerprints
- model interpretability
- reproducible experiment logging
