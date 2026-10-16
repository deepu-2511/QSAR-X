import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="QSAR Lab", page_icon="🧬", layout="wide")

st.title("🧬 QSAR Lab")
st.caption("MVP: molecular descriptors + machine-learning QSAR + validation + single-compound prediction")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit.Chem import AllChem
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.feature_selection import VarianceThreshold
except Exception as e:
    st.error("Missing dependencies. Run: pip install -r requirements.txt")
    st.stop()

def descriptors(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }

def make_features(df, smiles_col):
    rows, valid = [], []
    for i, s in enumerate(df[smiles_col]):
        d = descriptors(s)
        if d is not None:
            rows.append(d)
            valid.append(i)
    feat = pd.DataFrame(rows, index=valid)
    return feat, valid

st.sidebar.header("1. Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded is None:
    st.info("Upload a dataset to begin. Your file should contain a SMILES column and one numeric activity column.")
    st.code("SMILES,Activity\nCCO,5.2\nCCCO,5.8\nc1ccccc1,4.1")
    st.stop()

if uploaded.name.lower().endswith(".csv"):
    raw = pd.read_csv(uploaded)
else:
    raw = pd.read_excel(uploaded)

st.subheader("Dataset preview")
st.dataframe(raw.head(10), use_container_width=True)

smiles_col = st.sidebar.selectbox("SMILES column", raw.columns)
numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and c != smiles_col]
if not numeric_cols:
    st.error("No numeric activity column found.")
    st.stop()
target_col = st.sidebar.selectbox("Activity column", numeric_cols)

st.sidebar.header("2. Model settings")
test_size = st.sidebar.slider("Test-set fraction", 0.10, 0.40, 0.20, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

features, valid_idx = make_features(raw, smiles_col)
work = raw.loc[valid_idx].copy()
y = pd.to_numeric(work[target_col], errors="coerce")
mask = y.notna()
work = work.loc[mask]
features = features.loc[work.index]
y = y.loc[work.index]

st.write(f"Valid molecules: **{len(work)} / {len(raw)}**")
if len(work) < 20:
    st.warning("Very small datasets can produce unstable QSAR validation. Use a larger, experimentally consistent dataset when possible.")

X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=test_size, random_state=int(seed)
)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=400, random_state=int(seed), n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=400, random_state=int(seed), n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=int(seed)),
    "SVR": SVR(C=10, epsilon=0.1, gamma="scale"),
}

results = []
fitted = {}

for name, model in models.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    results.append({
        "Model": name,
        "Test R²": r2_score(y_test, pred),
        "Test RMSE": rmse,
        "Test MAE": mean_absolute_error(y_test, pred),
    })
    fitted[name] = pipe

results_df = pd.DataFrame(results).sort_values("Test RMSE")

st.subheader("Model comparison")
st.dataframe(results_df.style.format({
    "Test R²": "{:.3f}", "Test RMSE": "{:.3f}", "Test MAE": "{:.3f}"
}), use_container_width=True)

best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
st.success(f"Best model on this held-out test split by RMSE: {best_name}")

st.subheader("5-fold cross-validation")
cv = KFold(n_splits=5, shuffle=True, random_state=int(seed))
cv_rows = []
for name, model in models.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("model", model),
    ])
    scores = cross_val_score(pipe, features, y, cv=cv, scoring="r2")
    cv_rows.append({"Model": name, "CV R² mean": scores.mean(), "CV R² SD": scores.std()})
cv_df = pd.DataFrame(cv_rows).sort_values("CV R² mean", ascending=False)
st.dataframe(cv_df.style.format({"CV R² mean": "{:.3f}", "CV R² SD": "{:.3f}"}), use_container_width=True)

st.subheader("Descriptor table")
st.dataframe(pd.concat([work[[smiles_col, target_col]].reset_index(drop=True),
                        features.reset_index(drop=True)], axis=1),
             use_container_width=True)

st.subheader("Predict a new compound")
new_smiles = st.text_input("Enter SMILES", placeholder="Example: CCO")
if st.button("Predict activity"):
    d = descriptors(new_smiles)
    if d is None:
        st.error("Invalid SMILES.")
    else:
        one = pd.DataFrame([d])
        prediction = best_model.predict(one)[0]
        st.metric("Predicted activity", f"{prediction:.4f}")
        st.caption("This is a computational prediction. Check applicability domain and experimental validation before using it for scientific conclusions.")

st.subheader("Download generated descriptors")
csv = pd.concat([work[[smiles_col, target_col]].reset_index(drop=True),
                 features.reset_index(drop=True)], axis=1).to_csv(index=False).encode()
st.download_button("Download descriptor CSV", csv, "qsar_descriptors.csv", "text/csv")

st.warning("Research note: R² alone is not an 'accuracy' guarantee. Dataset quality, leakage control, external validation, applicability domain, endpoint consistency, and chemical-space coverage are essential.")
