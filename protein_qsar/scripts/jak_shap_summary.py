"""Compute a held-out SHAP summary for the pooled JAK Extra Trees model."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from research_evaluation import ROOT, fingerprint

base=ROOT/'models'/'JAK_FAMILY'/'baseline_v1'; model=joblib.load(base/'candidate_model.joblib')
test=pd.read_csv(base/'split_assignments.csv').query("split == 'scaffold_test'").sample(n=50,random_state=42)
x=np.stack([fingerprint(s) for s in test.canonical_smiles])
values=shap.TreeExplainer(model).shap_values(x)
mean_abs=np.mean(np.abs(values),axis=0); top=np.argsort(mean_abs)[::-1][:20]
out=pd.DataFrame({'fingerprint_bit':top,'mean_abs_SHAP_on_heldout_50':mean_abs[top]})
target=ROOT/'results'/'interpretability';target.mkdir(parents=True,exist_ok=True);out.to_csv(target/'JAK_SHAP_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(7,6));draw=out.iloc[::-1];ax.barh(draw.fingerprint_bit.astype(str),draw.mean_abs_SHAP_on_heldout_50,color='#1768e5');ax.set(title='JAK held-out sample: SHAP feature importance',xlabel='Mean absolute SHAP value (pIC50)',ylabel='Morgan fingerprint bit');fig.tight_layout();fig.savefig(target/'JAK_SHAP_summary.png',dpi=300,bbox_inches='tight')
print(out.head().to_string(index=False))
