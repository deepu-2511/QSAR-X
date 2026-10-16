"""Held-out permutation importance for the 20 most important JAK fingerprint bits.

The target set is selected from native tree importance and evaluation uses a
fixed 100-compound scaffold-test subset. This is a focused diagnostic, not a
permutation test over every hashed fingerprint bit.
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from research_evaluation import ROOT, fingerprint

SEED=42
base=ROOT/'models'/'JAK_FAMILY'/'baseline_v1'
model=joblib.load(base/'candidate_model.joblib')
split=pd.read_csv(base/'split_assignments.csv'); test=split[split['split']=='scaffold_test'].copy()
rng=np.random.default_rng(SEED); take=rng.choice(len(test),size=100,replace=False)
x=np.stack([fingerprint(s) for s in test.iloc[take].canonical_smiles]); y=test.iloc[take].pIC50.to_numpy(float)
baseline=float(np.sqrt(mean_squared_error(y,model.predict(x))))
top=np.argsort(model.feature_importances_)[::-1][:20]
rows=[]
for bit in top:
 xp=x.copy(); xp[:,bit]=rng.permutation(xp[:,bit]); score=float(np.sqrt(mean_squared_error(y,model.predict(xp))))
 rows.append({'fingerprint_bit':int(bit),'native_tree_importance':float(model.feature_importances_[bit]),'baseline_RMSE_pIC50':baseline,'permuted_RMSE_pIC50':score,'RMSE_increase_after_permutation':score-baseline,'test_subset_n':100,'n_repeats':1})
out=pd.DataFrame(rows).sort_values('RMSE_increase_after_permutation',ascending=False)
target=ROOT/'results'/'interpretability';target.mkdir(parents=True,exist_ok=True);out.to_csv(target/'JAK_targeted_permutation_importance.csv',index=False)
fig,ax=plt.subplots(figsize=(7,6));draw=out.iloc[::-1];ax.barh(draw.fingerprint_bit.astype(str),draw.RMSE_increase_after_permutation,color='#7b3fc6');ax.set(title='JAK: targeted permutation importance on scaffold-test sample',xlabel='Increase in RMSE after bit permutation',ylabel='Morgan fingerprint bit');fig.tight_layout();fig.savefig(target/'JAK_targeted_permutation_importance.png',dpi=300,bbox_inches='tight')
print(out.head().to_string(index=False))
