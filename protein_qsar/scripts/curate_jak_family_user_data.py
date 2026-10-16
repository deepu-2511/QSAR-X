"""Curate the user-supplied JAK-family pChEMBL file for a pan-JAK screening candidate."""
from pathlib import Path
import hashlib, json
import pandas as pd
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r"C:\Users\MANEESTHILAGAR\OneDrive\Desktop\QSAR12\Data\compound.csv")
OUT=ROOT/'data'/'targets'/'JAK_FAMILY'
OUT.mkdir(parents=True,exist_ok=True)
def canon(x):
    if not isinstance(x,str): return None
    m=Chem.MolFromSmiles(x)
    return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) if m else None
raw=pd.read_csv(SOURCE,low_memory=False)
work=pd.DataFrame({'source_chembl_id':raw['ChEMBL ID'].astype(str),'canonical_smiles':raw['Smiles'].map(canon),'pIC50':pd.to_numeric(raw['pChEMBL Value'],errors='coerce')})
valid=work.dropna(subset=['canonical_smiles','pIC50']).copy()
valid=valid[(valid.pIC50>0)&(valid.pIC50<14)]
clean=valid.groupby('canonical_smiles',as_index=False).agg(pIC50=('pIC50','median'),source_measurement_count=('pIC50','size'),source_chembl_ids=('source_chembl_id',lambda x:'|'.join(sorted(set(x)))))
clean.insert(0,'target_chembl_id','JAK_FAMILY')
clean.to_csv(OUT/'JAK_family_pChEMBL_clean.csv',index=False)
raw.to_csv(OUT/'source_compound_raw.csv',index=False)
jak2=pd.read_csv(ROOT/'data'/'targets'/'CHEMBL2971'/'IC50_clean.csv',low_memory=False)
overlap=len(set(clean.canonical_smiles)&set(jak2.canonical_smiles.astype(str)))
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
metadata={'target_id':'JAK_FAMILY','gene':'JAK','name':'Janus kinase family (JAK1/JAK2/JAK3, user-supplied pooled dataset)','organism':'Homo sapiens','target_type':'family / pooled endpoint','endpoint':'pChEMBL value','source_file':str(SOURCE),'source_sha256':sha(SOURCE),'source_rows':int(len(raw)),'source_columns':list(raw.columns),'valid_measurements':int(len(valid)),'unique_structures':int(len(clean)),'duplicate_measurements_aggregated':int(len(valid)-len(clean)),'pchembl_range':[float(clean.pIC50.min()),float(clean.pIC50.max())],'known_overlap_with_current_JAK2_training_unique_structures':int(overlap),'overlap_fraction_of_pan_jak_structures':float(overlap/len(clean)),'limitations':['Source file has no per-row JAK1/JAK2/JAK3 target label, assay type, relation, unit, or provenance.','The endpoint is retained as user-supplied pChEMBL value; it must not be represented as an independently validated IC50 model.','All 10,070 unique structures overlap the existing local JAK2 training dataset, so this dataset cannot be used as an independent test.']}
(OUT/'target_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(metadata,indent=2))
