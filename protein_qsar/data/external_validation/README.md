# External IC50 validation intake

Each target template requires independent IC50 records. Do not copy rows from the ChEMBL training CSV. Use exact `=` relations, nM units, canonical SMILES, a concise assay description (construct, assay format, conditions) in `assay_context`, and a publication or dataset identifier in `publication_id`.

At least 30 post-overlap-removal compounds are required. Validate a filled template with:

```powershell
python scripts\validate_external_target.py CHEMBLXXXX baseline_v1 `\n  --external-csv data\external_validation\CHEMBLXXXX_external_ic50.csv `\n  --source-name "Independent source name" `\n  --source-url "https://source.example"
```
