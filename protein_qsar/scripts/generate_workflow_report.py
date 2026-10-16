from pathlib import Path
import json, math
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'pdf'/'QSAR-X_Workflow_and_Validation_Report.pdf'
registry=json.loads((ROOT/'models'/'target_registry.json').read_text(encoding='utf-8'))['targets']

NAVY=colors.HexColor('#10233E'); BLUE=colors.HexColor('#1768E5'); PALE=colors.HexColor('#EEF5FF'); INK=colors.HexColor('#182436'); MUTED=colors.HexColor('#607089'); LINE=colors.HexColor('#D9E2EC'); GREEN=colors.HexColor('#168463'); AMBER=colors.HexColor('#A96612'); RED=colors.HexColor('#B53B4A')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=27,leading=32,textColor=NAVY,spaceAfter=9))
styles.add(ParagraphStyle(name='SubtitleX',parent=styles['Normal'],fontSize=11,leading=17,textColor=MUTED,spaceAfter=16))
styles.add(ParagraphStyle(name='H1X',parent=styles['Heading1'],fontName='Helvetica-Bold',fontSize=18,leading=23,textColor=NAVY,spaceBefore=12,spaceAfter=9))
styles.add(ParagraphStyle(name='H2X',parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=12.5,leading=16,textColor=NAVY,spaceBefore=10,spaceAfter=6))
styles.add(ParagraphStyle(name='BodyX',parent=styles['BodyText'],fontSize=9.3,leading=14,textColor=INK,spaceAfter=7))
styles.add(ParagraphStyle(name='SmallX',parent=styles['BodyText'],fontSize=7.8,leading=10,textColor=MUTED))
styles.add(ParagraphStyle(name='CalloutX',parent=styles['BodyText'],fontSize=9,leading=13,textColor=INK,backColor=PALE,borderColor=colors.HexColor('#BDD5F5'),borderWidth=.7,borderPadding=10,spaceBefore=6,spaceAfter=10))
styles.add(ParagraphStyle(name='FootX',parent=styles['Normal'],fontSize=7.5,textColor=MUTED,alignment=TA_CENTER))

P=lambda text,style='BodyX': Paragraph(text,styles[style])
def fmt(x, digits=3): return ' - ' if x is None else f'{float(x):.{digits}f}'
def flow_table(data,widths,head=True,small=True):
    table=Table(data,colWidths=widths,repeatRows=1 if head else 0,hAlign='LEFT')
    ts=[('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.35,LINE),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]
    if head: ts += [('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),7.4)]
    ts += [('FONTSIZE',(0,1 if head else 0),(-1,-1),7.2 if small else 8),('BACKGROUND',(0,1 if head else 0),(-1,-1),colors.white),('ROWBACKGROUNDS',(0,1 if head else 0),(-1,-1),[colors.white,colors.HexColor('#F8FAFC')])]
    table.setStyle(TableStyle(ts)); return table

def header_footer(canvas,doc):
    canvas.saveState(); w,h=A4
    canvas.setStrokeColor(LINE); canvas.line(18*mm,h-15*mm,w-18*mm,h-15*mm)
    canvas.setFont('Helvetica-Bold',7.5); canvas.setFillColor(NAVY); canvas.drawString(18*mm,h-11*mm,'QSAR-X  |  Workflow and Validation Report')
    canvas.setFont('Helvetica',7.2); canvas.setFillColor(MUTED); canvas.drawRightString(w-18*mm,h-11*mm,'Generated 15 October 2026')
    canvas.line(18*mm,13*mm,w-18*mm,13*mm); canvas.drawCentredString(w/2,8.5*mm,f'Page {doc.page}')
    canvas.restoreState()

story=[]
story += [Spacer(1,20*mm), P('QSAR-X', 'TitleX'), P('Target-specific QSAR workflow, training, validation, and deployment report','SubtitleX')]
story += [P('Scope', 'H1X'), P('This report documents the current local QSAR-X project from raw assay retrieval through the web application. It covers the 20-target kinase panel, the EGFR model in detail, model artifacts, validation evidence, prediction output, and the present release status.', 'BodyX')]
story += [P('<b>Current release state:</b> 20 individual target-specific models are registered and prediction-enabled for screening. No model is labelled externally validated because the strict evidence gates have not yet all passed. This distinction is deliberate and prevents overstatement of model performance.', 'CalloutX')]
story += [P('Executive inventory','H2X')]
common=ROOT/'data'/'common'/'kinase_panel_ic50.csv'
# known total captured from data local
story.append(flow_table([
    [P('Item','SmallX'),P('Current value','SmallX'),P('Evidence / location','SmallX')],
    [P('Target models','SmallX'),P(str(len(registry)),'SmallX'),P('models/target_registry.json','SmallX')],
    [P('Curated panel rows','SmallX'),P('87,095','SmallX'),P('data/common/kinase_panel_ic50.csv','SmallX')],
    [P('Endpoint','SmallX'),P('IC50 normalized to pIC50','SmallX'),P('pIC50 = 9 - log10(IC50 nM)','SmallX')],
    [P('Chemical representation','SmallX'),P('Morgan fingerprint, radius 2, 2048 bits','SmallX'),P('RDKit feature generation','SmallX')],
    [P('Selected model family','SmallX'),P('Extra Trees candidate, target-specific','SmallX'),P('300 estimators; per-target training report','SmallX')],
    [P('Prediction interfaces','SmallX'),P('Single SMILES and batch CSV','SmallX'),P('FastAPI backend and Vite frontend','SmallX')],
], [38*mm,48*mm,88*mm]))
story.append(PageBreak())

story += [P('1. End-to-end workflow','H1X')]
steps=[
('1. Target panel definition','A 20-member human kinase panel was defined with ChEMBL target identifiers, target metadata, UniProt accession, and local FASTA sequence files.'),
('2. Assay retrieval','The ChEMBL activity service was queried per target. IC50 records were retained only when the target was a direct single protein, the activity was numeric and positive, units were nM, and the relation was exact (=). Raw and cleaned target datasets were stored locally.'),
('3. Curation and standardization','SMILES were canonicalized with RDKit. Measurements were converted to pIC50. Duplicate structures were aggregated using the median activity. Target-specific datasets are retained in data/targets/CHEMBL*/IC50_clean.csv.'),
('4. Feature engineering','Each compound is represented by an RDKit Morgan circular fingerprint (radius 2; 2,048 bits). The individual target models use this chemical representation. Protein sequences and broader multitarget experimentation are retained separately but do not replace target-specific models.'),
('5. Training and selection','For every target, chirality-aware Bemis-Murcko scaffolds were used to separate training, validation, and scaffold-held-out test compounds. Median baseline, Extra Trees, Random Forest, and SVR were compared by training CV RMSE. The selected candidate is stored with its version, split assignments, evaluation report, and serialized model.'),
('6. External checks','BindingDB REST records were pulled using the target UniProt accession. Exact chemical overlap with the ChEMBL training set was removed. The reports are kept as external-source evidence; they do not automatically promote any model because BindingDB REST omits relation operators and complete assay provenance.'),
('7. Independent PubChem search','PubChem BioAssay PUG-REST was used to discover non-ChEMBL IC50 assays. Candidate records must provide explicit IC50, unit, exact relation, SMILES, assay context and PubMed provenance before ingestion. The JAK2 AID 1699 extraction produced 105 rows; 95 overlapped training structures, leaving 10 independent structures, below the 30-compound minimum gate.'),
('8. Deployment','The FastAPI service loads the versioned registry; the Vite frontend provides target selection, SMILES prediction, batch CSV input, CSV exports, and browser-local history export.'),
]
for title,body in steps: story += [P(title,'H2X'),P(body,'BodyX')]
story.append(PageBreak())

story += [P('2. Data sources, downloaded artifacts, and traceability','H1X')]
story += [P('The system uses public scientific sources. Data were saved as local immutable-ish project artifacts with target metadata, raw activities, curated activities, protein sequence, model evaluation, and external evidence files. The workflow never creates IC50 values synthetically.', 'BodyX')]
story.append(flow_table([
 [P('Source','SmallX'),P('Use in QSAR-X','SmallX'),P('Stored evidence','SmallX')],
 [P('ChEMBL','SmallX'),P('Training IC50 assay data and target metadata for each target.','SmallX'),P('data/targets/CHEMBL*/target_metadata.json; IC50_raw.csv; IC50_clean.csv','SmallX')],
 [P('UniProt via target metadata','SmallX'),P('Protein accession and FASTA identity for target resolution.','SmallX'),P('data/targets/CHEMBL*/protein.fasta','SmallX')],
 [P('BindingDB REST','SmallX'),P('External-source non-overlap check by UniProt accession.','SmallX'),P('models/CHEMBL*/baseline_v*/external_validation_report.json','SmallX')],
 [P('PubChem BioAssay PUG-REST','SmallX'),P('Discovery and curation of independent IC50 assay candidates.','SmallX'),P('data/external_validation/pubchem_assay_inventory.json; pubchem_independent/','SmallX')],
], [35*mm,66*mm,73*mm]))
story += [P('Curation rules','H2X'), P('The training path accepts IC50 endpoint values in nM with an exact relation and a positive numeric value. pIC50 is calculated as 9 - log10(IC50 nM). Canonical SMILES make duplicate detection and overlap removal structure-aware. The report records source hashes in individual evaluation reports so an artifact can be linked to its training dataset and model file.', 'BodyX')]
story += [P('Why external data cannot be reused blindly','H2X'),P('ChEMBL values are the training data, so they cannot serve as an independent test. BindingDB and PubChem are evaluated separately. Exact structures present in a training dataset are removed before external metrics are calculated. Assay endpoint, unit, relation, target accession, assay context, and publication provenance are checked before a validation result can be promoted.', 'BodyX')]
story.append(PageBreak())

story += [P('3. EGFR model dossier','H1X'),P('EGFR target: CHEMBL203 | UniProt: P00533 | Endpoint: IC50','SubtitleX')]
egfr=json.loads((ROOT/'models'/'CHEMBL203'/'baseline_v1'/'evaluation_report.json').read_text())
egfr_ext=json.loads((ROOT/'models'/'CHEMBL203'/'baseline_v1'/'external_validation_report.json').read_text())
story += [P('Training data and split','H2X')]
story.append(flow_table([
 [P('Measure','SmallX'),P('EGFR value','SmallX')],
 [P('Curated compounds','SmallX'),P(str(egfr['n_compounds']),'SmallX')], [P('Unique scaffolds','SmallX'),P(str(egfr['n_scaffolds']),'SmallX')],
 [P('Split counts','SmallX'),P(f"Train {egfr['split_counts']['train']}; validation {egfr['split_counts']['validation']}; scaffold test {egfr['split_counts']['scaffold_test']}",'SmallX')],
 [P('Fingerprint','SmallX'),P('Morgan radius 2, 2,048 bits','SmallX')], [P('Selected candidate','SmallX'),P(f"{egfr['selected_model_by_train_cv_rmse']} with 300 estimators",'SmallX')],
], [56*mm,118*mm]))
story += [P('EGFR internal model comparison','H2X')]
rows=[[P('Model','SmallX'),P('CV R2','SmallX'),P('CV RMSE','SmallX'),P('CV MAE','SmallX')]]
for r in egfr['cv_model_comparison']: rows.append([P(r['model'],'SmallX'),P(fmt(r['CV_R2_mean']),'SmallX'),P(fmt(r['CV_RMSE_mean']),'SmallX'),P(fmt(r['CV_MAE_mean']),'SmallX')])
story.append(flow_table(rows,[48*mm,38*mm,44*mm,44*mm]))
story += [P('EGFR selected candidate performance','H2X')]
story.append(flow_table([
 [P('Evaluation','SmallX'),P('R2','SmallX'),P('RMSE pIC50','SmallX'),P('MAE pIC50','SmallX')],
 [P('Validation split','SmallX'),P(fmt(egfr['validation_metrics']['R2']),'SmallX'),P(fmt(egfr['validation_metrics']['RMSE']),'SmallX'),P(fmt(egfr['validation_metrics']['MAE']),'SmallX')],
 [P('Scaffold-held-out test','SmallX'),P(fmt(egfr['scaffold_test_metrics']['R2']),'SmallX'),P(fmt(egfr['scaffold_test_metrics']['RMSE']),'SmallX'),P(fmt(egfr['scaffold_test_metrics']['MAE']),'SmallX')],
], [52*mm,35*mm,44*mm,43*mm]))
story += [P('EGFR external-source evidence','H2X')]
em=egfr_ext['metrics_on_external_source_nonoverlap']
story.append(flow_table([
 [P('Measure','SmallX'),P('Value','SmallX')],
 [P('BindingDB exact-structure overlaps removed','SmallX'),P(str(egfr_ext['exact_structure_overlaps_removed']),'SmallX')],
 [P('Remaining external non-overlap compounds','SmallX'),P(str(em['n_compounds']),'SmallX')],
 [P('External RMSE / MAE / Spearman','SmallX'),P(f"{fmt(em['RMSE_pIC50'])} / {fmt(em['MAE_pIC50'])} / {fmt(em['Spearman'])}",'SmallX')],
 [P('Quality-gate result','SmallX'),P('Not passed - relation and assay context are absent in the BindingDB REST response; external metrics are also weak.','SmallX')],
], [63*mm,111*mm]))
story += [P('Interpretation','H2X'),P('EGFR has an internally scaffold-held-out candidate and a documented BindingDB external-source check. The external check is intentionally not labelled a validated model: its RMSE is 1.665 pIC50 and the source does not provide the complete relation and assay-context fields required by the local gate. The UI therefore exposes this model for screening with an explicit model status, not as commercial validation.', 'CalloutX')]
story.append(PageBreak())

story += [P('4. 20-target model inventory','H1X'),P('Each listed target has an individual baseline_v1 or later artifact. “External record” means an external-source report exists. It does not mean that the strict validation gates were passed.', 'BodyX')]
rows=[[P('Gene','SmallX'),P('ChEMBL ID','SmallX'),P('n','SmallX'),P('Scaffold RMSE','SmallX'),P('External n','SmallX'),P('External RMSE','SmallX'),P('Gate','SmallX')]]
for target in sorted(registry,key=lambda x:x['gene']):
    base=ROOT/target['evaluation_report']; ev=json.loads(base.read_text())
    extpath=ROOT/target['external_validation_report'] if target.get('external_validation_report') else None
    ex=json.loads(extpath.read_text()) if extpath and extpath.exists() else {}
    met=ex.get('metrics_on_external_source_nonoverlap',{})
    rows.append([P(target['gene'],'SmallX'),P(target['target_id'],'SmallX'),P(str(ev.get('n_compounds','-')),'SmallX'),P(fmt(ev.get('scaffold_test_metrics',{}).get('RMSE')),'SmallX'),P(str(met.get('n_compounds','-')),'SmallX'),P(fmt(met.get('RMSE_pIC50')),'SmallX'),P('PASS' if ex.get('quality_gates_passed') else 'Not passed','SmallX')])
story.append(flow_table(rows,[20*mm,26*mm,17*mm,30*mm,22*mm,31*mm,28*mm]))
story += [P('Model availability in the application','H2X'),P('All 20 models are registered in models/target_registry.json and are selectable in the frontend. A prediction response contains canonical SMILES, pIC50, IC50 nM, a dataset-level error estimate, nearest training similarity, model version, and a screening confidence label. Batch predictions accept a CSV with a smiles column and can be exported as a CSV. Single predictions and browser history can also be exported as CSV.', 'BodyX')]
story.append(PageBreak())

story += [P('5. Validation protocol and present status','H1X')]
story += [P('Automated quality gates','H2X')]
gates=[
('Target identity','External target accession must match the expected UniProt accession.'),
('Endpoint and units','The dataset must be IC50 and normalized to nM.'),
('Relation','Only exact = values can enter the strict external validator.'),
('Assay context and publication','Assay context and a publication identifier must be documented.'),
('Independence','Exact training-structure overlaps are removed; publication overlap is also checked when provenance is present.'),
('Sample size','At least 30 independent external compounds are required.'),
('Performance floor','RMSE <= 1.0 pIC50, MAE <= 0.75 pIC50, Spearman >= 0.50.'),
('Reproducibility','Model hash must match the candidate hash recorded at training time.'),
]
story.append(flow_table([[P('Gate','SmallX'),P('Requirement','SmallX')]]+[[P(a,'SmallX'),P(b,'SmallX')] for a,b in gates],[49*mm,125*mm]))
story += [P('Current status','H2X'),P('No model currently passes every strict external validation gate. This is not a training failure: all 20 target-specific artifacts were trained and registered. It means the current independent evidence is incomplete, has missing source fields, insufficient remaining non-overlap compounds, or does not meet the predefined metric thresholds. The release must be described as screening models with recorded external evidence, rather than as commercially validated predictors.', 'CalloutX')]
story += [P('Recommended route to validated status','H2X'),P('For each target: collect an assay-level independent dataset with IC50, exact relation, nM units, target accession, documented protocol/context and publication ID; exclude all training structures and publication overlaps; retain at least 30 compounds; run the versioned external validator; inspect the prediction residuals and source provenance; then conduct an expert review before changing the registry status. A prospective wet-lab test remains the strongest production claim.', 'BodyX')]
story += [P('6. Software and reproducibility','H1X'),P('Backend: FastAPI. Frontend: React/Vite. Chemical processing: RDKit. Model training: scikit-learn Extra Trees, Random Forest, and SVR comparison. Python runtime recorded in training reports: 3.12.14; RDKit 2026.09.1; scikit-learn 1.9.1. Each candidate folder stores candidate_model.joblib, evaluation_report.json, model comparison CSV, and split assignments.', 'BodyX')]
story += [P('Project entry points','H2X'), P('scripts/build_kinase_panel.py builds the panel. scripts/train_panel_targets.py trains the individual targets. scripts/validate_bindingdb_target.py creates non-promotional BindingDB evidence reports. scripts/discover_pubchem_ic50_assays.py discovers PubChem candidate assays. scripts/extract_pubchem_independent_ic50.py filters eligible independent PubChem records. scripts/validate_external_target.py applies strict external quality gates. scripts/sync_target_registry.py registers the local target candidates for the API.', 'BodyX')]
story += [Spacer(1,8*mm),P('End of report','FootX')]

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=21*mm,bottomMargin=18*mm,title='QSAR-X Workflow and Validation Report',author='QSAR-X')
doc.build(story,onFirstPage=header_footer,onLaterPages=header_footer)
print(OUT)
