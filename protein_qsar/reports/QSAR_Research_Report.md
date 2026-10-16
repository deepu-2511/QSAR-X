# Machine Learning Based QSAR Modeling for Kinase Drug Discovery

Generated: 2026-10-16

## Abstract

This report evaluates 21 registered molecular activity models in QSAR-X using the saved model artifacts and their recorded scaffold-held-out test partitions. The registered panel contains 20 target-specific ChEMBL kinase models and one pooled JAK-family screening model. The recomputed mean internal scaffold-test R2 was 0.627 and mean RMSE was 0.717 pIC50. These are regression performance measures, not biological or clinical accuracy. Strict external validation gates passed for 0 of 21 registered models. Therefore the models are appropriate only for research screening and prioritization.

## Methods

Canonical molecular structures were represented with RDKit Morgan fingerprints (radius 2, 2048 bits). Existing candidate models and their saved split assignments were preserved. The independent partition in this report is a scaffold-held-out internal test, defined by the project’s chirality-aware Bemis-Murcko grouping. Recomputed predictions were used for R2, RMSE, MAE, median absolute error, explained variance, Pearson r, Spearman rho, residual diagnostics, and 300-resample bootstrap confidence intervals for selected metrics. No external-validation claim is made unless the strict external gate report passes.

## Registered Model Performance

| Gene | Model | n test | R2 | RMSE pIC50 | MAE pIC50 | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| ABL1 | SVR | 402 | 0.798 | 0.683 | 0.512 | 0.874 |
| AKT1 | SVR | 619 | 0.668 | 0.731 | 0.566 | 0.804 |
| BRAF | SVR | 1265 | 0.720 | 0.656 | 0.483 | 0.855 |
| CDK2 | SVR | 615 | 0.569 | 0.843 | 0.638 | 0.758 |
| EGFR | ExtraTrees | 2368 | 0.552 | 0.850 | 0.635 | 0.733 |
| ERBB2 | ExtraTrees | 474 | 0.629 | 0.741 | 0.571 | 0.801 |
| FGFR1 | SVR | 610 | 0.695 | 0.657 | 0.505 | 0.838 |
| FLT3 | SVR | 691 | 0.608 | 0.783 | 0.607 | 0.774 |
| IGF1R | SVR | 467 | 0.769 | 0.593 | 0.410 | 0.872 |
| JAK | ExtraTrees | 1864 | 0.657 | 0.703 | 0.523 | 0.804 |
| JAK2 | ExtraTrees | 2017 | 0.716 | 0.696 | 0.513 | 0.827 |
| KDR | ExtraTrees | 1909 | 0.541 | 0.759 | 0.580 | 0.744 |
| KIT | ExtraTrees | 348 | 0.420 | 0.750 | 0.570 | 0.630 |
| MAP2K1 | SVR | 212 | 0.419 | 0.767 | 0.613 | 0.539 |
| MAPK14 | SVR | 1039 | 0.546 | 0.720 | 0.550 | 0.717 |
| MET | SVR | 784 | 0.629 | 0.647 | 0.490 | 0.809 |
| MTOR | SVR | 786 | 0.747 | 0.603 | 0.457 | 0.853 |
| PDGFRA | SVR | 104 | 0.409 | 0.798 | 0.615 | 0.557 |
| PIK3CA | SVR | 1279 | 0.715 | 0.615 | 0.466 | 0.836 |
| RET | SVR | 869 | 0.616 | 0.612 | 0.443 | 0.751 |
| SRC | ExtraTrees | 719 | 0.733 | 0.841 | 0.610 | 0.886 |

## JAK Family Deep Dive

The pooled JAK model used 1864 scaffold-test compounds in the recomputed evaluation (R2 0.657, RMSE 0.703, MAE 0.523). It predicts a pooled pChEMBL activity endpoint. It cannot establish JAK1, JAK2, or JAK3 isoform selectivity because the source rows do not contain a per-row isoform label. The current JAK source also overlaps the locally stored JAK2 training chemistry, therefore it is not an independent external test.

### Random versus scaffold diagnostic

| Split | R2 | RMSE pIC50 | MAE pIC50 |
|---|---:|---:|---:|
| Recorded scaffold-held-out | 0.657 | 0.703 | 0.523 |

The JAK test-set applicability-domain similarity threshold is the empirical 5th percentile of nearest development-set similarity (0.627); this is a descriptive internal threshold, not a calibrated error guarantee.

## Dataset Quality and Leakage Audit

Every registered clean input was checked for missing pIC50, duplicate canonical structures, RDKit parsing failures, and exact canonical-structure overlap between development and scaffold-test partitions. All split-overlap counts in the audit table are expected to be zero. The source files retain their curation reports; clean datasets are not silently altered by this evaluator.

## External Validation Status

All models have strict external gates marked false or lack a qualifying external set. Some project reports contain exploratory external-source checks, but missing assay relation, provenance, context, sample-size, independence, or performance conditions prevent promotion. Internal scaffold test performance must not be called external validation.

## Reproducibility

Random seed: 42. Python: 3.14.2; RDKit: 2026.03.6; scikit-learn: 1.9.1; NumPy: 2.5.3; pandas: 3.0.6. The evaluator stores recomputed predictions, residuals, figures, model SHA256 values, and all tables under results/.

## Limitations

- The target-specific models use molecular fingerprints and do not infer protein binding mechanisms.
- Scaffold-held-out tests remain internal validation.
- Prediction errors are in pIC50 and are not individual compound confidence intervals.
- A high pIC50 means higher potency only for comparable assay conditions; it does not predict safety, efficacy, or clinical success.
- The pooled JAK model should be used for research screening, not isoform-specific or experimentally verified IC50 claims.

## References

1. OECD. Guidance Document on the Validation of (Quantitative) Structure-Activity Relationship [(Q)SAR] Models. OECD Series on Testing and Assessment No. 69, 2014. https://doi.org/10.1787/9789264085442-en
2. RDKit documentation. The RDKit Book: Morgan and Feature Morgan fingerprints. https://www.rdkit.org/docs/RDKit_Book.html
3. Rogers D, Hahn M. Extended-connectivity fingerprints. Journal of Chemical Information and Modeling. 2010;50:742-754. https://doi.org/10.1021/ci100050t
