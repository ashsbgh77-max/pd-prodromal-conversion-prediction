# Radiogenomic prediction of phenoconversion in prodromal Parkinson's disease

Code for the analyses in the manuscript *Cross-Validated Radiogenomic Prediction of Phenoconversion in Prodromal Parkinson's Disease*.

## Data
The data come from the Parkinson's Progression Markers Initiative (PPMI) and are **not included** here. Access is controlled: apply at https://www.ppmi-info.org/access-data-specimens/download-data and accept the PPMI Data Use Agreement. Place these files in a `data/` folder:
- `PPMI_Curated_Data_Cut_Public_20260810.xlsx` (sheet `20260810`)
- `iu_genetic_consensus_20251025_30Sep2026.csv` (LRRK2/GBA carrier status, used only in the secondary analysis)

## Run
```bash
pip install -r requirements.txt
python phenoconversion_pipeline.py        # or open phenoconversion_pipeline.ipynb
```
All results are written to `outputs/` (tables as CSV, figures as PNG).

## What the pipeline does
1. **Cohort**: baseline prodromal participants (`COHORT == 4`) with complete baseline predictors; time to the first later PD diagnosis; participant flow counts are printed and saved.
2. **Models**: penalised Cox models M1 to M7 (clinical, genetic, DaTSCAN and their combinations), 20 x 5-fold cross-validation with standardisation fitted on each training fold only.
3. **Comparison**: Nadeau-Bengio corrected resampled paired t-test, with Holm adjustment for the primary comparisons.
4. **Checks**: Schoenfeld residual test, variance inflation factors, sensitivity analysis without the striatum variables.
5. **Outputs**: hazard ratios, permutation importance (100 repeats, fixed seed), Kaplan-Meier curves by median risk split.
6. **Secondary analysis**: LRRK2/GBA carrier status, restricted to participants with carrier data.

## Notes
Random seeds are fixed (fold seed = repeat index; permutation seed = 42). Results can differ slightly with other library versions.
