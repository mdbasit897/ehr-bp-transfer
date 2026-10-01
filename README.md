# When Validation Fails for the Wrong Reasons

Code and aggregate results for:

> Azam MB, Singh SI. **When Validation Fails for the Wrong Reasons: Data Harmonization,
> Measurement Differences, and Cross-Institutional Transfer of EHR-Based Blood Pressure Models.**
> Manuscript under review. Preliminary version: arXiv:2507.19530 (superseded; see `legacy/`).

A blood pressure (BP) model trained on MIMIC-III appeared to fail completely in the eICU
database (external R² = −0.024). This repository shows, step by step, that the failure was
mostly produced by data extraction and harmonization, and that the remaining loss has
identifiable causes. The most important cause is a cuff-pressure measurement offset between
the databases: eICU cuff diastolic pressure reads 5.4 mmHg higher relative to the arterial
line, which matches the −5.0 mmHg transfer bias.

## Repository contents

| Path | Contents |
|---|---|
| `revision/` | Complete analysis pipeline: BigQuery SQL, harmonization, models, validation, shift analysis, figures and LaTeX tables |
| `configs/` | Run configurations (`revision.yaml` = main analysis) |
| `tests/` | Unit tests (12) |
| `results/` | Aggregate outputs of the final run: every number, table and figure in the paper |
| `docs/REVISION_CHANGES.md` | Errors found in the preliminary pipeline, and how each reviewer comment was addressed |
| `legacy/` | Preliminary analysis (arXiv v1), superseded, kept for transparency |

**No patient-level data or predictions are included.** MIMIC-III and eICU are
credentialed-access datasets. `results/` contains only aggregate statistics; hospital-level
results are shown without hospital identifiers or characteristics.

## Data access

1. Complete PhysioNet credentialing and sign the data use agreements for
   [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) and
   [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/).
2. Request access to both datasets on Google BigQuery through PhysioNet.
3. Set your Google Cloud billing project: `export GOOGLE_CLOUD_PROJECT=<your-project>`
   (or set `project_id` in `configs/revision.yaml`). The extraction scans more than 20 GB.

## Reproducing the analysis

```bash
pip install -r requirements.txt
python -m pytest tests -q                              # 12 tests, no data needed

# offline check of the whole pipeline on synthetic data (no data access needed)
python -m revision.run_revision synthetic --config configs/revision_smoke.yaml

# real data
python -m revision.run_revision sql --dataset eicu     # inspect the SQL first
python -m revision.run_revision extract                # BigQuery -> data_cache/ (patient-level; never commit)
python -m revision.run_revision analyze                # -> results/ (about 5-6 h; most of it is the stacked ensemble)

# figures and LaTeX tables only, from the CSVs in results/ (no data needed;
# the calibration figure, Fig. S2, needs the patient-level predictions and is not regenerated)
python -m revision.run_revision report
```

`configs/revision_fast.yaml` skips the stacked ensemble (under an hour);
`configs/revision_update.yaml` reruns only the stages added after the review audit.
Each run writes `results/run_manifest_<timestamp>.json` with the code version,
configuration and package versions.

Published results were produced with Python 3.8, numpy 1.24.4, pandas 2.0.3,
scikit-learn 1.3.2 and xgboost 2.1.4 (see `results/run_manifest.json`).

## Study design

- **Cohort:** first ICU stay per patient, age 18–89, still in the ICU at 24 h;
  MIMIC-III 17,962 stays, eICU 40,000 stays from 201 hospitals.
- **Task:** at 24 h after ICU admission, forecast mean cuff SBP, DBP and MAP and
  hypotension (≥2 hourly MAP values < 65 mmHg) during hours 24–48, from vital signs,
  laboratory values, medications and (optionally) BP history recorded in hours 0–24.
- **Validation:** MIMIC-III internal, eICU local, eICU unseen hospitals, MIMIC-III→eICU and
  eICU→MIMIC-III, with 95% cluster-bootstrap confidence intervals.

## Where each result comes from

| Paper | File in `results/` |
|---|---|
| Table 1 (cohort) | `cohort/table1.csv`, `cohort/flow.csv` |
| Table 2 (harmonization ladder) | `ladder/ladder_metrics.csv` |
| Table 3 (model comparison) | `models_transfer/metrics_ci.csv`, `models_transfer/paired_differences.csv` |
| Table 4 (gap decomposition) | `decomposition/gap_decomposition.csv` (`weights = linear`) |
| Table 5 (cuff vs arterial) | `modality_agreement/cuff_vs_arterial.csv` |
| Table 6 (recalibration) | `recalibration/summary.csv` |
| Table 7 (hospital random effects) | `hospitals/random_effects.csv` |
| Table 8 (feature attribution) | `hospitals/feature_shift_vs_loss.csv` |
| Table 9 (feature-set ablation) | `ablation/feature_set_ablation.csv` |
| Tables S1–S3 | `models_transfer/metrics_ci.csv` |
| Table S4 (tree-weight decomposition) | `decomposition/gap_decomposition.csv` (`weights = tree`) |
| Table S5 (importance weighting) | `domain_adaptation/summary.csv` |
| Table S6 (domain separability) | `shift/domain_auc_leave_one_out.csv` |
| Table S7 (complete cases) | `ablation/feature_set_ablation.csv` (`cohort = complete_case`) |
| Table S8 (imputation) | `imputation/imputation_sensitivity.csv` |
| Figures 2–8, S1–S3 | `figures/*.pdf` (Figure 1 is drawn in LaTeX) |
| LaTeX tables and figure captions | `tables/*.tex`, `figures/figures.tex` |

Data-quality checks are in `cohort/data_quality.csv` and `cohort/cross_dataset_checks.csv`.

## Citation

See `CITATION.cff`. Please also cite MIMIC-III (Johnson et al., *Sci Data* 2016),
eICU-CRD (Pollard et al., *Sci Data* 2018) and PhysioNet (Goldberger et al., *Circulation* 2000).

## License

Code: MIT License (`LICENSE`). Use of MIMIC-III and eICU is governed by the PhysioNet
credentialed data use agreement.
