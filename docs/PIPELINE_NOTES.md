# Pipeline notes

This document records (A) the errors found in an earlier version of the analysis pipeline
and how `revision/` corrects them, and (B) the design requirements whose IDs (R1.1, R2.4, ...)
appear in code comments. In the code, labels such as "submitted task" (ladder step S0) and
"Stacked ensemble (submitted)" refer to that earlier pipeline; they are kept unchanged so that
the code, `results/` and the paper agree.

## A. Corrections to the earlier pipeline

The earlier pipeline's results correspond to ladder step S0 in the paper. Corrections A1–A15
were made when the pipeline was rebuilt; A16–A26 were found in later runs of the rebuilt pipeline.

| # | Earlier pipeline | Effect | Fix |
|---|---|---|---|
| A1 | eICU target from `vitalperiodic.systemicsystolic/diastolic` | eICU target = **arterial line only**; MIMIC target mostly cuff. Only patients with arterial lines kept (1,060/5,000) | Cuff BP read from `vitalaperiodic.noninvasive*`; modality is a config option and **matched** between datasets (`sql.py`, `analysis.target_modality`) |
| A2 | MIMIC item 443 used as DBP; 6701 used as MAP; 220050/220051 labelled non-invasive; CareVue NBP 455/8441/456 missing | Wrong DBP/MAP values; wrong modality counts | Item IDs follow mimic-code (`schema.MIMIC_VITAL_ITEMIDS`) |
| A3 | SBP kept only in 70-250 mmHg | Hypotensive values silently removed | Identical ranges for both DBs, SBP 50-250 (`schema.VITAL_RANGES`) |
| A4 | Single random 80/20 `train_test_split` over ICU stays, described as grouped 5-fold CV | Same patient in train and test; paper/code mismatch | First ICU stay per patient; patient-grouped K-fold (and hospital-grouped for eICU) |
| A5 | "22-feature baseline" actually trains on **30** features incl. 6 medications; eICU gets the MIMIC median for them | Description and code disagreed; medications constant in eICU | Features defined once in `schema.FEATURE_SETS`; medications extracted from eICU too (`infusiondrug`, `medication`) |
| A6 | `measurement_frequency` = count of hourly charted values (MIMIC) vs. 5-minute rows (eICU) | The "largest shift" (W = 427.8) is an artifact | Vitals averaged per hour in both DBs before any statistic; frequency feature removed |
| A7 | eICU temperature not unit-checked (SD 5.0 °C vs 0.85) | Fahrenheit values treated as Celsius | Same F/C rule and range for both DBs (`sql.temp_to_celsius`) |
| A8 | eICU gender constant (mean 0, SD 0 in `feature_distribution_shifts.csv`) | Feature useless / encoding bug | Robust mapping + data-quality check that errors on single-level gender |
| A9 | Labs joined on `hadm_id` = whole hospital admission | Labs after the prediction time used | Labs restricted to [-6 h, index time) by timestamp |
| A10 | Medication item IDs (e.g. 225154, 225792) not the intended drug classes | Wrong exposures | Drug classes matched by name (`d_items.label`, eICU `drugname`) in both DBs |
| A11 | `LIMIT 10000` without `ORDER BY` | Arbitrary, non-reproducible cohort | Full cohort, or a deterministic hash sample (`sample_permille`) |
| A12 | Recalibration fitted on a random 20% of eICU **rows** (described as "on MIMIC-III"), evaluated on a different subset (R² −0.170 vs −0.115) | Reported and computed values disagreed | Recalibration fitted on a subset of eICU **hospitals**, evaluated on the others, repeated; compared with a local model trained on the same data |
| A13 | Domain adaptation reweighted only the GB model and compared it with the ensemble | Unfair comparison | Same model with/without weights, same evaluation rows, repeated splits |
| A14 | Imputation described as "KNN + KS test" but implemented as median; KS never run; Yeo-Johnson/variance filter never applied; correlation leakage check absent | Description and code disagreed | Methods describe `revision/` exactly; imputation inside each training fold; imputation sensitivity analysis |
| A15 | eICU DBP transferred reasonably (R² = 0.233) but only SBP was reported | Selective reporting | All targets (SBP, DBP, MAP, hypotension) reported with CIs |

Found in the first full run of the corrected pipeline (2026-09-26) and fixed afterwards:

| # | Issue | Effect | Fix |
|---|---|---|---|
| A16 | MIMIC age computed to the day (16,419 distinct values) vs eICU whole years (72 values) | Age resolution alone let a tree classifier separate the databases (domain AUC 0.987), distorting covariate-shift weights | Whole-year age in both (`harmonize.clean`) |
| A17 | Vitals averaged within each hour: eICU = mean of ~12 five-minute values, MIMIC = ~1 charted value | eICU values fractional and smoother (resp-rate SD 20% lower) | First valid value per hour in both databases (`sql.py`, `hourly` CTE) |
| A18 | eICU temperature read from `vitalperiodic` only (89% missing) | Temperature dropped from all models | Temperature also read from eICU `nursecharting` |
| A19 | Domain AUC reported from a tree model only | Sensitive to recording artifacts | Added linear-model domain AUC and leave-one-feature-out domain AUC (`shift/`) |

Found in the final audit (2026-09-28) and fixed; re-run with
`configs/revision_update.yaml` (same data, only the affected stages):

| # | Issue | Effect | Fix |
|---|---|---|---|
| A20 | Tree domain AUC still 0.95 vs linear 0.73; leave-one-out shows `temperature_max` (Fahrenheit-converted MIMIC vs charted eICU Celsius) and lab precision | Density-ratio weights keyed on recording resolution: decomposition, domain adaptation and per-hospital shift scores distorted | Logistic-regression weights by default; tree weights kept as sensitivity (`shift.domain_propensity(kind=...)`) |
| A21 | Recalibration and per-hospital analysis for SBP only | The ~5 mmHg DBP/MAP offset was not tested for correctability or heterogeneity | All targets |
| A22 | Hypotension modelled only without BP history (and persistence) | The clinically relevant forecasting model was missing [R2.3, R4] | Ablation over all targets, including hypotension with all features + BP history |
| A23 | Measurement explanation for the offset not directly tested [R3] | Interpretation, not evidence | New `modality_agreement`: cuff minus arterial in the same stays, compared between databases |
| A24 | No feature-level attribution of hospital-level loss [R3] | "Which differences matter" unanswered | `hospitals/feature_shift_vs_loss.csv`: per-feature hospital SMD vs error, BH-corrected |
| A25 | LaTeX tables used code names; recalibration table lacked calibration metrics | [R1.5] | Readable labels; O/E and calibration slope for hypotension |
| A26 | `git_dirty` counted untracked logs/archives; partial reruns overwrote the manifest | Provenance unclear | Only tracked `revision/` and `configs/` files count; timestamped manifests |

## B. Design requirements referred to in code comments

| ID | Requirement | Where |
|---|---|---|
| R1.1 | Every feature set is evaluated on the same patients | `experiments.ablation()`, `harmonize.py` |
| R1.2 | Models are also trained and evaluated within eICU, so that low predictability is not mistaken for transfer loss | `experiments.models_transfer()`, `experiments.ladder()` |
| R1.3 | One unit of analysis (first ICU stay per patient); cohort numbers generated from code | `experiments.cohort()`, `sql.py`, `evaluation.py` |
| R1.4 | Prediction task defined in time, with non-overlapping feature and target windows | `config.py` (`design`), `experiments.ladder()` |
| R1.5 | Readable, consistent figures and tables generated from the result files | `reporting.py` |
| R2.3 | Clinically meaningful target: forecasting hypotension | target `hypotension` |
| R2.4 | Simple baselines, single learners and uncertainty estimates | `models.py`, `evaluation.py`, `experiments.models_transfer()` |
| R2.5 | Feature definitions held in one place; reproducible runs (configuration, seeds, manifest, tests) | `schema.py`, `run_revision.py`, `tests/` |
| R3 | Causes of transfer loss demonstrated rather than asserted: measured shift, gap decomposition, per-hospital analysis and a direct test of the measurement explanation | `shift.py`, `experiments.decomposition()`, `experiments.hospitals()`, `experiments.modality_agreement()` |
| R4 | Forecast future BP rather than estimate current BP; leakage prevented by design rather than by dropping variables | `config.py`, feature sets `bp_history_only` and `full` |
