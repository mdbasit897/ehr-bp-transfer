> Note: this document was written in the working repository, where the results folder was named `revision_outputs/`. In this repository it is `results/`, and patient-level predictions are not included.

# Revision: code changes mapped to the IEEE BIBM 2026 reviews (paper B584)

The submitted pipeline `cross_institutional_bp_validation.py` is **unchanged**, so the
submitted numbers stay reproducible. The revision lives in `revision/`:

| File | Purpose |
|---|---|
| `revision/schema.py` | Single source of truth: item IDs, lab names, units, ranges, feature sets |
| `revision/sql.py` | BigQuery SQL; both databases share one aggregation template |
| `revision/extract.py` | Runs the queries, caches parquet, saves exact SQL and parameters |
| `revision/harmonize.py` | Cleaning, data-quality checks, target/cohort construction |
| `revision/models.py` | Baselines, single learners, submitted ensemble (imputation inside) |
| `revision/evaluation.py` | Metrics, patient/hospital-grouped CV, cluster bootstrap CIs |
| `revision/shift.py` | SMD, domain classifier, gap decomposition, random-effects pooling |
| `revision/experiments.py` | All experiments (one method per reviewer concern) |
| `revision/reporting.py` | Figures (PDF, IEEE widths) and LaTeX tables from saved results |
| `revision/synthetic.py` | Synthetic data with the extraction's schema, for offline tests only |
| `configs/revision.yaml` | Real run; `configs/revision_smoke.yaml` = fast synthetic test |
| `tests/test_revision.py` | Invariants (grouped folds, fixed cohort, no leakage, SQL checks) |

## How to run

```bash
pip install -r requirements.txt pyarrow pytest
python -m pytest tests -q                                              # 10 tests, ~1 s
python -m revision.run_revision synthetic --config configs/revision_smoke.yaml  # offline smoke test

# Real data (PhysioNet credentialed access to MIMIC-III and eICU on BigQuery):
export GOOGLE_CLOUD_PROJECT=<your-billing-project>
python -m revision.run_revision sql --dataset eicu      # inspect the SQL first
python -m revision.run_revision extract                 # BigQuery -> data_cache/*.parquet
python -m revision.run_revision analyze                 # results + figures + LaTeX tables
python -m revision.run_revision report                  # re-draw figures/tables only
```

Dry run first with `extraction.sample_permille: 100` (a deterministic 10% sample).
Runtime is dominated by the stacked ensemble (~50x slower than XGBoost). It is
evaluated in the model comparison; all other analyses use `primary_model: xgb`. To cut
runtime, drop `ensemble` from `analysis.models` or lower `cv_repeats`.

`data_cache/` contains patient-level data derived from MIMIC/eICU. It is gitignored; do not
commit or share it (PhysioNet DUA).

---

## A. Errors found in the submitted pipeline (fixed)

These were not raised by the reviewers but directly affect the headline claim.

| # | Submitted code | Effect | Fix |
|---|---|---|---|
| A1 | eICU target from `vitalperiodic.systemicsystolic/diastolic` (`:547`) | eICU target = **arterial line only**; MIMIC target mostly cuff. Only patients with arterial lines kept (1,060/5,000) | Cuff BP read from `vitalaperiodic.noninvasive*`; modality is a config option and **matched** between datasets (`sql.py`, `analysis.target_modality`) |
| A2 | MIMIC item 443 used as DBP; 6701 used as MAP; 220050/220051 labelled non-invasive; CareVue NBP 455/8441/456 missing (`:262-283`) | Wrong DBP/MAP values; wrong modality counts | Item IDs follow mimic-code (`schema.MIMIC_VITAL_ITEMIDS`) |
| A3 | SBP kept only in 70-250 mmHg | Hypotensive values silently removed | Identical ranges for both DBs, SBP 50-250 (`schema.VITAL_RANGES`) |
| A4 | Single random 80/20 `train_test_split` over ICU stays (`:1560`), paper says grouped 5-fold CV | Same patient in train and test; paper/code mismatch | First ICU stay per patient; patient-grouped K-fold (and hospital-grouped for eICU) |
| A5 | "22-feature baseline" actually trains on **30** features incl. 6 medications; eICU gets the MIMIC median for them (`:103`, `:1591`) | Paper/code mismatch; medications constant in eICU | Features defined once in `schema.FEATURE_SETS`; medications extracted from eICU too (`infusiondrug`, `medication`) |
| A6 | `measurement_frequency` = count of hourly charted values (MIMIC) vs. 5-minute rows (eICU) | The "largest shift" (W = 427.8) is an artifact | Vitals averaged per hour in both DBs before any statistic; frequency feature removed |
| A7 | eICU temperature not unit-checked (SD 5.0 °C vs 0.85) | Fahrenheit values treated as Celsius | Same F/C rule and range for both DBs (`sql.temp_to_celsius`) |
| A8 | eICU gender constant (mean 0, SD 0 in `feature_distribution_shifts.csv`) | Feature useless / encoding bug | Robust mapping + data-quality check that errors on single-level gender |
| A9 | Labs joined on `hadm_id` = whole hospital admission | Labs after the prediction time used | Labs restricted to [-6 h, index time) by timestamp |
| A10 | Medication item IDs (e.g. 225154, 225792) not the intended drug classes | Wrong exposures | Drug classes matched by name (`d_items.label`, eICU `drugname`) in both DBs |
| A11 | `LIMIT 10000` without `ORDER BY` | Arbitrary, non-reproducible cohort | Full cohort, or a deterministic hash sample (`sample_permille`) |
| A12 | Recalibration fitted on a random 20% of eICU **rows** (paper: "on MIMIC-III"), evaluated on a different subset (R² −0.170 vs −0.115) | Paper/code mismatch; R1.3's discrepancy | Recalibration fitted on a subset of eICU **hospitals**, evaluated on the others, repeated; compared with a local model trained on the same data |
| A13 | Domain adaptation reweighted only the GB model and compared it with the ensemble | Unfair comparison | Same model with/without weights, same evaluation rows, repeated splits |
| A14 | Imputation "KNN + KS test" (paper) vs median (code); KS never run; Yeo-Johnson/variance filter never applied; correlation leakage check absent | Paper/code mismatch | Methods section must describe `revision/` exactly; imputation inside each training fold; sensitivity analysis kept |
| A15 | eICU DBP transferred reasonably (R² = 0.233) but only SBP was reported | Selective reporting | All targets (SBP, DBP, MAP, hypotension) reported with CIs |

Found in the first revision run (2026-09-26) and fixed afterwards:

| # | Issue | Effect | Fix |
|---|---|---|---|
| A16 | MIMIC age computed to the day (16,419 distinct values) vs eICU whole years (72 values) | Age resolution alone let a tree classifier separate the databases (domain AUC 0.987), distorting covariate-shift weights | Whole-year age in both (`harmonize.clean`) |
| A17 | Vitals averaged within each hour: eICU = mean of ~12 five-minute values, MIMIC = ~1 charted value | eICU values fractional and smoother (resp-rate SD 20% lower) | First valid value per hour in both databases (`sql.py`, `hourly` CTE) |
| A18 | eICU temperature read from `vitalperiodic` only (89% missing) | Temperature dropped from all models | Temperature also read from eICU `nursecharting` |
| A19 | Domain AUC reported from a tree model only | Sensitive to recording artifacts | Added linear-model domain AUC and leave-one-feature-out domain AUC (`shift/`) |

Found in the review audit of the final run (2026-09-28) and fixed; re-run with
`configs/revision_update.yaml` (same data, only the affected stages):

| # | Issue | Effect | Fix |
|---|---|---|---|
| A20 | Tree domain AUC still 0.95 vs linear 0.73; leave-one-out shows `temperature_max` (Fahrenheit-converted MIMIC vs charted eICU Celsius) and lab precision | Density-ratio weights keyed on recording resolution: decomposition, domain adaptation and per-hospital shift scores distorted | Logistic-regression weights by default; tree weights kept as sensitivity (`shift.domain_propensity(kind=...)`) |
| A21 | Recalibration and per-hospital analysis for SBP only | The ~5 mmHg DBP/MAP offset was not tested for correctability or heterogeneity | All targets |
| A22 | Hypotension modelled only without BP history (and persistence) | The clinically relevant forecasting model was missing (R2.3, R4) | Ablation over all targets, including hypotension with all features + BP history |
| A23 | Measurement explanation for the offset not directly tested (R3) | Interpretation, not evidence | New `modality_agreement`: cuff minus arterial in the same stays, compared between databases |
| A24 | No feature-level attribution of hospital-level loss (R3) | "Which differences matter" unanswered | `hospitals/feature_shift_vs_loss.csv`: per-feature hospital SMD vs error, BH-corrected |
| A25 | LaTeX tables used code names; recalibration table lacked calibration metrics | R1.5 | Readable labels; O/E and calibration slope for hypotension |
| A26 | `git_dirty` counted untracked logs/archives; partial reruns overwrote the manifest | Provenance unclear | Only tracked `revision/` and `configs/` files count; timestamped manifests |

## B. Reviewer comments → code → output

Outputs are under `revision_outputs/` (`figures/`, `tables/*.tex`, and one folder per experiment).

### Reviewer 1
| Comment | Change | Output |
|---|---|---|
| 1. Intersection experiment changes features **and** cohort (7,167 → 6,010) | Rows depend only on the target, never on features; every feature set runs on identical patients (asserted in code and tested). Complete-case sensitivity uses one fixed complete-case subset for all feature sets | `ablation/feature_set_ablation.csv`, `table_ablation_*.tex`, `fig_feature_ablation.pdf` |
| 2. Distinguish transfer failure from low intrinsic predictability: train locally in eICU | eICU-local models (patient-grouped CV) and eICU "unseen hospitals" (hospital-grouped CV); reverse direction eICU→MIMIC; gap decomposition | `models_transfer/metrics_ci.csv`, `paired_differences.csv`, `decomposition/` |
| 3. Inconsistent numbers (10,000 vs 7,167; 74 vs 30/24/22/16 features; −0.170 vs −0.115; unit of analysis) | One unit (first ICU stay per patient); automatic cohort flow; all tables generated from results; feature lists written out | `cohort/flow.csv`, `fig_cohort_flow.pdf`, `ablation/feature_lists.csv`, `tables/*.tex` |
| 4. Prediction task not defined in time | Index time = 24 h after ICU admission; features from [0, 24 h); target = mean BP over [24, 48 h); labs from [−6 h, 24 h). Configurable. The old "concurrent" task is kept only as ladder step S0-S2 | `configs/revision.yaml`, ladder |
| 5. Figures/tables rudimentary | Consistent vector figures at IEEE widths, colour-blind-safe palette, CIs everywhere; booktabs LaTeX tables | `figures/*.pdf`, `tables/*.tex` |

### Reviewer 2
| Comment | Change | Output |
|---|---|---|
| 1. Contribution not significant | The **harmonization ladder** changes one thing at a time from the submitted task to the corrected one, showing how much of the "failure" was data harmonization; decomposition separates covariate shift from conditional shift | `ladder/`, `fig_harmonization_ladder.pdf`, `decomposition/` |
| 2. Literature (not code) | – | Expand related work (see section C) |
| 3. Practical significance of whole-stay average BP | Prospective forecast of BP **and of hypotension** (≥2 h with MAP < 65 mmHg in the next 24 h) | target `hypotension` in every analysis |
| 4. No simple baselines / single learners / uncertainty | Models: mean/prevalence, **persistence** (last-window BP), ridge/logistic, GB, RF, XGB, submitted ensemble. Cluster-bootstrap 95% CIs (hospital clusters for eICU), repeated grouped CV, paired bootstrap for differences | `models_transfer/`, `table_models_*.tex`, `fig_models_transfer.pdf` |
| 5. Rigor / reproducibility | Config-driven, seeded, deterministic cohort, saved SQL + parameters, run manifest (git commit, package versions), unit tests | `run_manifest.json`, `data_cache/*.sql`, `tests/` |

### Reviewer 3
| Comment | Change | Output |
|---|---|---|
| "Fundamental barriers" not demonstrated; causes are interpretations | Measured shift (SMD, domain-classifier AUC); **gap decomposition**: source → source reweighted to eICU case-mix (pure covariate shift) → observed eICU → eICU-local ceiling | `shift/`, `decomposition/gap_decomposition.csv`, `fig_gap_decomposition_sbp.pdf` |
| Task is odd (same-window features and target) | Prospective design (see R1.4) | – |
| Weak internal model makes the failure ambiguous | eICU-local ceiling + persistence baseline show what is achievable in each dataset | `models_transfer/` |
| No uncertainty estimates | See R2.4 | – |
| Fig. 1 says 74 features; "complete data" vs imputation | Feature lists generated from code; "complete case" used only for the named sensitivity analysis | `ablation/feature_lists.csv` |
| eICU is multi-centre, so do per-site analysis | Per-hospital metrics for the MIMIC model and for models trained on other eICU hospitals; random-effects pooling (τ, I², prediction interval); Spearman correlation of hospital shift vs performance loss; breakdown by teaching status, size, region | `hospitals/*.csv`, `fig_hospitals_sbp.pdf` |
| Present as a negative case study | Framing (section C) | – |

### Reviewer 4
| Comment | Change | Output |
|---|---|---|
| BP is cheap to measure, so why predict it? | Primary clinical target becomes **forecasting** future BP/hypotension (which cannot be measured in advance); BP-history features allowed because windows do not overlap | `hypotension` target, feature set `full`, ladder S4 |
| Only a bespoke model; relate to other EHR approaches | Standard baselines and single learners (R2.4); related work (text) | `models_transfer/` |
| Aggressive leakage filtering may make the task too hard | Leakage is prevented by **design** (disjoint time windows), not by dropping variables; `bp_history_only` and `full` feature sets quantify what BP history adds | `ablation/` |

## C. Manuscript changes (not code)

1. New title and framing, e.g. *"Apparent cross-institutional failure of EHR blood-pressure
   models: a harmonization case study from MIMIC-III to eICU"*. Drop "fundamental barriers"
   unless the decomposition shows a large conditional-shift component after harmonization.
2. Methods: describe exactly what `revision/` does (cohort flow, time windows, modality,
   features, grouped CV, bootstrap). Remove claims the code does not implement (A14).
3. Report the submitted pipeline's result as ladder step S0 and state errors A1-A13 openly;
   this is the paper's lesson, not something to hide.
4. Related work: EHR/ICU hypotension prediction and external validation, dataset shift
   in clinical ML, and the MIMIC/eICU transfer literature (R2.2, R4).
5. Update the arXiv version (2507.19530) and the repository README once revised results
   exist, since both currently describe methods the code does not implement.
