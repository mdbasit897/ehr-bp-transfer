# Guide: creating the public repository for the paper

This guide explains how to build a **new, clean public repository** for the paper
*"When Validation Fails for the Wrong Reasons: Data Harmonization, Measurement Differences,
and Cross-Institutional Transfer of EHR-Based Blood Pressure Models"*, starting from this
working repository (`ehr-bp-ensemble`). You will do the steps manually. Each step lists the
exact files and commands.

---

## 1. What to release, and what never to release

MIMIC-III and eICU are credentialed-access datasets. The PhysioNet Credentialed Health Data
Use Agreement does not allow you to share the data, or anything derived from it **at the
level of individual patients or ICU stays**. Aggregate results (metrics, summary
statistics, figures, tables) are what papers publish, and they are fine to release.

| Release? | What | Why |
|---|---|---|
| ✅ **Yes** | All code (`revision/`, tests, configs, the SQL it generates) | Needed to reproduce the paper; contains no data |
| ✅ **Yes** | Aggregate results: metric CSVs, figures, LaTeX tables, run manifest | These are the paper's numbers. Releasing them lets reviewers and readers check every value without data access, and supports the paper's claim that one pipeline generates all tables and figures |
| ⚠️ **Yes, after editing** | `hospitals/per_hospital.csv`, `hospitals/by_hospital_characteristics.csv` | Hospital-level results. Remove the hospital ID and hospital characteristics, and drop groups with fewer than 5 hospitals (see step 4) |
| ⚠️ **Yes, as a separate folder** | The preliminary-analysis script and its outputs | Transparency: the paper discusses the errors in it. Must be clearly labelled as superseded |
| ❌ **Never** | `data_cache/` (BigQuery extracts, `*.parquet`) | Patient-level data |
| ❌ **Never** | `revision_outputs/predictions/` (`*.parquet`) | One row per ICU stay with stay IDs and observed BP: patient-level derived data |
| ❌ **Never** | `revision_outputs.zip`, `review_driven_analyses.zip` | `revision_outputs.zip` contains the prediction files |
| ❌ **No** | `revision_outputs_backup/`, `*.log`, `smoke_test/`, `__pycache__/` | Superseded run, run logs, test scratch; clutter |
| ❌ **No** | This repository's `.git` history | It contains `revision_outputs.zip` (patient-level predictions) in commit `886c3d2`. **Start the new repository with fresh history.** Keep the old repository private or archive it |

**Short answer to "code only, or outputs too?":** release the code **and** the aggregate
outputs (results CSVs, figures, tables), never the data or the predictions. Code alone is
acceptable under the data use agreement, but because nobody can rerun the analysis
without credentialed access and BigQuery, the aggregate outputs are what make the paper
checkable. JAMIA and TRIPOD+AI both expect this kind of availability.

---

## 2. Target directory structure

Suggested repository name: `ehr-bp-transfer` (or keep `ehr-bp-ensemble`; see step 7).

```
ehr-bp-transfer/
├── README.md                    # new README written for the paper (skeleton in step 6)
├── LICENSE                      # MIT (missing in the current repository; add it)
├── CITATION.cff                 # how to cite (template in step 6)
├── requirements.txt             # cleaned, with tested versions (step 5)
├── .gitignore                   # blocks data, predictions, archives, logs (step 5)
│
├── revision/                    # analysis package (unchanged)
│   ├── __init__.py
│   ├── config.py  schema.py  sql.py  extract.py  harmonize.py
│   ├── models.py  evaluation.py  shift.py  experiments.py
│   ├── reporting.py  synthetic.py  run_revision.py
│
├── configs/
│   ├── revision.yaml            # main run (project_id set to null)
│   ├── revision_update.yaml     # partial rerun after the review audit
│   ├── revision_fast.yaml       # quick run without the stacked ensemble
│   └── revision_smoke.yaml      # offline test on synthetic data
│
├── tests/
│   └── test_revision.py
│
├── results/                     # aggregate outputs of the final run (no patient data)
│   ├── cohort/  shift/  models_transfer/  ladder/  decomposition/
│   ├── hospitals/  ablation/  recalibration/  domain_adaptation/
│   ├── modality_agreement/  imputation/
│   ├── figures/                 # *.pdf, *.png, figures.tex
│   ├── tables/                  # *.tex
│   ├── run_manifest.json        # + run_manifest_<timestamp>.json
│   └── timings.json
│
├── docs/
│   └── REVISION_CHANGES.md      # errors found and how each reviewer comment was addressed
│
└── legacy/                      # preliminary analysis (arXiv v1), superseded
    ├── README.md                # what this is and why it is superseded (step 3)
    ├── cross_institutional_bp_validation.py
    └── outputs/                 # its original figures/results/tables (optional)
```

Notes:
- The package keeps the name `revision/`, so imports, tests and commands
  (`python -m revision.run_revision ...`) work unchanged. Renaming it is optional; see step 7.
- `revision_outputs/` becomes `results/`; the configs are updated to match in step 4.

---

## 3. Copy the files

Run from the folder that contains `ehr-bp-ensemble/`:

```bash
OLD=ehr-bp-ensemble
NEW=ehr-bp-transfer
mkdir -p $NEW/{revision,configs,tests,results,docs,legacy/outputs}

# code, configs, tests
cp $OLD/revision/*.py            $NEW/revision/
cp $OLD/configs/*.yaml           $NEW/configs/
cp $OLD/tests/test_revision.py   $NEW/tests/
cp $OLD/REVISION_CHANGES.md      $NEW/docs/

# aggregate results (everything except predictions and logs)
rsync -a --exclude 'predictions/' --exclude '*.log' --exclude '*.parquet' \
      $OLD/revision_outputs/ $NEW/results/

# legacy (preliminary) analysis and its original outputs from the last pre-revision commit
cp $OLD/cross_institutional_bp_validation.py $NEW/legacy/
git -C $OLD archive e9d30a0 figures results tables | tar -x -C $NEW/legacy/outputs
rm -f $NEW/legacy/outputs/results/*.log
```

Create `legacy/README.md` with a short note, for example:

> This folder contains the preliminary analysis reported in arXiv:2507.19530v1. It is
> **superseded** by the analysis in `revision/`. Known errors (for example, the eICU target was
> arterial-line BP while the MIMIC-III target was mostly cuff BP) are listed in
> `docs/REVISION_CHANGES.md`, section A. Kept for transparency only; do not use.

---

## 4. Clean the copied files

```bash
cd ehr-bp-transfer

# (a) remove your GCP project ID from configs, manifests and the legacy script
sed -i 's/^project_id: .*/project_id: null   # your GCP billing project (or set GOOGLE_CLOUD_PROJECT)/' configs/*.yaml
sed -i 's/"project_id": "[^"]*"/"project_id": null/' results/run_manifest*.json
sed -i "s/d1namo-ecg-processing/your-gcp-project-id/g" legacy/cross_institutional_bp_validation.py

# (b) point the configs at results/ instead of revision_outputs/
sed -i 's#^output_dir: revision_outputs$#output_dir: results#' configs/revision.yaml configs/revision_update.yaml
sed -i 's#^output_dir: revision_outputs_fast$#output_dir: results_fast#' configs/revision_fast.yaml

# (c) hospital-level files: drop hospital IDs/characteristics and groups of < 5 hospitals
python3 - <<'EOF'
import pandas as pd
p = 'results/hospitals/per_hospital.csv'
d = pd.read_csv(p).drop(columns=['site', 'hospital_beds', 'hospital_teaching', 'hospital_region'])
d.to_csv(p, index=False)
p = 'results/hospitals/by_hospital_characteristics.csv'
d = pd.read_csv(p)
d[d['n_hospitals'] >= 5].to_csv(p, index=False)
print('hospital files cleaned')
EOF
```

Then check `configs/revision_smoke.yaml`: it writes to `smoke_test/`, which `.gitignore` excludes.

---

## 5. `.gitignore` and `requirements.txt`

`.gitignore`:

```gitignore
# patient-level data: never commit (PhysioNet data use agreement)
data_cache/
results/predictions/
results_fast/predictions/
*.parquet
*.zip

# runs and scratch
*.log
smoke_test/
results_fast/
__pycache__/
*.pyc
.pytest_cache/
```

`requirements.txt` (versions used for the published results on Python 3.8; newer versions
also pass the tests):

```text
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
pyyaml>=6.0
jinja2>=3.1
pyarrow>=12
google-cloud-bigquery>=3.0
db-dtypes>=1.1
pytest>=7
```

(The old file also listed `shap`, `scikit-optimize`, `missingno`, `lightgbm`, `catboost`,
`seaborn` and `gcloud`; the revision does not use them.)

---

## 6. README, CITATION and LICENSE

**`README.md` skeleton:**

```markdown
# When Validation Fails for the Wrong Reasons
Code and aggregate results for: Azam MB, Singh SI. "When Validation Fails for the Wrong
Reasons: Data Harmonization, Measurement Differences, and Cross-Institutional Transfer of
EHR-Based Blood Pressure Models." (journal, year; arXiv:2507.19530)

## What this repository contains
- `revision/`: the complete analysis pipeline (BigQuery extraction to figures and tables)
- `results/`: aggregate outputs of the final run; every number, table and figure in the paper
- `legacy/`: the superseded preliminary analysis (arXiv v1), kept for transparency
- No patient-level data or predictions (MIMIC-III and eICU are credentialed-access)

## Data access
1. Complete PhysioNet credentialing; sign the data use agreements for MIMIC-III v1.4 and eICU-CRD v2.0.
2. Link PhysioNet to Google BigQuery; set `GOOGLE_CLOUD_PROJECT` to your billing project.

## Reproduce
pip install -r requirements.txt
python -m pytest tests -q                                          # 12 tests
python -m revision.run_revision synthetic --config configs/revision_smoke.yaml  # offline check
python -m revision.run_revision sql --dataset eicu                 # inspect the SQL
python -m revision.run_revision extract                            # BigQuery -> data_cache/
python -m revision.run_revision analyze                            # results/ (about 5-6 h)

## Mapping to the paper
Table 1: results/cohort/table1.csv; Table 2: results/ladder/; Table 3: results/models_transfer/;
... (one line per table and figure; figures/figures.tex and tables/*.tex are generated)

## Citation, license
See CITATION.cff. Code: MIT License.
```

**`CITATION.cff`:**

```yaml
cff-version: 1.2.0
message: "If you use this code, please cite the paper."
title: "When Validation Fails for the Wrong Reasons: Data Harmonization, Measurement Differences, and Cross-Institutional Transfer of EHR-Based Blood Pressure Models"
authors:
  - family-names: Azam
    given-names: Md Basit
    affiliation: Tezpur University
  - family-names: Singh
    given-names: Sarangthem Ibotombi
    affiliation: Tezpur University
repository-code: "https://github.com/mdbasit897/ehr-bp-transfer"
license: MIT
identifiers:
  - type: other
    value: "arXiv:2507.19530"
```

**`LICENSE`:** on GitHub choose *Add file, Create new file*, name it `LICENSE`, and pick the
MIT template; or copy the standard MIT text with your names and the year.

---

## 7. Optional choices

- **Repository name.** If the new name differs from `ehr-bp-ensemble`, update the code URL in
  both manuscripts (IEEE and JAMIA versions: the first-page footnote or title page, and the
  *Data and Code Availability* section). The alternative is to rename the old repository first
  and give the new one the old name.
- **Package name.** To rename `revision/` (for example to `bptransfer/`), change the folder
  name, `from revision` imports in `tests/test_revision.py`, the `python -m revision...`
  commands in README and configs comments, and rerun the tests.
- **Archival DOI.** Connect the repository to Zenodo and create a release (for example
  `v1.0-jamia-submission`) to obtain a DOI that you can cite in the paper.

---

## 8. Final checks before the first push

```bash
cd ehr-bp-transfer
git init && git add -A && git status --short | head -50     # read the list

# must print nothing:
git ls-files | grep -Ei '\.parquet$|\.zip$|predictions/|data_cache/|\.log$'
grep -rIl 'stay_id\|patient_id\|subject_id' results/ || echo "results: no identifier columns"
grep -rIn 'd1namo' . && echo "!! project ID still present" || echo "no project ID"
python3 -c "import pandas as pd; print(pd.read_csv('results/hospitals/per_hospital.csv').columns.tolist()[:4])"

# must pass:
python -m pytest tests -q
python -m revision.run_revision synthetic --config configs/revision_smoke.yaml

du -sh .        # expect a few MB; anything near 40 MB means predictions slipped in
git commit -m "Initial public release: code and aggregate results"
```

Create the GitHub repository as **private first**, push, browse it once to confirm nothing
patient-level is there, then make it public.

---

## 9. The old repository

Its history contains `revision_outputs.zip` with patient-level predictions (commit `886c3d2`).
Keep it **private** or **archive** it on GitHub. Do not make it public unless you first remove
that file from the history (for example with `git filter-repo`). With the new repository
using fresh history, you do not need to clean the old one.
