"""Configuration for the revision pipeline (loaded from YAML, no hard-coded project IDs)."""

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    'project_id': None,               # GCP billing project; or env GOOGLE_CLOUD_PROJECT
    'cache_dir': 'data_cache',
    'output_dir': 'revision_outputs',
    'seed': 42,
    'n_jobs': -1,
    'allow_synthetic': False,         # only the smoke-test config sets this to true

    # Task definition (R1.4, R2.3, R3). Hours are relative to ICU admission.
    'extraction': {
        'obs_hours': 24,              # features use data from [0, obs_hours)
        'horizon_start_hours': 24,    # target window start (prediction/index time)
        'horizon_end_hours': 48,      # target window end
        'lab_lookback_hours': 6,      # labs drawn up to 6 h before ICU admission allowed
        'hypotension_map': 65,        # mmHg
        'min_age': 18,
        'max_age_exclusive': 90,      # 18-89 as in the submitted version
        'sample_permille': 1000,      # deterministic hash sample; 1000 = everyone
    },

    'analysis': {
        # 'prospective': features [0, obs), target [horizon_start, horizon_end)
        # 'concurrent' : features and target both [0, obs) (the submitted task)
        'design': 'prospective',
        'target_modality': {'mimic': 'ni', 'eicu': 'ni'},
        'min_target_hours': 4,        # hours with a BP value required in target window
        'min_hypo_hours': 2,          # hypotension = >= 2 hourly MAP values < threshold
        'max_feature_missing': 0.5,   # feature dropped if missing > 50% in EITHER dataset
        'max_rows': 40000,            # deterministic per-dataset cap for runtime; null = all
        'targets': ['sbp', 'dbp', 'mbp', 'hypotension'],
        'primary_target': 'sbp',
        'primary_feature_set': 'no_bp_core_meds',
        'primary_model': 'xgb',        # the ensemble is still compared in models_transfer
        # no_bp_core = no_bp_core_meds without medications
        'feature_sets': ['no_bp_core_meds', 'no_bp_core', 'no_bp_no_labs',
                         'no_bp_no_vitals', 'bp_history_only', 'full'],
        'models': ['mean', 'persistence', 'ridge', 'gb', 'rf', 'xgb', 'ensemble'],
        'cv_folds': 5,
        'cv_repeats': 3,
        'n_bootstrap': 1000,
        'min_hospital_n': 100,
        'n_calibration_splits': 5,
        'calibration_fraction': 0.3,  # fraction of eICU hospitals used for recalibration
        'sensitivity_max_rows': 15000,
    },

    # Harmonization ladder: each step changes one thing relative to the previous one.
    'ladder': [
        {'name': 'S0 submitted task (concurrent, MIMIC any vs eICU arterial)',
         'design': 'concurrent', 'target_modality': {'mimic': 'any', 'eicu': 'art'},
         'feature_set': 'no_bp_core_meds'},
        {'name': 'S1 matched modality: cuff vs cuff',
         'design': 'concurrent', 'target_modality': {'mimic': 'ni', 'eicu': 'ni'},
         'feature_set': 'no_bp_core_meds'},
        {'name': 'S2 matched modality: arterial vs arterial',
         'design': 'concurrent', 'target_modality': {'mimic': 'art', 'eicu': 'art'},
         'feature_set': 'no_bp_core_meds'},
        {'name': 'S3 prospective task, no BP history',
         'design': 'prospective', 'target_modality': {'mimic': 'ni', 'eicu': 'ni'},
         'feature_set': 'no_bp_core_meds'},
        {'name': 'S4 prospective task, with BP history',
         'design': 'prospective', 'target_modality': {'mimic': 'ni', 'eicu': 'ni'},
         'feature_set': 'full'},
    ],

    'experiments': {
        'cohort': True,
        'shift': True,
        'models_transfer': True,       # internal / external / local / reverse, all models
        'ladder': True,
        'decomposition': True,
        'hospitals': True,
        'ablation': True,
        'recalibration': True,
        'domain_adaptation': True,
        'modality_agreement': True,    # cuff minus arterial BP, compared between databases
        'imputation': True,
        'figures': True,
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    cfg = _merge(cfg, overrides)
    ex = cfg['extraction']
    if ex['horizon_start_hours'] < ex['obs_hours']:
        raise ValueError('horizon_start_hours must be >= obs_hours (no overlap between '
                         'feature and target windows)')
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfg['cache_dir']).mkdir(parents=True, exist_ok=True)
    return cfg
