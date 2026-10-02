"""Single source of truth for variable definitions shared by MIMIC-III and eICU.

Every feature list, item ID, unit conversion and plausibility range used anywhere in
the revision is defined here, so the paper, the SQL and the models cannot disagree
(requirements R1.3 / R2.5 in docs/PIPELINE_NOTES.md: consistent feature counts).
"""

# ---------------------------------------------------------------------------
# Hourly vital-sign variables
# ---------------------------------------------------------------------------

VITALS = ['heart_rate', 'resp_rate', 'spo2', 'temperature']
BP_TYPES = ['sbp', 'dbp', 'mbp']
# ni = non-invasive cuff, art = invasive arterial line, any = average of both
MODALITIES = ['ni', 'art', 'any']
MEASURED_MODALITIES = ['ni', 'art']

HOURLY_VARS = VITALS + [f'{b}_{m}' for m in MEASURED_MODALITIES for b in BP_TYPES]

# Plausibility ranges, applied identically to both databases after unit conversion.
# The submitted version used SBP 70-250, which silently removed hypotensive values.
VITAL_RANGES = {
    'heart_rate': (20, 250),
    'resp_rate': (4, 60),
    'spo2': (50, 100),
    'temperature': (32, 43),
    'sbp': (50, 250),
    'dbp': (20, 180),
    'mbp': (30, 200),
}


def base_var(var):
    """'sbp_ni' -> 'sbp'; 'heart_rate' -> 'heart_rate'."""
    for b in BP_TYPES:
        if var.startswith(b + '_'):
            return b
    return var


# MIMIC-III chartevents item IDs (CareVue + MetaVision), following mimic-code.
# Fixes vs. the submitted version: 443 is "Manual BP Mean" (was used as DBP),
# 6701 is "Arterial BP #2 [Systolic]" (was used as MAP), 220050/220051 are arterial
# (were labelled non-invasive), and the CareVue NBP items 455/8441/456 were missing.
MIMIC_VITAL_ITEMIDS = {
    'heart_rate': [211, 220045],
    'resp_rate': [615, 618, 220210, 224690],
    'spo2': [646, 220277],
    'temperature': [676, 678, 223761, 223762],
    'sbp_art': [51, 6701, 220050],
    'dbp_art': [8368, 8555, 220051],
    'mbp_art': [52, 6702, 220052, 225312],
    'sbp_ni': [442, 455, 220179],
    'dbp_ni': [8440, 8441, 220180],
    'mbp_ni': [443, 456, 220181],
}

# eICU source columns. The submitted version took the target from
# vitalperiodic.systemic* only, which is the *arterial line*; cuff pressures live in
# vitalaperiodic.noninvasive*.
EICU_PERIODIC_COLUMNS = {
    'heart_rate': 'heartrate',
    'resp_rate': 'respiration',
    'spo2': 'sao2',
    'temperature': 'temperature',
    'sbp_art': 'systemicsystolic',
    'dbp_art': 'systemicdiastolic',
    'mbp_art': 'systemicmean',
}
EICU_APERIODIC_COLUMNS = {
    'sbp_ni': 'noninvasivesystolic',
    'dbp_ni': 'noninvasivediastolic',
    'mbp_ni': 'noninvasivemean',
}

# ---------------------------------------------------------------------------
# Laboratory variables (same units in both databases)
# ---------------------------------------------------------------------------

LABS = ['sodium', 'potassium', 'chloride', 'bicarbonate', 'creatinine', 'bun',
        'hemoglobin', 'wbc', 'platelet', 'glucose', 'lactate']

MIMIC_LAB_ITEMIDS = {
    'sodium': [50983],
    'potassium': [50971],
    'chloride': [50902],
    'bicarbonate': [50882],
    'creatinine': [50912],
    'bun': [51006],
    'hemoglobin': [51222],
    'wbc': [51301],
    'platelet': [51265],
    'glucose': [50931],
    'lactate': [50813],
}

EICU_LAB_NAMES = {
    'sodium': ['sodium'],
    'potassium': ['potassium'],
    'chloride': ['chloride'],
    'bicarbonate': ['bicarbonate'],
    'creatinine': ['creatinine'],
    'bun': ['BUN'],
    'hemoglobin': ['Hgb'],
    'wbc': ['WBC x 1000'],
    'platelet': ['platelets x 1000'],
    'glucose': ['glucose'],
    'lactate': ['lactate'],
}

LAB_RANGES = {
    'sodium': (100, 180),
    'potassium': (1.5, 9),
    'chloride': (70, 140),
    'bicarbonate': (5, 50),
    'creatinine': (0.1, 20),
    'bun': (1, 250),
    'hemoglobin': (3, 25),
    'wbc': (0.1, 200),
    'platelet': (5, 2000),
    'glucose': (20, 1500),
    'lactate': (0.2, 30),
}

# ---------------------------------------------------------------------------
# Medications: matched by drug name in BOTH databases (MIMIC d_items.label,
# eICU drugname). The submitted version used hard-coded MIMIC item IDs, some of
# which are not the intended drugs, and assumed no medications in eICU.
# ---------------------------------------------------------------------------

DRUG_CLASS_PATTERNS = {
    'vasopressor': r'norepinephrine|levophed|epinephrine|adrenalin|phenylephrine'
                   r'|neo-?synephrine|vasopressin|dopamine',
    'antihypertensive_iv': r'nicardipine|cardene|clevidipine|esmolol|brevibloc|labetalol'
                           r'|diltiazem|cardizem|nitroglycerin|nitroprusside|nipride'
                           r'|hydralazine|enalaprilat|metoprolol',
    'diuretic_iv': r'furosemide|lasix|bumetanide|bumex',
}
MEDS = [f'{c}_any' for c in DRUG_CLASS_PATTERNS]

# ---------------------------------------------------------------------------
# Model feature groups
# ---------------------------------------------------------------------------

VITAL_FEATURES = [
    'heart_rate_mean', 'heart_rate_std', 'heart_rate_min', 'heart_rate_max',
    'resp_rate_mean', 'resp_rate_std',
    'spo2_mean', 'spo2_min',
    'temperature_mean', 'temperature_max',
]
LAB_FEATURES = list(LABS)
MED_FEATURES = list(MEDS)
DEMO_FEATURES = ['age', 'gender']
# Observation-window BP of the target modality. Only legal in the prospective design
# (features and target come from different, non-overlapping windows).
BP_HISTORY_FEATURES = [
    'hist_sbp_mean', 'hist_sbp_min', 'hist_sbp_last',
    'hist_dbp_mean', 'hist_dbp_last',
    'hist_mbp_mean', 'hist_mbp_min', 'hist_mbp_last',
]

FEATURE_SETS = {
    # Original research question: BP from non-BP EHR variables.
    'no_bp_core': VITAL_FEATURES + LAB_FEATURES + DEMO_FEATURES,
    'no_bp_core_meds': VITAL_FEATURES + LAB_FEATURES + MED_FEATURES + DEMO_FEATURES,
    'no_bp_no_labs': VITAL_FEATURES + MED_FEATURES + DEMO_FEATURES,
    'no_bp_no_vitals': LAB_FEATURES + MED_FEATURES + DEMO_FEATURES,
    # Clinically motivated forecasting task (prospective design only).
    'bp_history_only': BP_HISTORY_FEATURES + DEMO_FEATURES,
    'full': VITAL_FEATURES + LAB_FEATURES + MED_FEATURES + DEMO_FEATURES + BP_HISTORY_FEATURES,
}


def uses_bp_history(feature_set):
    return any(f in BP_HISTORY_FEATURES for f in FEATURE_SETS[feature_set])


# ---------------------------------------------------------------------------
# Columns produced by the extraction query (identical for both databases)
# ---------------------------------------------------------------------------

COHORT_COLUMNS = ['stay_id', 'patient_id', 'site', 'ehr_system', 'unit_type', 'age',
                  'gender_raw', 'los_hours', 'hospital_beds', 'hospital_teaching',
                  'hospital_region']


def vital_agg_columns():
    cols = []
    for v in VITALS:
        cols += [f'{v}_mean', f'{v}_std', f'{v}_min', f'{v}_max', f'{v}_nhours']
    return cols


def bp_agg_columns():
    cols = []
    for m in MODALITIES:
        for b in BP_TYPES:
            cols += [f'hist_{b}_{m}_mean', f'hist_{b}_{m}_min', f'hist_{b}_{m}_last',
                     f'tgt_{b}_{m}']
        cols += [f'hist_nhours_{m}', f'hist_hypo_hours_{m}',
                 f'tgt_nhours_{m}', f'tgt_hypo_hours_{m}']
    return cols


EXTRACT_COLUMNS = COHORT_COLUMNS + vital_agg_columns() + bp_agg_columns() + LABS + MEDS

TARGETS = ['sbp', 'dbp', 'mbp', 'hypotension']
