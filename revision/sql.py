"""BigQuery SQL for MIMIC-III and eICU.

Both databases are reduced to the same three long tables, and then go through ONE
shared aggregation template, so time windows, plausibility ranges and hourly
resampling are identical by construction:

  hourly_long(stay_id, hour_idx, var, val)   vitals/BP, hour since ICU admission
  labs_long(stay_id, var, offset_hr, val)    labs, hours since ICU admission
  meds_long(stay_id, drug_class, offset_hr)  medication starts

Vitals are reduced to ONE value per hour -- the first valid value in the hour -- in
both databases. eICU vitalperiodic holds 5-minute automated values while MIMIC holds
roughly hourly charted values; taking the first value per hour gives both the same
sampling and resolution. (The submitted version counted raw rows, producing the
"measurement_frequency" W = 427.8 artifact; averaging within the hour would instead
smooth eICU and make its values fractional, which a domain classifier can detect.)
"""

from . import schema as S

MIMIC = 'physionet-data.mimiciii_clinical'
EICU = 'physionet-data.eicu_crd'

OBS = 'hour_idx < @obs_hours'
TGT = 'hour_idx >= @horizon_start_hours AND hour_idx < @horizon_end_hours'


def _ids(ids):
    return ', '.join(str(i) for i in ids)


def _strs(values):
    return ', '.join(f"'{v}'" for v in values)


def temp_to_celsius(expr):
    """Same rule for both databases: values in 77-113 are Fahrenheit."""
    return (f'CASE WHEN {expr} BETWEEN 25 AND 45 THEN {expr} '
            f'WHEN {expr} BETWEEN 77 AND 113 THEN ({expr} - 32) * 5 / 9 ELSE NULL END')


def _drug_class_case(name_expr):
    whens = ' '.join(f"WHEN REGEXP_CONTAINS(LOWER({name_expr}), r'{pat}') THEN '{cls}'"
                     for cls, pat in S.DRUG_CLASS_PATTERNS.items())
    return f'CASE {whens} ELSE NULL END'


def _vital_range_condition():
    parts = []
    for var in S.HOURLY_VARS:
        lo, hi = S.VITAL_RANGES[S.base_var(var)]
        parts.append(f"(var = '{var}' AND val BETWEEN {lo} AND {hi})")
    return '\n      OR '.join(parts)


def _lab_range_condition():
    return '\n      OR '.join(f"(var = '{lab}' AND val BETWEEN {lo} AND {hi})"
                              for lab, (lo, hi) in S.LAB_RANGES.items())


COHORT_FILTER = """
      stay_seq = 1
      AND age >= @min_age AND age < @max_age_exclusive
      AND los_hours >= @obs_hours
      AND MOD(ABS(FARM_FINGERPRINT(CAST(stay_id AS STRING))), 1000) < @sample_permille"""


# ---------------------------------------------------------------------------
# Dataset-specific CTEs
# ---------------------------------------------------------------------------

def mimic_cohort_all():
    return f"""
cohort_all AS (
  SELECT
    icu.icustay_id AS stay_id,
    icu.subject_id,
    icu.intime,
    CAST(icu.subject_id AS STRING) AS patient_id,
    'MIMIC-III' AS site,
    icu.dbsource AS ehr_system,
    icu.first_careunit AS unit_type,
    DATETIME_DIFF(icu.intime, p.dob, DAY) / 365.242 AS age,
    p.gender AS gender_raw,
    DATETIME_DIFF(icu.outtime, icu.intime, MINUTE) / 60.0 AS los_hours,
    CAST(NULL AS STRING) AS hospital_beds,
    CAST(NULL AS STRING) AS hospital_teaching,
    CAST(NULL AS STRING) AS hospital_region,
    ROW_NUMBER() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS stay_seq
  FROM `{MIMIC}.icustays` icu
  JOIN `{MIMIC}.patients` p ON icu.subject_id = p.subject_id
  WHERE icu.intime IS NOT NULL AND icu.outtime IS NOT NULL
)"""


def mimic_long_ctes():
    item_case = ' '.join(f"WHEN ce.itemid IN ({_ids(ids)}) THEN '{var}'"
                         for var, ids in S.MIMIC_VITAL_ITEMIDS.items())
    all_items = _ids(sorted({i for ids in S.MIMIC_VITAL_ITEMIDS.values() for i in ids}))
    temp_items = _ids(S.MIMIC_VITAL_ITEMIDS['temperature'])
    lab_case = ' '.join(f"WHEN le.itemid IN ({_ids(ids)}) THEN '{lab}'"
                        for lab, ids in S.MIMIC_LAB_ITEMIDS.items())
    lab_items = _ids(sorted({i for ids in S.MIMIC_LAB_ITEMIDS.values() for i in ids}))
    return f"""
hourly_long AS (
  SELECT
    c.stay_id,
    DATETIME_DIFF(ce.charttime, c.intime, MINUTE) AS minute_idx,
    CASE {item_case} END AS var,
    CASE WHEN ce.itemid IN ({temp_items}) THEN {temp_to_celsius('ce.valuenum')}
         ELSE ce.valuenum END AS val
  FROM `{MIMIC}.chartevents` ce
  JOIN cohort c ON ce.icustay_id = c.stay_id
  WHERE ce.itemid IN ({all_items})
    AND ce.valuenum IS NOT NULL
    AND (ce.error IS NULL OR ce.error = 0)
    AND ce.charttime >= c.intime
    AND ce.charttime < DATETIME_ADD(c.intime, INTERVAL @horizon_end_hours HOUR)
),
labs_long AS (
  -- joined on subject + time window (NOT hadm_id), so only labs from the
  -- feature window are used; the submitted version averaged the whole admission
  SELECT
    c.stay_id,
    CASE {lab_case} END AS var,
    DATETIME_DIFF(le.charttime, c.intime, MINUTE) / 60.0 AS offset_hr,
    le.valuenum AS val
  FROM `{MIMIC}.labevents` le
  JOIN cohort c ON le.subject_id = c.subject_id
  WHERE le.itemid IN ({lab_items})
    AND le.valuenum IS NOT NULL
    AND le.charttime >= DATETIME_SUB(c.intime, INTERVAL @lab_lookback_hours HOUR)
    AND le.charttime < DATETIME_ADD(c.intime, INTERVAL @obs_hours HOUR)
),
meds_long AS (
  SELECT c.stay_id, {_drug_class_case('di.label')} AS drug_class,
         DATETIME_DIFF(ie.starttime, c.intime, MINUTE) / 60.0 AS offset_hr
  FROM (
    SELECT icustay_id, itemid, starttime
    FROM `{MIMIC}.inputevents_mv`
    WHERE statusdescription IS NULL OR statusdescription != 'Rewritten'
    UNION ALL
    SELECT icustay_id, itemid, charttime AS starttime
    FROM `{MIMIC}.inputevents_cv`
  ) ie
  JOIN cohort c ON ie.icustay_id = c.stay_id
  JOIN `{MIMIC}.d_items` di ON ie.itemid = di.itemid
)"""


def eicu_cohort_all():
    return f"""
cohort_all AS (
  SELECT
    p.patientunitstayid AS stay_id,
    p.uniquepid AS patient_id,
    CAST(p.hospitalid AS STRING) AS site,
    'eICU' AS ehr_system,
    p.unittype AS unit_type,
    CASE WHEN TRIM(CAST(p.age AS STRING)) = '> 89' THEN 90.0
         ELSE SAFE_CAST(p.age AS FLOAT64) END AS age,
    p.gender AS gender_raw,
    p.unitdischargeoffset / 60.0 AS los_hours,
    CAST(h.numbedscategory AS STRING) AS hospital_beds,
    CAST(h.teachingstatus AS STRING) AS hospital_teaching,
    CAST(h.region AS STRING) AS hospital_region,
    ROW_NUMBER() OVER (PARTITION BY p.uniquepid
                       ORDER BY p.hospitaldischargeyear, p.patienthealthsystemstayid,
                                p.unitvisitnumber) AS stay_seq
  FROM `{EICU}.patient` p
  LEFT JOIN `{EICU}.hospital` h ON p.hospitalid = h.hospitalid
)"""


def _unnest_structs(alias, mapping, transform=None):
    structs = []
    for var, col in mapping.items():
        expr = f'SAFE_CAST({alias}.{col} AS FLOAT64)'
        if transform and var in transform:
            expr = transform[var](expr)
        structs.append(f"STRUCT('{var}' AS var, {expr} AS val)")
    return ',\n      '.join(structs)


def eicu_long_ctes():
    lab_case = ' '.join(f"WHEN labname IN ({_strs(names)}) THEN '{lab}'"
                        for lab, names in S.EICU_LAB_NAMES.items())
    lab_names = _strs(sorted({n for names in S.EICU_LAB_NAMES.values() for n in names}))
    periodic = _unnest_structs('vp', S.EICU_PERIODIC_COLUMNS,
                               transform={'temperature': temp_to_celsius})
    aperiodic = _unnest_structs('va', S.EICU_APERIODIC_COLUMNS)
    return f"""
hourly_long AS (
  SELECT vp.patientunitstayid AS stay_id, vp.observationoffset AS minute_idx,
         v.var, v.val
  FROM `{EICU}.vitalperiodic` vp
  CROSS JOIN UNNEST([
      {periodic}
  ]) v
  WHERE vp.patientunitstayid IN (SELECT stay_id FROM cohort)
    AND vp.observationoffset >= 0
    AND vp.observationoffset < @horizon_end_hours * 60
    AND v.val IS NOT NULL
  UNION ALL
  -- cuff BP: vitalaperiodic (the submitted version never read this table)
  SELECT va.patientunitstayid AS stay_id, va.observationoffset AS minute_idx,
         v.var, v.val
  FROM `{EICU}.vitalaperiodic` va
  CROSS JOIN UNNEST([
      {aperiodic}
  ]) v
  WHERE va.patientunitstayid IN (SELECT stay_id FROM cohort)
    AND va.observationoffset >= 0
    AND va.observationoffset < @horizon_end_hours * 60
    AND v.val IS NOT NULL
  UNION ALL
  -- temperature: eICU records it mainly in nursecharting (vitalperiodic ~89% missing)
  SELECT nc.patientunitstayid AS stay_id, nc.nursingchartoffset AS minute_idx,
         'temperature' AS var,
         {temp_to_celsius('SAFE_CAST(nc.nursingchartvalue AS FLOAT64)')} AS val
  FROM `{EICU}.nursecharting` nc
  WHERE nc.patientunitstayid IN (SELECT stay_id FROM cohort)
    AND nc.nursingchartcelltypevallabel = 'Temperature'
    AND nc.nursingchartcelltypevalname IN ('Temperature (C)', 'Temperature (F)')
    AND nc.nursingchartoffset >= 0
    AND nc.nursingchartoffset < @horizon_end_hours * 60
),
labs_long AS (
  SELECT patientunitstayid AS stay_id,
         CASE {lab_case} END AS var,
         labresultoffset / 60.0 AS offset_hr,
         SAFE_CAST(labresult AS FLOAT64) AS val
  FROM `{EICU}.lab`
  WHERE patientunitstayid IN (SELECT stay_id FROM cohort)
    AND labname IN ({lab_names})
    AND labresult IS NOT NULL
    AND labresultoffset >= -@lab_lookback_hours * 60
    AND labresultoffset < @obs_hours * 60
),
meds_long AS (
  SELECT patientunitstayid AS stay_id, {_drug_class_case('drugname')} AS drug_class,
         infusionoffset / 60.0 AS offset_hr
  FROM `{EICU}.infusiondrug`
  WHERE patientunitstayid IN (SELECT stay_id FROM cohort)
    AND IFNULL(SAFE_CAST(drugrate AS FLOAT64), 1) > 0
  UNION ALL
  -- IV orders only, to match MIMIC inputevents (IV administrations)
  SELECT patientunitstayid AS stay_id, {_drug_class_case('drugname')} AS drug_class,
         drugstartoffset / 60.0 AS offset_hr
  FROM `{EICU}.medication`
  WHERE patientunitstayid IN (SELECT stay_id FROM cohort)
    AND drugordercancelled = 'No'
    AND REGEXP_CONTAINS(LOWER(IFNULL(routeadmin, '')), r'\\biv\\b|intraven')
)"""


# ---------------------------------------------------------------------------
# Shared aggregation template
# ---------------------------------------------------------------------------

def _aggregation_ctes():
    wide = ',\n    '.join(f"MAX(IF(var = '{v}', val, NULL)) AS {v}" for v in S.HOURLY_VARS)
    any_cols = ',\n    '.join(
        f'(IFNULL({b}_ni, 0) + IFNULL({b}_art, 0)) / '
        f'NULLIF(IF({b}_ni IS NULL, 0, 1) + IF({b}_art IS NULL, 0, 1), 0) AS {b}_any'
        for b in S.BP_TYPES)
    mapx = ',\n    '.join(f'COALESCE(mbp_{m}, (sbp_{m} + 2 * dbp_{m}) / 3) AS mapx_{m}'
                          for m in S.MODALITIES)

    agg = []
    for v in S.VITALS:
        agg += [f'AVG(IF({OBS}, {v}, NULL)) AS {v}_mean',
                f'STDDEV_SAMP(IF({OBS}, {v}, NULL)) AS {v}_std',
                f'MIN(IF({OBS}, {v}, NULL)) AS {v}_min',
                f'MAX(IF({OBS}, {v}, NULL)) AS {v}_max',
                f'COUNTIF({OBS} AND {v} IS NOT NULL) AS {v}_nhours']
    for m in S.MODALITIES:
        for b in S.BP_TYPES:
            col = f'mapx_{m}' if b == 'mbp' else f'{b}_{m}'
            agg += [f'AVG(IF({OBS}, {col}, NULL)) AS hist_{b}_{m}_mean',
                    f'MIN(IF({OBS}, {col}, NULL)) AS hist_{b}_{m}_min',
                    f'ARRAY_AGG(IF({OBS}, {col}, NULL) IGNORE NULLS '
                    f'ORDER BY hour_idx DESC LIMIT 1)[SAFE_OFFSET(0)] AS hist_{b}_{m}_last',
                    f'AVG(IF({TGT}, {col}, NULL)) AS tgt_{b}_{m}']
        agg += [f'COUNTIF({OBS} AND sbp_{m} IS NOT NULL) AS hist_nhours_{m}',
                f'COUNTIF({OBS} AND mapx_{m} < @hypotension_map) AS hist_hypo_hours_{m}',
                f'COUNTIF({TGT} AND sbp_{m} IS NOT NULL) AS tgt_nhours_{m}',
                f'COUNTIF({TGT} AND mapx_{m} < @hypotension_map) AS tgt_hypo_hours_{m}']
    agg_sql = ',\n    '.join(agg)

    labs = ',\n    '.join(f"AVG(IF(var = '{lab}', val, NULL)) AS {lab}" for lab in S.LABS)
    meds = ',\n    '.join(f"MAX(IF(drug_class = '{c}', 1, 0)) AS {c}_any"
                          for c in S.DRUG_CLASS_PATTERNS)
    return f"""
hourly AS (
  -- first valid value in each hour (same rule for both databases)
  SELECT stay_id, DIV(minute_idx, 60) AS hour_idx, var,
         ARRAY_AGG(val ORDER BY minute_idx, val LIMIT 1)[OFFSET(0)] AS val
  FROM hourly_long
  WHERE var IS NOT NULL AND minute_idx >= 0 AND (
      {_vital_range_condition()})
  GROUP BY stay_id, hour_idx, var
),
wide AS (
  SELECT stay_id, hour_idx,
    {wide}
  FROM hourly
  GROUP BY stay_id, hour_idx
),
wide_any AS (
  SELECT *,
    {any_cols}
  FROM wide
),
wide_map AS (
  SELECT *,
    {mapx}
  FROM wide_any
),
vit_agg AS (
  SELECT stay_id,
    {agg_sql}
  FROM wide_map
  GROUP BY stay_id
),
labs_agg AS (
  SELECT stay_id,
    {labs}
  FROM labs_long
  WHERE var IS NOT NULL AND (
      {_lab_range_condition()})
  GROUP BY stay_id
),
meds_agg AS (
  SELECT stay_id,
    {meds}
  FROM meds_long
  WHERE drug_class IS NOT NULL AND offset_hr >= 0 AND offset_hr < @obs_hours
  GROUP BY stay_id
)"""


def _final_select():
    cohort_cols = ', '.join(f'c.{c}' for c in S.COHORT_COLUMNS)
    vit_cols = ', '.join(f'v.{c}' for c in S.vital_agg_columns() + S.bp_agg_columns())
    lab_cols = ', '.join(f'l.{c}' for c in S.LABS)
    med_cols = ', '.join(f'IFNULL(md.{c}, 0) AS {c}' for c in S.MEDS)
    return f"""
SELECT {cohort_cols},
  {vit_cols},
  {lab_cols},
  {med_cols}
FROM cohort c
LEFT JOIN vit_agg v ON v.stay_id = c.stay_id
LEFT JOIN labs_agg l ON l.stay_id = c.stay_id
LEFT JOIN meds_agg md ON md.stay_id = c.stay_id
ORDER BY c.stay_id"""


def _cohort_cte():
    return f"""
cohort AS (
  SELECT * FROM cohort_all
  WHERE {COHORT_FILTER}
)"""


def feature_query(dataset):
    if dataset == 'mimic':
        head, longs = mimic_cohort_all(), mimic_long_ctes()
    elif dataset == 'eicu':
        head, longs = eicu_cohort_all(), eicu_long_ctes()
    else:
        raise ValueError(dataset)
    return ('WITH' + head + ',' + _cohort_cte() + ',' + longs + ',' + _aggregation_ctes()
            + _final_select())


def flow_query(dataset):
    """Counts for the cohort flow diagram (Fig. 1)."""
    head = mimic_cohort_all() if dataset == 'mimic' else eicu_cohort_all()
    return f"""WITH{head}
SELECT
  COUNT(*) AS n_icu_stays,
  COUNT(DISTINCT patient_id) AS n_patients,
  COUNTIF(stay_seq = 1) AS n_first_stay,
  COUNTIF(stay_seq = 1 AND age >= @min_age AND age < @max_age_exclusive) AS n_age_eligible,
  COUNTIF(stay_seq = 1 AND age >= @min_age AND age < @max_age_exclusive
          AND los_hours >= @obs_hours) AS n_los_eligible,
  COUNTIF({COHORT_FILTER}) AS n_sampled
FROM cohort_all"""


# ---------------------------------------------------------------------------
# Outcomes for Table 1. A separate query with its own cache file, so the feature
# extraction (and its cache key) is unchanged.
# ---------------------------------------------------------------------------

def outcome_query(dataset):
    """Hospital mortality per ICU stay; stay_id matches the feature extraction."""
    if dataset == 'mimic':
        return f"""
SELECT icu.icustay_id AS stay_id,
       CAST(adm.hospital_expire_flag AS FLOAT64) AS hosp_mortality
FROM `{MIMIC}.icustays` icu
JOIN `{MIMIC}.admissions` adm ON icu.hadm_id = adm.hadm_id"""
    if dataset == 'eicu':
        return f"""
SELECT patientunitstayid AS stay_id,
       CASE hospitaldischargestatus WHEN 'Expired' THEN 1.0 WHEN 'Alive' THEN 0.0
            ELSE NULL END AS hosp_mortality
FROM `{EICU}.patient`"""
    raise ValueError(dataset)


def severity_query(spec):
    """Database-specific severity score (e.g. SAPS II, APACHE IV); one row per stay.
    Scores differ between databases and are not directly comparable."""
    where = f"WHERE {spec['where']}" if spec.get('where') else ''
    return f"""
SELECT stay_id, MAX(severity) AS severity
FROM (
  SELECT {spec['id']} AS stay_id, CAST({spec['column']} AS FLOAT64) AS severity
  FROM `{spec['table']}`
  {where}
)
GROUP BY stay_id"""
