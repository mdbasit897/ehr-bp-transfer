"""Synthetic stand-in for the BigQuery extraction, for offline testing ONLY.

Produces the exact column layout of the extraction query, with a known structure:
covariate shift, per-hospital intercepts/slopes (conditional shift) and arterial
pressures that read higher than cuff pressures. Never use these numbers in a paper.
"""

import json

import numpy as np
import pandas as pd

from . import extract
from . import schema as S


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _dataset(n, rng, dataset, n_hospitals=1):
    is_eicu = dataset == 'eicu'
    if is_eicu:
        sizes = rng.dirichlet(np.ones(n_hospitals) * 1.5) * n
        site_idx = np.repeat(np.arange(n_hospitals), np.maximum(sizes.astype(int), 1))[:n]
        site_idx = np.r_[site_idx, rng.integers(0, n_hospitals, n - len(site_idx))]
        sites = np.array([str(100 + i) for i in range(n_hospitals)])[site_idx]
        h_int = rng.normal(0, 4, n_hospitals)[site_idx]
        h_slope = rng.normal(0, 2, n_hospitals)[site_idx]
    else:
        sites = np.array(['MIMIC-III'] * n)
        h_int = h_slope = np.zeros(n)

    sev = rng.normal(-0.2 if is_eicu else 0.0, 1, n)
    base = rng.normal(0, 1, n)
    age = np.clip(rng.normal(64 if is_eicu else 66, 15, n), 18, 89.9)
    female = rng.random(n) < (0.45 if is_eicu else 0.43)
    df = pd.DataFrame({
        'stay_id': np.arange(n) + (3_000_000 if is_eicu else 200_000),
        'patient_id': [f'{dataset}-{i}' for i in range(n)],
        'site': sites,
        'ehr_system': 'eICU' if is_eicu else rng.choice(['carevue', 'metavision'], n),
        'unit_type': rng.choice(['MICU', 'SICU', 'CCU', 'CSRU'], n),
        'age': age,
        'gender_raw': np.where(female, 'Female' if is_eicu else 'F',
                               'Male' if is_eicu else 'M'),
        'los_hours': 24 + rng.exponential(48, n),
    })
    if is_eicu:
        hosp = pd.DataFrame({'site': np.unique(sites)})
        hosp['hospital_beds'] = rng.choice(['<100', '100 - 249', '250 - 499', '>= 500'],
                                           len(hosp))
        hosp['hospital_teaching'] = rng.choice(['true', 'false'], len(hosp))
        hosp['hospital_region'] = rng.choice(['Midwest', 'South', 'West', 'Northeast'],
                                             len(hosp))
        df = df.merge(hosp, on='site', how='left')
    else:
        for c in ['hospital_beds', 'hospital_teaching', 'hospital_region']:
            df[c] = None

    hr = 85 + 8 * sev + rng.normal(0, 8, n) - (3 if is_eicu else 0)
    rr = 19 + 2 * sev + rng.normal(0, 3, n)
    spo2 = np.clip(96.5 - 0.8 * sev + rng.normal(0, 1.5, n), 80, 100)
    temp = 37 + 0.3 * sev + rng.normal(0, 0.5, n)
    for name, mu, sd in [('heart_rate', hr, 8 + 2 * np.abs(sev)), ('resp_rate', rr, 3.5),
                         ('spo2', spo2, 1.5), ('temperature', temp, 0.4)]:
        sd = np.abs(sd * np.ones(n) + rng.normal(0, 0.5, n))
        df[f'{name}_mean'] = mu
        df[f'{name}_std'] = sd
        df[f'{name}_min'] = mu - 2 * sd
        df[f'{name}_max'] = mu + 2 * sd
        df[f'{name}_nhours'] = rng.integers(18, 25, n)
    df.loc[rng.random(n) < 0.15, [c for c in df.columns if c.startswith('temperature_')
                                  and not c.endswith('nhours')]] = np.nan

    labs = {
        'sodium': 138.5 + rng.normal(0, 4, n), 'potassium': 4.1 + rng.normal(0, 0.5, n),
        'chloride': 104 + rng.normal(0, 5, n), 'bicarbonate': 24 - 1.5 * sev + rng.normal(0, 3, n),
        'creatinine': np.exp(0.2 + 0.4 * sev + rng.normal(0, 0.5, n)),
        'bun': np.exp(3.0 + 0.4 * sev + rng.normal(0, 0.5, n)),
        'hemoglobin': 10.5 - 0.5 * sev + rng.normal(0, 1.5, n) + (1.2 if is_eicu else 0),
        'wbc': np.exp(2.3 + 0.2 * sev + rng.normal(0, 0.4, n)),
        'platelet': np.exp(5.3 + rng.normal(0, 0.5, n)),
        'glucose': 135 + 15 * sev + rng.normal(0, 35, n),
        'lactate': np.exp(0.6 + 0.4 * sev + rng.normal(0, 0.4, n)),
    }
    for lab, v in labs.items():
        lo, hi = S.LAB_RANGES[lab]
        v = np.clip(v, lo, hi)
        miss = 0.55 if lab == 'lactate' else (0.25 if lab in ('chloride', 'platelet') else 0.1)
        df[lab] = np.where(rng.random(n) < miss, np.nan, v)

    vaso = rng.random(n) < _sigmoid(1.2 * sev - 0.6 * base - 1.8)
    antihyp = rng.random(n) < _sigmoid(1.0 * base - 2.0)
    df['vasopressor_any'] = vaso.astype(int)
    df['antihypertensive_iv_any'] = antihyp.astype(int)
    df['diuretic_iv_any'] = (rng.random(n) < 0.2).astype(int)

    # patient-level "true" cuff pressures, with hospital-specific relationships
    sbp = (121 + 11 * base - (6 + h_slope) * sev + 0.25 * (age - 65) + 4 * antihyp
           - 6 * vaso + h_int + rng.normal(0, 6, n))
    dbp = 62 + 6 * base - 2.5 * sev - 0.15 * (age - 65) + 0.5 * h_int + rng.normal(0, 4, n)
    has_art = rng.random(n) < _sigmoid(1.0 * sev - (0.6 if is_eicu else 1.0))
    has_ni_hist = rng.random(n) < 0.95
    for window, drift, noise in [('hist', 0, 4), ('tgt', rng.normal(0, 6, n), 7)]:
        vals = {'ni': (sbp + drift + rng.normal(0, noise, n),
                       dbp + 0.4 * drift + rng.normal(0, noise / 2, n))}
        vals['art'] = (vals['ni'][0] + 6 + rng.normal(0, 4, n),
                       vals['ni'][1] - 3 + rng.normal(0, 2, n))
        for m in S.MEASURED_MODALITIES:
            s, d = vals[m]
            avail = has_art if m == 'art' else (has_ni_hist if window == 'hist'
                                                else rng.random(n) < 0.93)
            mb = (s + 2 * d) / 3
            if window == 'hist':
                for b, v in [('sbp', s), ('dbp', d), ('mbp', mb)]:
                    df[f'hist_{b}_{m}_mean'] = np.where(avail, v, np.nan)
                    df[f'hist_{b}_{m}_min'] = np.where(avail, v - np.abs(rng.normal(12, 4, n)),
                                                       np.nan)
                    df[f'hist_{b}_{m}_last'] = np.where(avail, v + rng.normal(0, 6, n), np.nan)
                df[f'hist_nhours_{m}'] = np.where(avail, rng.integers(8, 25, n), 0)
                df[f'hist_hypo_hours_{m}'] = np.where(
                    avail, rng.binomial(df[f'hist_nhours_{m}'], _sigmoid((66 - mb) / 3)), 0)
            else:
                nh = np.where(avail, rng.integers(0, 25, n), 0)
                for b, v in [('sbp', s), ('dbp', d), ('mbp', mb)]:
                    df[f'tgt_{b}_{m}'] = np.where(nh > 0, v, np.nan)
                df[f'tgt_nhours_{m}'] = nh
                df[f'tgt_hypo_hours_{m}'] = rng.binomial(nh, _sigmoid((66 - mb) / 3))
    for prefix in ['hist', 'tgt']:
        for b in S.BP_TYPES:
            stats = ['mean', 'min', 'last'] if prefix == 'hist' else ['']
            for st in stats:
                col = lambda m: f'{prefix}_{b}_{m}' + (f'_{st}' if st else '')  # noqa: E731
                df[col('any')] = df[[col('ni'), col('art')]].mean(axis=1)
        df[f'{prefix}_nhours_any'] = df[[f'{prefix}_nhours_ni', f'{prefix}_nhours_art']].max(axis=1)
        df[f'{prefix}_hypo_hours_any'] = df[[f'{prefix}_hypo_hours_ni',
                                             f'{prefix}_hypo_hours_art']].max(axis=1)
    return df[S.EXTRACT_COLUMNS]


def write_synthetic_cache(cfg, n_mimic=3000, n_eicu=4000, n_hospitals=25, seed=0):
    if not cfg.get('allow_synthetic'):
        raise RuntimeError('Refusing to write synthetic data: this config does not set '
                           'allow_synthetic: true (use configs/revision_smoke.yaml)')
    for ds in ('mimic', 'eicu'):
        p = extract.cache_paths(cfg, ds)['data']
        if p.exists() and not extract.is_synthetic(cfg, ds):
            raise RuntimeError(f'{p} holds REAL extracted data; refusing to overwrite it')
    rng = np.random.default_rng(seed)
    for ds, n in [('mimic', n_mimic), ('eicu', n_eicu)]:
        df = _dataset(n, rng, ds, n_hospitals)
        paths = extract.cache_paths(cfg, ds)
        df.to_parquet(paths['data'], index=False)
        flow = {'n_icu_stays': int(n * 1.6), 'n_patients': int(n * 1.3),
                'n_first_stay': int(n * 1.3), 'n_age_eligible': int(n * 1.2),
                'n_los_eligible': n, 'n_sampled': n}
        paths['flow'].write_text(json.dumps(flow, indent=2))
        paths['sql'].write_text(f'{extract.SYNTHETIC_MARKER}: not extracted from BigQuery\n')
    return {ds: extract.cache_paths(cfg, ds)['data'] for ds in ('mimic', 'eicu')}
