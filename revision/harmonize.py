"""Cleaning, data-quality checks and construction of analysis tables.

Key rules:
- Unit of analysis = first ICU stay per patient; grouping variable = patient .
- Rows are defined ONLY by target availability, never by feature availability, so
  every feature set is evaluated on the same patients .
- BP-history features are refused in the concurrent design (would be leakage).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema as S

log = logging.getLogger(__name__)


def map_gender(series):
    s = series.astype('string').str.strip().str.lower().str[:1]
    return s.map({'m': 0.0, 'f': 1.0}).astype(float)


def clean(df, dataset):
    df = df.copy()
    df['dataset'] = dataset
    df['gender'] = map_gender(df['gender_raw'])
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    for c in df.columns:
        if c not in ('patient_id', 'site', 'ehr_system', 'unit_type', 'gender_raw',
                     'hospital_beds', 'hospital_teaching', 'hospital_region', 'dataset'):
            # BigQuery INT64 arrives as nullable Int64 (pd.NA); use plain float + NaN
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    df['stay_id'] = df['stay_id'].astype('int64')
    # eICU reports age in whole years; MIMIC age is computed to the day. Use whole years
    # in both, otherwise age resolution alone identifies the database (domain AUC ~0.99).
    df['age'] = np.floor(df['age'])
    df['patient_id'] = df['patient_id'].astype(str)
    df['site'] = df['site'].astype(str)
    if df['stay_id'].duplicated().any():
        raise ValueError(f'{dataset}: duplicated stay_id')
    if df['patient_id'].duplicated().any():
        raise ValueError(f'{dataset}: more than one stay per patient (expected first stay only)')
    return df


def data_quality_report(df, dataset):
    """Per-column checks. The submitted pipeline would have failed these for eICU
    gender (constant) and eICU temperature (Fahrenheit mixed with Celsius)."""
    feature_cols = (S.VITAL_FEATURES + S.LAB_FEATURES + S.MED_FEATURES + S.DEMO_FEATURES
                    + [f'hist_{b}_{m}_mean' for m in S.MODALITIES for b in S.BP_TYPES]
                    + [f'tgt_{b}_{m}' for m in S.MODALITIES for b in S.BP_TYPES])
    rows = []
    for c in feature_cols:
        x = df[c]
        nonnull = x.dropna()
        row = {'dataset': dataset, 'column': c, 'n': len(x),
               'missing_frac': float(x.isna().mean()),
               'n_unique': int(nonnull.nunique()),
               'mean': float(nonnull.mean()) if len(nonnull) else np.nan,
               'sd': float(nonnull.std()) if len(nonnull) > 1 else np.nan,
               'p01': float(nonnull.quantile(0.01)) if len(nonnull) else np.nan,
               'p99': float(nonnull.quantile(0.99)) if len(nonnull) else np.nan}
        flags = []
        if len(nonnull) and row['n_unique'] <= 1:
            flags.append('CONSTANT')
        if row['missing_frac'] > 0.5:
            flags.append('MOSTLY_MISSING')
        base = c.replace('_mean', '').replace('_max', '').replace('_min', '')
        rng = S.LAB_RANGES.get(c) or S.VITAL_RANGES.get(base)
        if rng and len(nonnull) and (nonnull.min() < rng[0] or nonnull.max() > rng[1]):
            flags.append('OUT_OF_RANGE')
        row['flags'] = ';'.join(flags)
        rows.append(row)
    rep = pd.DataFrame(rows)
    for _, r in rep[rep['flags'] != ''].iterrows():
        log.warning('%s data quality: %s -> %s', dataset, r['column'], r['flags'])
    if df['gender'].nunique() < 2:
        log.error('%s: gender has a single level -- encoding bug?', dataset)
    return rep


def cross_dataset_checks(df_a, df_b, name_a='mimic', name_b='eicu'):
    """Flag features whose medians differ by >2x between databases (unit mismatch)."""
    rows = []
    for c in S.VITAL_FEATURES + S.LAB_FEATURES + ['age']:
        ma, mb = df_a[c].median(), df_b[c].median()
        ratio = mb / ma if pd.notna(ma) and ma != 0 else np.nan
        mismatch = pd.notna(ratio) and (ratio > 2 or ratio < 0.5) and not c.endswith('_std')
        flag = 'POSSIBLE_UNIT_MISMATCH' if mismatch else ''
        rows.append({'feature': c, f'median_{name_a}': ma, f'median_{name_b}': mb,
                     'ratio': ratio, 'flag': flag})
        if flag:
            log.warning('Cross-dataset check: %s median %s=%.3g vs %s=%.3g', c,
                        name_a, ma, name_b, mb)
    return pd.DataFrame(rows)


@dataclass
class Prepared:
    dataset: str
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray       # patient id (CV grouping)
    sites: np.ndarray        # hospital id (cluster bootstrap / LOHO)
    meta: pd.DataFrame       # stay-level info for stratification
    target: str
    task: str                # 'regression' | 'classification'
    feature_set: str
    design: str
    modality: str
    n_before_target_filter: int

    def subset(self, idx):
        idx = np.asarray(idx)
        return Prepared(self.dataset, self.X.iloc[idx].reset_index(drop=True), self.y[idx],
                        self.groups[idx], self.sites[idx],
                        self.meta.iloc[idx].reset_index(drop=True), self.target, self.task,
                        self.feature_set, self.design, self.modality,
                        self.n_before_target_filter)

    def with_features(self, cols):
        return Prepared(self.dataset, self.X[cols].copy(), self.y, self.groups, self.sites,
                        self.meta, self.target, self.task, self.feature_set, self.design,
                        self.modality, self.n_before_target_filter)


def task_of(target):
    return 'classification' if target == 'hypotension' else 'regression'


def target_rows(df, target, design, modality, cfg):
    """Return (row mask, y). Row selection depends only on the target."""
    a = cfg['analysis']
    if design == 'prospective':
        enough = df[f'tgt_nhours_{modality}'] >= a['min_target_hours']
        if target == 'hypotension':
            y = (df[f'tgt_hypo_hours_{modality}'] >= a['min_hypo_hours']).astype(float)
        else:
            y = df[f'tgt_{target}_{modality}']
    elif design == 'concurrent':
        if target == 'hypotension':
            raise ValueError('hypotension forecasting requires the prospective design')
        enough = df[f'hist_nhours_{modality}'] >= a['min_target_hours']
        y = df[f'hist_{target}_{modality}_mean']
    else:
        raise ValueError(design)
    mask = enough & y.notna()
    return mask.to_numpy(), y.to_numpy(dtype=float)


def bp_history_frame(df, modality):
    out = pd.DataFrame(index=df.index)
    for f in S.BP_HISTORY_FEATURES:
        _, b, stat = f.split('_')
        out[f] = df[f'hist_{b}_{modality}_{stat}']
    return out


def _row_cap_mask(df, max_rows):
    if not max_rows or len(df) <= max_rows:
        return np.ones(len(df), dtype=bool)
    # deterministic: keep the max_rows stays with the smallest hash of stay_id
    h = pd.util.hash_pandas_object(df['stay_id'].astype('int64'), index=False).to_numpy()
    keep = np.argsort(h, kind='stable')[:max_rows]
    mask = np.zeros(len(df), dtype=bool)
    mask[keep] = True
    return mask


def prepare(df, cfg, target, feature_set, design=None, modality=None, max_rows='config'):
    a = cfg['analysis']
    design = design or a['design']
    ds = df['dataset'].iloc[0]
    modality = modality or a['target_modality'][ds]
    if design == 'concurrent' and S.uses_bp_history(feature_set):
        raise ValueError(f'{feature_set} uses BP history, which leaks the target in the '
                         f'concurrent design')
    if max_rows == 'config':
        max_rows = a['max_rows']

    mask, y = target_rows(df, target, design, modality, cfg)
    n_before = int(len(df))
    sub = df.loc[mask].reset_index(drop=True)
    y = y[mask]
    cap = _row_cap_mask(sub, max_rows)
    sub, y = sub.loc[cap].reset_index(drop=True), y[cap]

    feats = S.FEATURE_SETS[feature_set]
    hist = bp_history_frame(sub, modality)
    X = pd.concat([sub[[f for f in feats if f in sub.columns]],
                   hist[[f for f in feats if f in hist.columns]]], axis=1)[feats]
    meta_cols = ['stay_id', 'patient_id', 'site', 'ehr_system', 'unit_type',
                 'hospital_beds', 'hospital_teaching', 'hospital_region', 'age', 'gender',
                 'los_hours']
    return Prepared(ds, X.astype(float), y, sub['patient_id'].to_numpy(),
                    sub['site'].to_numpy(), sub[meta_cols].copy(), target, task_of(target),
                    feature_set, design, modality, n_before)


def shared_features(prepared_list, max_missing):
    """Drop features missing in > max_missing of rows in ANY dataset. This is the
    data-driven, reproducible version of the 'intersection' feature set."""
    cols = list(prepared_list[0].X.columns)
    keep, dropped = [], []
    for c in cols:
        worst = max(p.X[c].isna().mean() for p in prepared_list)
        (keep if worst <= max_missing else dropped).append(c)
    if dropped:
        log.info('Dropping features missing in >%.0f%% of a dataset: %s',
                 100 * max_missing, dropped)
    return keep, dropped


def complete_case_mask(prepared, cols):
    return prepared.X[cols].notna().all(axis=1).to_numpy()
