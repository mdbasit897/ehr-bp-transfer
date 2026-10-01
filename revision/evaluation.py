"""Metrics, grouped cross-validation and (cluster) bootstrap confidence intervals.

Addresses R2.4 / R3 (no uncertainty) and R1.3 (the submitted version used a single
random 80/20 split although the paper described grouped 5-fold CV).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import models as M


# ---------------------------------------------------------------------------
# Metrics (all support sample weights, needed for the covariate-shift analysis)
# ---------------------------------------------------------------------------

def regression_metrics(y, p, w=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    w = w / w.sum()
    err = p - y
    mse = np.sum(w * err ** 2)
    ybar, pbar = np.sum(w * y), np.sum(w * p)
    sst = np.sum(w * (y - ybar) ** 2)
    var_p = np.sum(w * (p - pbar) ** 2)
    cov = np.sum(w * (p - pbar) * (y - ybar))
    slope = cov / var_p if var_p > 1e-12 else np.nan
    return {
        'RMSE': float(np.sqrt(mse)),
        'MAE': float(np.sum(w * np.abs(err))),
        'R2': float(1 - mse / sst) if sst > 0 else np.nan,
        'bias': float(np.sum(w * err)),                 # mean(pred - obs)
        'cal_slope': float(slope),                      # obs ~ a + b * pred
        'cal_intercept': float(ybar - slope * pbar) if np.isfinite(slope) else np.nan,
        'pearson_r': float(cov / np.sqrt(var_p * sst)) if var_p > 0 and sst > 0 else np.nan,
        'within_10mmHg_pct': float(100 * np.sum(w * (np.abs(err) <= 10))),
    }


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _logistic_calibration(y, p, w, iters=25):
    """Weighted logistic recalibration y ~ a + b * logit(p) via Newton-Raphson."""
    x = _logit(p)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        W = w * mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        g = X.T @ (w * (y - mu))
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def classification_metrics(y, p, w=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    wn = w / w.sum()
    both = len(np.unique(y)) == 2
    a, b = _logistic_calibration(y, p, wn * len(y)) if both else (np.nan, np.nan)
    obs, exp = np.sum(wn * y), np.sum(wn * p)
    return {
        'AUROC': float(roc_auc_score(y, p, sample_weight=w)) if both else np.nan,
        'AUPRC': float(average_precision_score(y, p, sample_weight=w)) if both else np.nan,
        'Brier': float(np.sum(wn * (p - y) ** 2)),
        'prevalence': float(obs),
        'OE_ratio': float(obs / exp) if exp > 0 else np.nan,
        'cal_intercept': a,
        'cal_slope': b,
    }


def metrics_for(task):
    return regression_metrics if task == 'regression' else classification_metrics


PRIMARY_METRICS = {'regression': ['R2', 'RMSE', 'MAE', 'bias', 'cal_slope'],
                   'classification': ['AUROC', 'AUPRC', 'Brier', 'OE_ratio', 'cal_slope']}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class ClusterSampler:
    """Resamples whole clusters (patients or hospitals) with replacement."""

    def __init__(self, clusters):
        codes, uniques = pd.factorize(pd.Series(clusters))
        order = np.argsort(codes, kind='stable')
        bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
        self.members = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniques))]
        self.n = len(codes)
        self.singletons = len(uniques) == self.n

    def draw(self, rng):
        if self.singletons:
            return rng.integers(0, self.n, self.n)
        pick = rng.integers(0, len(self.members), len(self.members))
        return np.concatenate([self.members[i] for i in pick])

    def full(self):
        return np.arange(self.n)


def bootstrap(stat_fn, samplers, n_boot, seed, alpha=0.05):
    """stat_fn(*index_arrays) -> dict of scalars. Returns DataFrame with estimate and
    percentile CI for every key. Several samplers = independent resampling of several
    datasets (e.g. source and target for a transfer gap)."""
    point = stat_fn(*[s.full() for s in samplers])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        try:
            draws.append(stat_fn(*[s.draw(rng) for s in samplers]))
        except ValueError:
            continue
    boot = pd.DataFrame(draws)
    rows = []
    for k, v in point.items():
        col = boot[k].dropna() if k in boot else pd.Series(dtype=float)
        lo, hi = (col.quantile(alpha / 2), col.quantile(1 - alpha / 2)) if len(col) else \
            (np.nan, np.nan)
        rows.append({'metric': k, 'estimate': v, 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(rows)


def metric_ci(y, p, task, clusters, n_boot, seed, w=None):
    fn = metrics_for(task)
    y, p = np.asarray(y), np.asarray(p)
    w = None if w is None else np.asarray(w)

    def stat(idx):
        return fn(y[idx], p[idx], None if w is None else w[idx])
    return bootstrap(stat, [ClusterSampler(clusters)], n_boot, seed)


def paired_diff_ci(y, p_a, p_b, task, clusters, n_boot, seed):
    """CI for metric(p_a) - metric(p_b) on the same rows."""
    fn = metrics_for(task)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))

    def stat(idx):
        ma, mb = fn(y[idx], p_a[idx]), fn(y[idx], p_b[idx])
        return {k: ma[k] - mb[k] for k in ma}
    return bootstrap(stat, [ClusterSampler(clusters)], n_boot, seed)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def grouped_folds(groups, n_splits, seed):
    """K folds with no group (patient or hospital) split across folds."""
    uniq = pd.unique(pd.Series(groups))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of_group = pd.Series(np.arange(len(uniq)) % n_splits, index=uniq[perm])
    fold = fold_of_group.reindex(groups).to_numpy()
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


def oof_predictions(model_name, prep, n_splits, seed, n_jobs=-1, group_by='patient',
                    imputer='median'):
    groups = prep.groups if group_by == 'patient' else prep.sites
    p = np.full(len(prep.y), np.nan)
    for tr, te in grouped_folds(groups, n_splits, seed):
        model = M.make_model(model_name, prep.task, prep.target, seed, n_jobs, imputer)
        M.fit(model, prep.X.iloc[tr], prep.y[tr])
        p[te] = M.predict(model, prep.X.iloc[te], prep.task)
    return p


def fit_full(model_name, prep, seed, n_jobs=-1, sample_weight=None, imputer='median'):
    model = M.make_model(model_name, prep.task, prep.target, seed, n_jobs, imputer)
    return M.fit(model, prep.X, prep.y, sample_weight)
