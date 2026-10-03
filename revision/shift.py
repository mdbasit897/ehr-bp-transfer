"""Quantifying dataset shift and attributing the transfer gap.

- Standardised mean differences (scale-free; the submitted Wasserstein distances were
  in raw units, so large-unit features dominated).
- Cross-fitted domain classifier (how separable are the datasets?).
- Gap decomposition: covariate shift vs. change in the feature->BP relationship.
- DerSimonian-Laird random-effects pooling of per-hospital performance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import ClusterSampler, bootstrap, metrics_for


def smd_table(Xs, Xt, name_s='MIMIC-III', name_t='eICU'):
    rows = []
    for c in Xs.columns:
        a, b = Xs[c].dropna(), Xt[c].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = np.sqrt((a.var() + b.var()) / 2)
        rows.append({
            'feature': c,
            f'mean_{name_s}': a.mean(), f'sd_{name_s}': a.std(),
            f'mean_{name_t}': b.mean(), f'sd_{name_t}': b.std(),
            'SMD': (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan,
            'variance_ratio': b.var() / a.var() if a.var() > 0 else np.nan,
            f'missing_{name_s}': Xs[c].isna().mean(),
            f'missing_{name_t}': Xt[c].isna().mean(),
            'KS': stats.ks_2samp(a, b).statistic,
        })
    df = pd.DataFrame(rows)
    df['abs_SMD'] = df['SMD'].abs()
    return df.sort_values('abs_SMD', ascending=False).reset_index(drop=True)


def _domain_classifier(kind, seed):
    if kind == 'tree':
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                              random_state=seed)
    if kind == 'linear':
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                             StandardScaler(), LogisticRegression(max_iter=2000))
    raise ValueError(kind)


def domain_propensity(Xs, Xt, seed, n_splits=5, kind='linear'):
    """Cross-fitted P(target | x) for every row; returns (p_source, p_target, auc).

    kind='linear' (logistic regression) is the default for importance weights: trees can
    separate the databases on recording resolution (e.g. Fahrenheit-converted vs charted
    Celsius temperatures), which says nothing about case-mix. kind='tree' is kept as a
    sensitivity analysis and for the leave-one-feature-out artifact search."""
    X = pd.concat([Xs, Xt], ignore_index=True)
    d = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    p = np.zeros(len(X))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, d):
        clf = _domain_classifier(kind, seed)
        clf.fit(X.iloc[tr], d[tr])
        p[te] = clf.predict_proba(X.iloc[te])[:, 1]
    return p[:len(Xs)], p[len(Xs):], float(roc_auc_score(d, p))


def benjamini_hochberg(p):
    """BH-adjusted q-values (NaNs kept)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def importance_weights(p_source, n_source, n_target, clip_quantile=0.99):
    p = np.clip(p_source, 1e-4, 1 - 1e-4)
    w = p / (1 - p) * (n_source / n_target)
    w = np.minimum(w, np.quantile(w, clip_quantile))
    w = w / w.mean()
    ess = w.sum() ** 2 / np.sum(w ** 2)
    return w, float(ess)


def gap_decomposition(task, y_s, p_s_oof, w_s, clusters_s, y_t, p_t_ext, p_t_local,
                      clusters_t, n_boot, seed):
    """Metrics for four evaluation settings and the differences between them.

    source            model on held-out source rows (internal validation)
    source_reweighted source rows reweighted to look like the target in X:
                      performance expected under PURE covariate shift
    target_external   source model on target (what we observed)
    target_local      model trained within the target (achievable in the target)

    covariate_shift_effect   = source_reweighted - source
    conditional_shift_effect = target_external - source_reweighted
                               (change in P(BP | features) and anything not in X)
    transfer_loss            = target_external - target_local
    """
    fn = metrics_for(task)
    ys, ps, ws = map(np.asarray, (y_s, p_s_oof, w_s))
    yt, pe, pl = map(np.asarray, (y_t, p_t_ext, p_t_local))

    def stat(i_s, i_t):
        m = {'source': fn(ys[i_s], ps[i_s]),
             'source_reweighted': fn(ys[i_s], ps[i_s], ws[i_s]),
             'target_external': fn(yt[i_t], pe[i_t]),
             'target_local': fn(yt[i_t], pl[i_t])}
        out = {}
        for setting, vals in m.items():
            for k, v in vals.items():
                out[f'{setting}|{k}'] = v
        for k in m['source']:
            out[f'covariate_shift_effect|{k}'] = m['source_reweighted'][k] - m['source'][k]
            out[f'conditional_shift_effect|{k}'] = (m['target_external'][k]
                                                    - m['source_reweighted'][k])
            out[f'total_gap|{k}'] = m['target_external'][k] - m['source'][k]
            out[f'transfer_loss|{k}'] = m['target_external'][k] - m['target_local'][k]
        return out

    res = bootstrap(stat, [ClusterSampler(clusters_s), ClusterSampler(clusters_t)],
                    n_boot, seed)
    res[['component', 'metric']] = res['metric'].str.split('|', expand=True)
    return res[['component', 'metric', 'estimate', 'ci_low', 'ci_high']]


def dersimonian_laird(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    k = len(est)
    if k < 2:
        return {'k': k}
    w = 1 / se ** 2
    fixed = np.sum(w * est) / np.sum(w)
    Q = np.sum(w * (est - fixed) ** 2)
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / C)
    wr = 1 / (se ** 2 + tau2)
    pooled = np.sum(wr * est) / np.sum(wr)
    se_p = np.sqrt(1 / np.sum(wr))
    t = stats.t.ppf(0.975, max(k - 2, 1))
    return {'k': k, 'pooled': pooled, 'ci_low': pooled - 1.96 * se_p,
            'ci_high': pooled + 1.96 * se_p, 'tau2': tau2, 'tau': np.sqrt(tau2),
            'I2': max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, 'Q': Q,
            'pi_low': pooled - t * np.sqrt(tau2 + se_p ** 2),
            'pi_high': pooled + t * np.sqrt(tau2 + se_p ** 2)}
