"""Publication figures and LaTeX tables, generated only from saved result CSVs.

Every number in the revised manuscript should be pasted from tables/*.tex so that the
text, tables and code cannot disagree again.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .experiments import SCENARIO_LABELS  # noqa: E402
from .models import MODEL_LABELS  # noqa: E402

log = logging.getLogger(__name__)

# Okabe-Ito colour-blind-safe palette
C = {'mimic_internal': '#0072B2', 'eicu_local': '#009E73', 'eicu_local_newhosp': '#56B4E9',
     'mimic_to_eicu': '#D55E00', 'eicu_to_mimic': '#CC79A7', 'grey': '#7f7f7f'}
MARK = {'mimic_internal': 'o', 'eicu_local': 's', 'eicu_local_newhosp': 'D',
        'mimic_to_eicu': '^', 'eicu_to_mimic': 'v'}
SHORT = {'mimic_internal': 'MIMIC internal', 'eicu_local': 'eICU local',
         'eicu_local_newhosp': 'eICU unseen hospitals', 'mimic_to_eicu': 'MIMIC→eICU',
         'eicu_to_mimic': 'eICU→MIMIC'}
TARGET_LABEL = {'sbp': 'SBP', 'dbp': 'DBP', 'mbp': 'MAP', 'hypotension': 'Hypotension'}
COL1, COL2 = 3.5, 7.16  # IEEE column / page width, inches


def _style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5,
        'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'legend.frameon': False, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25,
        'grid.linewidth': 0.5, 'lines.linewidth': 1.0, 'savefig.dpi': 300,
        'pdf.fonttype': 42, 'ps.fonttype': 42})


def _read(out, sub, name):
    p = Path(out) / sub / f'{name}.csv'
    return pd.read_csv(p) if p.exists() else None


def _savefig(fig, out, name):
    d = Path(out) / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(d / f'{name}.png', bbox_inches='tight')
    plt.close(fig)


def fmt_ci(est, lo, hi, digits=3):
    if pd.isna(est):
        return '--'
    if pd.isna(lo):
        return f'{est:.{digits}f}'
    return f'{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]'


def _tex(df, out, name, caption, label, col_format=None):
    d = Path(out) / 'tables'
    d.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(index=False, escape=True, column_format=col_format)
    body = body.replace('->', '$\\rightarrow$').replace('→', '$\\rightarrow$')
    body = body.replace('±', '$\\pm$')
    tex = ('\\begin{table}[t]\n\\centering\n\\scriptsize\n'
           f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
           '\\resizebox{\\columnwidth}{!}{%\n' + body + '}\n\\end{table}\n')
    (d / f'{name}.tex').write_text(tex)


def _pivot_ci(df, index, columns, metric, digits=3):
    d = df[df['metric'] == metric].copy()
    d['cell'] = [fmt_ci(e, l, h, digits) for e, l, h in
                 zip(d['estimate'], d['ci_low'], d['ci_high'])]
    return d.pivot_table(index=index, columns=columns, values='cell', aggfunc='first')


# ---------------------------------------------------------------------------
# Figures (no caption text inside images: captions live in figures/figures.tex)
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    'heart_rate_mean': 'Heart rate, mean', 'heart_rate_std': 'Heart rate, SD',
    'heart_rate_min': 'Heart rate, min', 'heart_rate_max': 'Heart rate, max',
    'resp_rate_mean': 'Resp. rate, mean', 'resp_rate_std': 'Resp. rate, SD',
    'spo2_mean': 'SpO$_2$, mean', 'spo2_min': 'SpO$_2$, min',
    'temperature_mean': 'Temperature, mean', 'temperature_max': 'Temperature, max',
    'sodium': 'Sodium', 'potassium': 'Potassium', 'chloride': 'Chloride',
    'bicarbonate': 'Bicarbonate', 'creatinine': 'Creatinine', 'bun': 'BUN',
    'hemoglobin': 'Haemoglobin', 'wbc': 'WBC', 'platelet': 'Platelets',
    'glucose': 'Glucose', 'lactate': 'Lactate', 'vasopressor_any': 'Vasopressor',
    'antihypertensive_iv_any': 'IV antihypertensive', 'diuretic_iv_any': 'IV diuretic',
    'age': 'Age', 'gender': 'Female sex',
}
FEATURE_SET_LABELS = {
    'no_bp_core_meds': 'Vitals + labs + meds', 'no_bp_core': 'Vitals + labs',
    'no_bp_no_labs': 'Vitals + meds', 'no_bp_no_meds': 'Vitals + labs',
    'no_bp_no_vitals': 'Labs + meds', 'bp_history_only': 'BP history only',
    'full': 'All + BP history',
}


def _err(est, lo, hi):
    """Error-bar lengths; percentile CIs need not contain the estimate, so clip at 0."""
    est, lo, hi = (np.asarray(v, float) for v in (est, lo, hi))
    return [np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)]


def fig_flow(out):
    flow = _read(out, 'cohort', 'flow')
    if flow is None:
        return
    steps = [('n_icu_stays', 'All ICU stays'),
             ('n_first_stay', 'First ICU stay per patient'),
             ('n_age_eligible', 'Age 18-89 years'),
             ('n_los_eligible', 'In ICU at 24 h (index time)'),
             ('n_with_target_sbp', 'Cuff BP during 24-48 h'),
             ('n_analysed_sbp', 'Analysed')]
    cols = [0.66, 0.9]
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(steps) + 0.2)
    top = len(steps) - 0.2
    for x, (_, r) in zip(cols, flow.iterrows()):
        ax.text(x, top, 'MIMIC-III' if r['dataset'] == 'mimic' else 'eICU', ha='center',
                va='bottom', fontweight='bold')
    for i, (key, lab) in enumerate(steps):
        y = len(steps) - 1 - i
        ax.text(0.0, y, lab, ha='left', va='center', fontsize=7.5)
        for x, (_, r) in zip(cols, flow.iterrows()):
            n = r.get(key)
            if pd.notna(n):
                ax.text(x, y, f'{int(n):,}', ha='center', va='center', fontsize=7.5,
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='black', lw=0.5))
            if i:
                ax.annotate('', xy=(x, y + 0.27), xytext=(x, y + 0.73),
                            arrowprops=dict(arrowstyle='->', lw=0.5))
    _savefig(fig, out, 'fig_cohort_flow')


def fig_models_forest(out, targets=('sbp', 'dbp', 'mbp', 'hypotension')):
    df = _read(out, 'models_transfer', 'metrics_ci')
    if df is None:
        return
    targets = [t for t in targets if t in set(df['target'])]
    scen = ['mimic_internal', 'eicu_local', 'eicu_local_newhosp', 'mimic_to_eicu',
            'eicu_to_mimic']
    models = [m for m in MODEL_LABELS if m in set(df['model']) and m != 'mean']
    fig, axes = plt.subplots(1, len(targets), figsize=(COL2, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        metric = 'AUROC' if t == 'hypotension' else 'R2'
        d = df[(df['target'] == t) & (df['metric'] == metric)]
        for j, s in enumerate(scen):
            ds = d[d['scenario'] == s].set_index('model').reindex(models)
            y = np.arange(len(models)) + (j - 2) * 0.15
            ax.errorbar(ds['estimate'], y, xerr=_err(ds['estimate'], ds['ci_low'],
                                                     ds['ci_high']),
                        fmt=MARK[s], ms=3, color=C[s], lw=0.8, capsize=0, label=SHORT[s])
        ax.axvline(0.5 if metric == 'AUROC' else 0, color=C['grey'], lw=0.6, ls='--')
        ax.set_title(f'{TARGET_LABEL[t]} ({"AUROC" if metric == "AUROC" else "$R^2$"})')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS[m].replace(' (submitted)', '') for m in models])
        ax.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.2, wspace=0.12)
    _savefig(fig, out, 'fig_models_transfer')


def fig_ladder(out):
    df = _read(out, 'ladder', 'ladder_metrics')
    if df is None:
        return
    d = df[df['metric'] == 'R2']
    steps = list(dict.fromkeys(d['step']))
    targets = [t for t in ['sbp', 'dbp', 'mbp'] if t in set(d['target'])]
    fig, axes = plt.subplots(1, len(targets), figsize=(COL2, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        for j, s in enumerate(['mimic_internal', 'eicu_local', 'mimic_to_eicu']):
            ds = d[(d['target'] == t) & (d['scenario'] == s)].set_index('step').reindex(steps)
            x = np.arange(len(steps)) + (j - 1) * 0.15
            ax.errorbar(x, ds['estimate'], yerr=_err(ds['estimate'], ds['ci_low'],
                                                     ds['ci_high']),
                        fmt=MARK[s] + '-', ms=3, color=C[s], lw=0.8, label=SHORT[s])
        ax.axhline(0, color=C['grey'], lw=0.6, ls='--')
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([s.split()[0] for s in steps])
        ax.set_title(TARGET_LABEL[t])
    axes[0].set_ylabel('$R^2$ (95% CI)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.22, wspace=0.1)
    _savefig(fig, out, 'fig_harmonization_ladder')


def fig_decomposition(out, target='sbp'):
    df = _read(out, 'decomposition', 'gap_decomposition')
    if df is None or target not in set(df['target']):
        return
    d = df[df['target'] == target]
    if 'weights' in d:
        d = d[d['weights'] == 'linear']
    comps = ['source', 'source_reweighted', 'target_external', 'target_local']
    labels = ['MIMIC-III internal', 'MIMIC-III, eICU case-mix', 'MIMIC-III → eICU',
              'eICU local']
    colors = [C['mimic_internal'], '#E69F00', C['mimic_to_eicu'], C['eicu_local']]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 1.6), sharey=True)
    for ax, metric, xl in zip(axes, ['R2', 'RMSE'], ['$R^2$', 'RMSE (mmHg)']):
        dm = d[d['metric'] == metric].set_index('component').reindex(comps)
        y = np.arange(len(comps))
        for k in range(len(comps)):
            ax.errorbar(dm['estimate'].iloc[k], y[k],
                        xerr=[[e] for e in _err(dm['estimate'].iloc[k], dm['ci_low'].iloc[k],
                                                dm['ci_high'].iloc[k])],
                        fmt='o', ms=4, color=colors[k], lw=1)
        ax.set_xlabel(xl)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
    axes[0].invert_yaxis()   # shared y-axis: invert once
    fig.subplots_adjust(wspace=0.08)
    _savefig(fig, out, f'fig_gap_decomposition_{target}')


def fig_hospitals(out, target='sbp'):
    per = _read(out, 'hospitals', 'per_hospital')
    re_ = _read(out, 'hospitals', 'random_effects')
    if per is None or target not in set(per['target']):
        return
    p = per[per['target'] == target].sort_values('ext_bias')
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6),
                             gridspec_kw={'width_ratios': [1, 1.4]})
    ax = axes[0]
    ax.scatter(p['ext_RMSE'], p['loho_RMSE'], s=np.sqrt(p['n']) / 1.5, alpha=0.7,
               color=C['mimic_to_eicu'], edgecolor='none')
    lo = min(p['ext_RMSE'].min(), p['loho_RMSE'].min()) - 0.5
    hi = max(p['ext_RMSE'].max(), p['loho_RMSE'].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=C['grey'], ls='--', lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('RMSE, MIMIC-III model (mmHg)')
    ax.set_ylabel('RMSE, model from other\neICU hospitals (mmHg)')
    ax.set_title('(a) Per-hospital error', loc='left')
    ax = axes[1]
    y = np.arange(len(p))
    ax.errorbar(p['ext_bias'], y, xerr=1.96 * p['ext_bias_se'], fmt='o', ms=1.5,
                color=C['mimic_to_eicu'], lw=0.4, elinewidth=0.4)
    if re_ is not None:
        r = re_[(re_['target'] == target) & (re_['model_source'] == 'ext')
                & (re_['metric'] == 'bias')]
        if len(r):
            r = r.iloc[0]
            ax.axvspan(r['pi_low'], r['pi_high'], color=C['grey'], alpha=0.15, lw=0)
            ax.axvline(r['pooled'], color='black', lw=0.8)
            ax.text(0.98, 0.03, f'pooled {r["pooled"]:.1f} mmHg\n'
                                f'$I^2$ = {100 * r["I2"]:.0f}%\n'
                                f'95% PI {r["pi_low"]:.1f} to {r["pi_high"]:.1f}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5,
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
    ax.axvline(0, color=C['grey'], ls='--', lw=0.6)
    ax.set_yticks([])
    ax.grid(False, axis='y')
    ax.set_xlabel('Bias, predicted − observed (mmHg)')
    ax.set_ylabel(f'eICU hospitals (n = {len(p)})')
    ax.set_title('(b) Per-hospital bias, MIMIC-III model', loc='left')
    fig.subplots_adjust(wspace=0.3)
    _savefig(fig, out, f'fig_hospitals_{target}')


def fig_calibration(out, target='sbp', model='xgb', bins=10):
    pe = Path(out) / 'predictions' / f'eicu_{target}_{model}.parquet'
    pm = Path(out) / 'predictions' / f'mimic_{target}_{model}.parquet'
    if not (pe.exists() and pm.exists()):
        return
    e, m = pd.read_parquet(pe), pd.read_parquet(pm)
    series = [('mimic_internal', m['y'], m['mimic_internal']),
              ('eicu_local', e['y'], e['eicu_local']),
              ('mimic_to_eicu', e['y'], e['mimic_to_eicu'])]
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    pts = []
    for s, y, p in series:
        q = pd.qcut(p, bins, duplicates='drop')
        g = pd.DataFrame({'y': y, 'p': p, 'q': q}).groupby('q', observed=True).mean()
        pts.append(g)
        ax.plot(g['p'], g['y'], MARK[s] + '-', ms=3, color=C[s], label=SHORT[s])
    allv = pd.concat([pd.concat([g['p'], g['y']]) for g in pts])
    lo, hi = allv.min() - 3, allv.max() + 3
    ax.plot([lo, hi], [lo, hi], color=C['grey'], ls='--', lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    unit = '' if target == 'hypotension' else ' (mmHg)'
    ax.set_xlabel(f'Predicted {TARGET_LABEL[target]}, decile mean{unit}')
    ax.set_ylabel(f'Observed {TARGET_LABEL[target]}{unit}')
    ax.legend(loc='upper left')
    _savefig(fig, out, f'fig_calibration_{target}_{model}')


def fig_shift(out, top=20):
    df = _read(out, 'shift', 'feature_shift_smd')
    if df is None:
        return
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(COL1, 0.13 * len(d) + 0.6))
    ax.scatter(d['SMD'], range(len(d)), color=C['mimic_to_eicu'], s=10)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=C['grey'], ls=':', lw=0.6)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in d['feature']])
    ax.set_xlabel('Standardised mean difference (eICU − MIMIC-III)')
    _savefig(fig, out, 'fig_feature_shift')


def fig_ablation(out):
    df = _read(out, 'ablation', 'feature_set_ablation')
    if df is None:
        return
    if 'target' not in df:
        df = df.assign(target='sbp')
    panels = [(t, m) for t, m in [('sbp', 'R2'), ('hypotension', 'AUROC')]
              if t in set(df['target'])]
    fig, axes = plt.subplots(1, len(panels), figsize=(COL2, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (t, metric) in zip(axes, panels):
        d = df[(df['metric'] == metric) & (df['cohort'] == 'all') & (df['target'] == t)]
        sets = list(dict.fromkeys(d['feature_set']))
        for j, s in enumerate(['mimic_internal', 'eicu_local', 'mimic_to_eicu']):
            ds = d[d['scenario'] == s].set_index('feature_set').reindex(sets)
            y = np.arange(len(sets)) + (j - 1) * 0.22
            ax.barh(y, ds['estimate'], height=0.22, color=C[s], label=SHORT[s],
                    xerr=_err(ds['estimate'], ds['ci_low'], ds['ci_high']),
                    error_kw=dict(lw=0.6))
        ax.set_yticks(range(len(sets)))
        ax.set_yticklabels([FEATURE_SET_LABELS.get(x, x) for x in sets])
        ax.axvline(0.5 if metric == 'AUROC' else 0, color='black', lw=0.5)
        if metric == 'AUROC':
            ax.set_xlim(left=0.5)
        ax.set_xlabel(f'{"AUROC" if metric == "AUROC" else "$R^2$"} (95% CI), '
                      f'{TARGET_LABEL[t]}')
    axes[0].invert_yaxis()   # shared y-axis: invert once
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.25, wspace=0.08)
    _savefig(fig, out, 'fig_feature_ablation')


def fig_modality_agreement(out):
    df = _read(out, 'modality_agreement', 'cuff_vs_arterial')
    if df is None:
        return
    bps = ['sbp', 'dbp', 'mbp']
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 1.9),
                             gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axes[0]
    for j, (name, col) in enumerate([('MIMIC-III', C['mimic_internal']),
                                     ('eICU', C['eicu_local'])]):
        d = df[df['comparison'] == name].set_index('bp').reindex(bps)
        y = np.arange(len(bps)) + (j - 0.5) * 0.25
        ax.errorbar(d['estimate'], y, xerr=_err(d['estimate'], d['ci_low'], d['ci_high']),
                    fmt='o', ms=4, color=col, lw=1, label=name)
    ax.axvline(0, color=C['grey'], ls='--', lw=0.6)
    ax.set_yticks(range(len(bps)))
    ax.set_yticklabels([TARGET_LABEL[b] for b in bps])
    ax.invert_yaxis()
    ax.set_xlabel('Cuff − arterial BP (mmHg), same stays')
    ax.set_title('(a) Cuff vs arterial, within database', loc='left')
    ax.legend(loc='lower right')
    ax = axes[1]
    d = df[df['comparison'] == 'eICU minus MIMIC-III'].set_index('bp').reindex(bps)
    ax.errorbar(d['estimate'], range(len(bps)),
                xerr=_err(d['estimate'], d['ci_low'], d['ci_high']),
                fmt='o', ms=4, color=C['mimic_to_eicu'], lw=1)
    ax.axvline(0, color=C['grey'], ls='--', lw=0.6)
    ax.set_yticks(range(len(bps)))
    ax.set_yticklabels([TARGET_LABEL[b] for b in bps])
    ax.invert_yaxis()
    ax.set_xlabel('eICU − MIMIC-III (mmHg)')
    ax.set_title('(b) Between databases', loc='left')
    fig.subplots_adjust(wspace=0.35)
    _savefig(fig, out, 'fig_modality_agreement')


FIGURE_CAPTIONS = [
    ('fig_cohort_flow', False,
     'Cohort selection in MIMIC-III and eICU. The unit of analysis is the first ICU stay '
     'per patient. Index time is 24 h after ICU admission; the prediction target is '
     'non-invasive (cuff) blood pressure during hours 24-48. eICU was reduced to a '
     'deterministic random sample of 40,000 stays for computational reasons.'),
    ('fig_harmonization_ladder', True,
     'Harmonization ladder: $R^2$ (95\\% CI) for SBP, DBP and MAP as one analysis choice '
     'is changed at a time. S0, the submitted task (features and target from the same '
     'window; MIMIC-III any BP modality, eICU arterial line); S1, cuff BP in both '
     'datasets; S2, arterial BP in both; S3, prospective task (features 0-24 h, target '
     '24-48 h) without BP history; S4, S3 with observation-window BP history. '
     'MIMIC internal and eICU local use patient-grouped 5-fold cross-validation; '
     'MIMIC$\\rightarrow$eICU trains on all MIMIC-III stays. CIs from 1,000 bootstrap '
     'resamples (patients for MIMIC-III, hospitals for eICU).'),
    ('fig_models_transfer', True,
     'Discrimination of all models under five validation settings: MIMIC-III internal, '
     'eICU local (patient-grouped cross-validation), eICU unseen hospitals '
     '(hospital-grouped cross-validation), MIMIC-III$\\rightarrow$eICU and '
     'eICU$\\rightarrow$MIMIC-III. $R^2$ for SBP, DBP and MAP; AUROC for hypotension '
     '($\\geq$2 hourly MAP values $<$65 mmHg during hours 24-48). Persistence predicts '
     'the observation-window BP. Bars show 95\\% bootstrap CIs.'),
    ('fig_gap_decomposition_sbp', True,
     'Decomposition of the MIMIC-III$\\rightarrow$eICU performance gap for SBP. '
     "``eICU case-mix'' reweights held-out MIMIC-III predictions by the estimated density "
     'ratio so the covariate distribution matches eICU (performance expected under pure '
     'covariate shift); the remaining difference to MIMIC-III$\\rightarrow$eICU reflects '
     'a change in the feature--BP relationship. eICU local is the achievable reference.'),
    ('fig_hospitals_sbp', True,
     'Per-hospital performance in eICU hospitals with $\\geq$100 stays. (a) RMSE of the '
     'MIMIC-III model versus a model trained on other eICU hospitals (hospital-grouped '
     'cross-validation); points on the diagonal indicate no MIMIC-specific transfer loss; '
     'point size reflects the number of stays. (b) Mean prediction bias of the MIMIC-III '
     'model per hospital (95\\% CI), with the random-effects pooled estimate, '
     'heterogeneity ($I^2$) and 95\\% prediction interval.'),
    ('fig_feature_shift', False,
     'Standardised mean differences of the model features between eICU and MIMIC-III '
     '(dotted lines at $\\pm$0.1).'),
    ('fig_feature_ablation', True,
     'Feature-set ablation on a fixed cohort (every feature set is evaluated on the same '
     'patients): $R^2$ for SBP and AUROC for hypotension, 95\\% bootstrap CIs. '
     "``All + BP history'' is the clinically relevant forecasting model."),
    ('fig_modality_agreement', True,
     'Cuff versus arterial blood pressure in stays with both measurements during the '
     'first 24 h (window means; not time-matched pairs). (a) Mean cuff minus arterial '
     'difference within each database. (b) Between-database difference in (a); a '
     'non-zero value indicates that cuff pressures are measured or recorded differently, '
     'since arterial pressures agree across databases. 95\\% cluster-bootstrap CIs.'),
    ('fig_hospitals_dbp', True,
     'As Fig.~\\ref{fig:hospitals_sbp}, for diastolic blood pressure.'),
    ('fig_calibration_sbp_xgb', False,
     'Calibration of SBP predictions (XGBoost): observed versus predicted SBP by decile '
     'of predicted value. The dashed line is perfect calibration.'),
]


def write_figure_tex(out):
    """LaTeX snippets for every produced figure. Two-column figures use figure*."""
    d = Path(out) / 'figures'
    parts = ['% Generated by revision/reporting.py -- paste into the manuscript and edit.',
             '% Copy figures/*.pdf next to main.tex (vector PDF; do not use the PNGs).', '']
    for name, wide, cap in FIGURE_CAPTIONS:
        if not (d / f'{name}.pdf').exists():
            continue
        env = 'figure*' if wide else 'figure'
        width = '\\textwidth' if wide else '\\columnwidth'
        parts += [f'\\begin{{{env}}}[t]', '\\centering',
                  f'\\includegraphics[width={width}]{{{name}.pdf}}',
                  f'\\caption{{{cap}}}', f'\\label{{fig:{name.replace("fig_", "")}}}',
                  f'\\end{{{env}}}', '']
    (d / 'figures.tex').write_text('\n'.join(parts))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

COMPONENT_LABELS = {
    'source': 'MIMIC-III internal',
    'source_reweighted': 'MIMIC-III reweighted to eICU case-mix',
    'target_external': 'MIMIC-III $\\rightarrow$ eICU (observed)',
    'target_local': 'eICU local (reference)',
    'covariate_shift_effect': '$\\Delta$ case-mix (covariate shift)',
    'conditional_shift_effect': '$\\Delta$ feature--BP relationship',
    'total_gap': 'Total gap (observed $-$ internal)',
    'transfer_loss': 'Transfer loss (observed $-$ local)',
}
METHOD_LABELS = {
    'none': 'None (MIMIC-III model as is)', 'intercept_only': 'Intercept (offset) only',
    'linear': 'Linear', 'platt': 'Logistic (Platt)', 'isotonic': 'Isotonic',
    'local_model_on_calibration_hospitals': 'New model trained on calibration hospitals',
    'unweighted': 'Unweighted',
    'importance_weighted': 'Importance-weighted (tree weights; pre-audit run)',
    'importance_weighted_linear': 'Importance-weighted (logistic weights)',
    'importance_weighted_tree': 'Importance-weighted (tree weights, sensitivity)',
}


def _method_label(m):
    return METHOD_LABELS.get(m, str(m).replace('_', ' '))
METRIC_LABELS = {'R2': '$R^2$', 'RMSE': 'RMSE', 'bias': 'Bias', 'AUROC': 'AUROC',
                 'OE_ratio': 'O/E', 'cal_slope': 'Cal. slope', 'MAE': 'MAE'}
LADDER_LABELS = {
    'S0': 'S0: submitted task (concurrent; MIMIC any vs eICU arterial)',
    'S1': 'S1: concurrent, cuff vs cuff',
    'S2': 'S2: concurrent, arterial vs arterial',
    'S3': 'S3: prospective, no BP history',
    'S4': 'S4: prospective, with BP history',
}


def _latexsafe(label):
    return label.replace('$_2$', '2')


def _tex_raw(body_df, out, name, caption, label):
    """Like _tex, but cell text may already contain LaTeX math (no escaping)."""
    d = Path(out) / 'tables'
    d.mkdir(parents=True, exist_ok=True)
    body = body_df.to_latex(index=False, escape=False)
    tex = ('\\begin{table}[t]\n\\centering\n\\scriptsize\n'
           f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
           '\\resizebox{\\columnwidth}{!}{%\n' + body + '}\n\\end{table}\n')
    (d / f'{name}.tex').write_text(tex)


def _scen_cols(pv):
    pv = pv[[c for c in SCENARIO_LABELS if c in pv.columns]]
    pv.columns = [SHORT[c].replace('→', '$\\rightarrow$') for c in pv.columns]
    return pv


def tables(out, primary_target='sbp'):
    t1 = _read(out, 'cohort', 'table1')
    if t1 is not None:
        _tex(t1.rename(columns={'characteristic': 'Characteristic'}), out, 'table1_cohort',
             'Cohort characteristics (first ICU stay per patient). Values are mean (SD), '
             'median [IQR] or \\%.', 'tab:cohort')

    mt = _read(out, 'models_transfer', 'metrics_ci')
    if mt is not None:
        for target in mt['target'].unique():
            for metric, digits in ([('AUROC', 3), ('OE_ratio', 2), ('cal_slope', 2)]
                                   if target == 'hypotension'
                                   else [('R2', 3), ('RMSE', 2), ('bias', 2)]):
                pv = _pivot_ci(mt[mt['target'] == target], 'model', 'scenario', metric, digits)
                pv = _scen_cols(pv.reindex([m for m in MODEL_LABELS if m in pv.index]))
                pv.index = [MODEL_LABELS[m] for m in pv.index]
                suffix = '' if metric in ('R2', 'AUROC') else f'_{metric.lower()}'
                unit = ' (mmHg)' if metric in ('RMSE', 'bias') else ''
                _tex_raw(pv.reset_index().rename(columns={'index': 'Model'}), out,
                         f'table_models_{target}{suffix}',
                         f'{TARGET_LABEL[target]}: {METRIC_LABELS[metric]}{unit} [95\\% CI] '
                         f'by model and validation setting.', f'tab:models_{target}{suffix}')

    lad = _read(out, 'ladder', 'ladder_metrics')
    if lad is not None:
        lad = lad.assign(step=lad['step'].str.split().str[0])
        for target in ['sbp', 'dbp', 'mbp']:
            d = lad[lad['target'] == target]
            if d.empty:
                continue
            pv = _pivot_ci(d, 'step', 'scenario', 'R2')
            pv = _scen_cols(pv.reindex(list(dict.fromkeys(d['step']))))
            pv.index = [LADDER_LABELS.get(s, s) for s in pv.index]
            _tex_raw(pv.reset_index().rename(columns={'index': 'Step'}), out,
                     f'table_ladder_{target}',
                     f'Harmonization ladder, {TARGET_LABEL[target]} $R^2$ [95\\% CI].',
                     f'tab:ladder_{target}')

    dec = _read(out, 'decomposition', 'gap_decomposition')
    if dec is not None:
        if 'weights' not in dec:
            dec = dec.assign(weights='tree')
        for weights in dec['weights'].unique():
            for target in dec['target'].unique():
                metrics = (['AUROC', 'OE_ratio'] if target == 'hypotension'
                           else ['R2', 'RMSE', 'bias'])
                d = dec[(dec['target'] == target) & (dec['weights'] == weights)
                        & dec['metric'].isin(metrics)]
                d = d.assign(cell=[fmt_ci(e, l, h, 3) for e, l, h in
                                   zip(d['estimate'], d['ci_low'], d['ci_high'])])
                pv = d.pivot_table(index='component', columns='metric', values='cell',
                                   aggfunc='first')[metrics]
                pv = pv.reindex([c for c in COMPONENT_LABELS if c in pv.index])
                pv.index = [COMPONENT_LABELS[c] for c in pv.index]
                pv.columns = [METRIC_LABELS[m] for m in pv.columns]
                sfx = '' if weights == 'linear' else f'_{weights}'
                _tex_raw(pv.reset_index().rename(columns={'index': 'Setting / component'}),
                         out, f'table_decomposition_{target}{sfx}',
                         f'Decomposition of the MIMIC-III$\\rightarrow$eICU gap, '
                         f'{TARGET_LABEL[target]} ({weights} density-ratio weights).',
                         f'tab:decomposition_{target}{sfx}')

    ab = _read(out, 'ablation', 'feature_set_ablation')
    if ab is not None:
        if 'target' not in ab:
            ab = ab.assign(target='sbp')
        for target in ab['target'].unique():
            metric = 'AUROC' if target == 'hypotension' else 'R2'
            for cohort in ab['cohort'].unique():
                d = ab[(ab['cohort'] == cohort) & (ab['target'] == target)]
                pv = _pivot_ci(d, 'feature_set', 'scenario', metric)
                pv = _scen_cols(pv.reindex(list(dict.fromkeys(d['feature_set']))))
                nf = d.groupby('feature_set')['n_features'].first()
                pv.insert(0, 'Features', nf.reindex(pv.index).astype(int).to_numpy())
                pv.index = [FEATURE_SET_LABELS.get(f, f) for f in pv.index]
                _tex_raw(pv.reset_index().rename(columns={'index': 'Feature set'}), out,
                         f'table_ablation_{target}_{cohort}',
                         f'Feature-set ablation, {TARGET_LABEL[target]} '
                         f'{METRIC_LABELS[metric]} [95\\% CI], same patients for every set '
                         f'({cohort.replace("_", " ")}).', f'tab:ablation_{target}_{cohort}')

    method_order = list(METHOD_LABELS)
    for sub in ['recalibration', 'domain_adaptation']:
        s = _read(out, sub, 'summary')
        if s is None:
            continue
        s = s.assign(_o=s['method'].map({m: i for i, m in enumerate(method_order)}))
        s = s.sort_values(['target', '_o'])
        for group, metrics in [('bp', ['R2', 'RMSE', 'bias']),
                               ('hypotension', ['AUROC', 'OE_ratio', 'cal_slope'])]:
            g = s[(s['target'] == 'hypotension') == (group == 'hypotension')]
            if g.empty:
                continue
            tab = pd.DataFrame({
                'Target': g['target'].map(TARGET_LABEL).to_numpy(),
                'Method': g['method'].map(_method_label).to_numpy()})
            for m in metrics:
                unit = ' (mmHg)' if m in ('RMSE', 'bias') else ''
                tab[METRIC_LABELS[m] + unit] = [
                    f'{mu:.3f} $\\pm$ {sd:.3f}' if pd.notna(mu) else '--'
                    for mu, sd in zip(g.get(f'{m}_mean', np.nan), g.get(f'{m}_std', np.nan))]
            _tex_raw(tab, out, f'table_{sub}_{group}',
                     f'{sub.replace("_", " ").capitalize()}: MIMIC-III model evaluated on '
                     f'held-out eICU hospitals (mean $\\pm$ SD over hospital splits). ``None'' '
                     f'is evaluated on the held-out hospitals only, so it differs slightly '
                     f'from the full-eICU estimate.', f'tab:{sub}_{group}')

    ma = _read(out, 'modality_agreement', 'cuff_vs_arterial')
    if ma is not None:
        d = ma.assign(cell=[fmt_ci(e, l, h, 2) for e, l, h in
                            zip(ma['estimate'], ma['ci_low'], ma['ci_high'])])
        pv = d.pivot_table(index='comparison', columns='bp', values='cell', aggfunc='first')
        pv = pv.reindex(['MIMIC-III', 'eICU', 'eICU minus MIMIC-III'])[['sbp', 'dbp', 'mbp']]
        pv.columns = [TARGET_LABEL[c] for c in pv.columns]
        n = ma.set_index(['comparison', 'bp'])['n_stays']
        pv.insert(0, 'Stays', [int(n.get((c, 'sbp'), 0)) for c in pv.index])
        _tex_raw(pv.reset_index().rename(columns={'comparison': ''}), out,
                 'table_modality_agreement',
                 'Cuff minus arterial BP (mmHg) in stays with both, first 24 h [95\\% CI].',
                 'tab:modality_agreement')

    fs = _read(out, 'hospitals', 'feature_shift_vs_loss')
    if fs is not None:
        top = (fs[fs['outcome'].isin(['ext_bias', 'transfer_loss_AUROC'])]
               .sort_values('p_value').groupby('target').head(5))
        top = top.assign(Target=top['target'].map(TARGET_LABEL),
                         Feature=top['feature'].map(lambda f: _latexsafe(
                             FEATURE_LABELS.get(f, f))),
                         Outcome=top['outcome'].map({'ext_bias': 'Bias',
                                                     'transfer_loss_AUROC': '$\\Delta$AUROC'}),
                         rho=top['spearman_rho'].map('{:.2f}'.format),
                         q=top['q_value_bh'].map('{:.3f}'.format))
        _tex_raw(top[['Target', 'Feature', 'Outcome', 'rho', 'q']]
                 .rename(columns={'rho': 'Spearman $\\rho$', 'q': '$q$ (BH)'}), out,
                 'table_feature_shift_vs_loss',
                 "Hospital-level association between each feature's shift from MIMIC-III "
                 "(standardised mean difference) and the MIMIC-III model's error in that "
                 "hospital; top five per target, Benjamini-Hochberg q-values.",
                 'tab:feature_shift_vs_loss')

    re_ = _read(out, 'hospitals', 'random_effects')
    if re_ is not None:
        r = re_.assign(Target=re_['target'].map(TARGET_LABEL),
                       Model=re_['model_source'].map({'ext': 'MIMIC-III',
                                                      'loho': 'Other eICU hospitals'}),
                       Pooled=[fmt_ci(e, l, h, 2) for e, l, h in
                               zip(re_['pooled'], re_['ci_low'], re_['ci_high'])],
                       PI=[f'{a:.1f} to {b:.1f}' for a, b in zip(re_['pi_low'], re_['pi_high'])],
                       I2=(100 * re_['I2']).map('{:.0f}\\%'.format))
        _tex_raw(r[['Target', 'Model', 'metric', 'k', 'Pooled', 'PI', 'I2']]
                 .rename(columns={'metric': 'Metric', 'k': 'Hospitals',
                                  'PI': '95\\% PI', 'I2': '$I^2$'}), out,
                 'table_hospital_random_effects',
                 'Per-hospital performance in eICU pooled with random effects '
                 '(DerSimonian--Laird).', 'tab:hospital_random_effects')


def make_all(out, primary_target='sbp', primary_model='xgb'):
    _style()
    for fn in [fig_flow, fig_models_forest, fig_ladder, fig_shift, fig_ablation,
               fig_modality_agreement]:
        try:
            fn(out)
        except Exception as e:  # figures must never kill a long run
            log.exception('figure %s failed: %s', fn.__name__, e)
    for fn, kw in [(fig_decomposition, {'target': primary_target}),
                   (fig_hospitals, {'target': primary_target}),
                   (fig_hospitals, {'target': 'dbp'}),
                   (fig_calibration, {'target': primary_target, 'model': primary_model})]:
        try:
            fn(out, **kw)
        except Exception as e:
            log.exception('figure %s failed: %s', fn.__name__, e)
    for stale in (Path(out) / 'tables').glob('*.tex'):   # no leftovers from older runs
        stale.unlink()
    tables(out, primary_target)
    write_figure_tex(out)
