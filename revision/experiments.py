"""All revision experiments. Each method writes CSVs under output_dir/<experiment>/.

Design requirement (docs/PIPELINE_NOTES.md) -> experiment
  R1.1  fixed cohort across feature sets ..................... ablation()
  R1.2  train/evaluate locally in eICU ........................ models_transfer(), ladder()
  R1.3  consistent cohort numbers / unit of analysis .......... cohort()
  R1.4  explicit temporal task ................................ config 'design', ladder()
  R2.3/R4  clinically meaningful task (hypotension forecast) .. target 'hypotension'
  R2.4  simple baselines, single learners, uncertainty ........ models_transfer()
  R3    'fundamental barriers' not demonstrated ............... decomposition(), shift()
  R3    eICU is multi-centre -> per-site analysis ............. hospitals()
  R3    measurement explanation demonstrated, not asserted .... modality_agreement(),
                                                                hospitals() feature attribution
  (fix) recalibration / domain adaptation compared fairly ..... recalibration(),
                                                                domain_adaptation()
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

from . import evaluation as E
from . import harmonize as H
from . import models as M
from . import schema as S
from . import shift as SH

log = logging.getLogger(__name__)

SCENARIO_LABELS = {
    'mimic_internal': 'MIMIC-III internal (patient-grouped CV)',
    'eicu_local': 'eICU local (patient-grouped CV)',
    'eicu_local_newhosp': 'eICU local, unseen hospitals (hospital-grouped CV)',
    'mimic_to_eicu': 'MIMIC-III -> eICU (external)',
    'eicu_to_mimic': 'eICU -> MIMIC-III (external, reverse)',
}


def _stay_hash(prep):
    return int(pd.util.hash_pandas_object(prep.meta['stay_id'], index=False).sum() % (2 ** 61))


def _key(prep):
    return (prep.dataset, prep.target, prep.feature_set, prep.design, prep.modality,
            tuple(prep.X.columns), _stay_hash(prep))


class Runner:
    def __init__(self, cfg, raw, flows=None):
        self.cfg = cfg
        self.a = cfg['analysis']
        self.seed = cfg['seed']
        self.n_jobs = cfg['n_jobs']
        self.out = Path(cfg['output_dir'])
        self.flows = flows or {}
        self.data = {ds: H.clean(df, ds) for ds, df in raw.items()}
        self._oof_cache, self._ext_cache, self._model_cache = {}, {}, {}

    # ------------------------------------------------------------------ utils
    def _save(self, df, sub, name):
        d = self.out / sub
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / f'{name}.csv', index=False)
        return df

    def pair(self, target, feature_set, design=None, modality=None, max_rows='config'):
        design = design or self.a['design']
        mod = modality or self.a['target_modality']
        ps = H.prepare(self.data['mimic'], self.cfg, target, feature_set, design,
                       mod['mimic'], max_rows)
        pt = H.prepare(self.data['eicu'], self.cfg, target, feature_set, design,
                       mod['eicu'], max_rows)
        cols, _ = H.shared_features([ps, pt], self.a['max_feature_missing'])
        return ps.with_features(cols), pt.with_features(cols)

    def _model_ok(self, model, prep):
        if model != 'persistence':
            return True
        if prep.design != 'prospective':
            return False
        need = ['hist_mbp_min', 'hist_mbp_last'] if prep.task == 'classification' \
            else [f'hist_{prep.target}_mean']
        return all(c in prep.X.columns for c in need)

    def oof(self, model, prep, seed=None, group_by='patient', imputer='median'):
        seed = self.seed if seed is None else seed
        k = (_key(prep), model, seed, group_by, imputer)
        if k not in self._oof_cache:
            t = time.time()
            self._oof_cache[k] = E.oof_predictions(model, prep, self.a['cv_folds'], seed,
                                                   self.n_jobs, group_by, imputer)
            log.info('  OOF %s %s/%s %s (%s) %.0fs', prep.dataset, prep.target,
                     prep.feature_set, model, group_by, time.time() - t)
        return self._oof_cache[k]

    def full_model(self, model, prep, imputer='median'):
        k = (_key(prep), model, imputer)
        if k not in self._model_cache:
            t = time.time()
            self._model_cache[k] = E.fit_full(model, prep, self.seed, self.n_jobs,
                                              imputer=imputer)
            log.info('  fit %s %s/%s %s %.0fs', prep.dataset, prep.target, prep.feature_set,
                     model, time.time() - t)
        return self._model_cache[k]

    def external(self, model, src, tgt, imputer='median'):
        k = (_key(src), _key(tgt), model, imputer)
        if k not in self._ext_cache:
            self._ext_cache[k] = M.predict(self.full_model(model, src, imputer), tgt.X,
                                           tgt.task)
        return self._ext_cache[k]

    def _ci(self, y, p, task, clusters, **labels):
        res = E.metric_ci(y, p, task, clusters, self.a['n_bootstrap'], self.seed)
        for k, v in labels.items():
            res.insert(0, k, v)
        return res

    # ----------------------------------------------------------------- cohort
    def cohort(self):
        """R1.3: one unit of analysis, full flow counts, data-quality checks, Table 1."""
        a = self.a
        flows, quality = [], []
        for ds, df in self.data.items():
            quality.append(H.data_quality_report(df, ds))
            flow = dict(self.flows.get(ds, {}))
            flow['dataset'] = ds
            flow['n_extracted'] = len(df)
            for target in a['targets']:
                try:
                    mask, _ = H.target_rows(df, target, a['design'],
                                            a['target_modality'][ds], self.cfg)
                except ValueError:
                    continue
                flow[f'n_with_target_{target}'] = int(mask.sum())
                flow[f'n_analysed_{target}'] = int(min(mask.sum(), a['max_rows'] or 1e12))
            flows.append(flow)
        self._save(pd.DataFrame(flows), 'cohort', 'flow')
        self._save(pd.concat(quality), 'cohort', 'data_quality')
        self._save(H.cross_dataset_checks(self.data['mimic'], self.data['eicu']), 'cohort',
                   'cross_dataset_checks')
        self._save(self.table1(), 'cohort', 'table1')

    def table1(self):
        a = self.a
        rows = {}
        for ds, df in self.data.items():
            m = a['target_modality'][ds]
            mask, _ = H.target_rows(df, a['primary_target'], a['design'], m, self.cfg)
            d = df.loc[mask]
            if a['max_rows']:
                d = d.loc[H._row_cap_mask(d, a['max_rows'])]
            t = f'tgt_{{}}_{m}' if a['design'] == 'prospective' else f'hist_{{}}_{m}_mean'

            def msd(x):
                return f'{x.mean():.1f} ({x.std():.1f})'

            def miqr(x):
                return f'{x.median():.1f} [{x.quantile(.25):.1f}-{x.quantile(.75):.1f}]'
            r = {
                'ICU stays (first per patient)': f'{len(d):,}',
                'Hospitals': f'{d["site"].nunique():,}',
                'Age, years': msd(d['age']),
                'Female, %': f'{100 * d["gender"].mean():.1f}',
                'ICU LOS, hours': miqr(d['los_hours']),
                f'SBP target ({m}), mmHg': msd(d[t.format('sbp')]),
                f'DBP target ({m}), mmHg': msd(d[t.format('dbp')]),
                f'MAP target ({m}), mmHg': msd(d[t.format('mbp')]),
                'Arterial line in obs. window, %': f'{100 * (d["hist_nhours_art"] > 0).mean():.1f}',
                'Cuff BP in obs. window, %': f'{100 * (d["hist_nhours_ni"] > 0).mean():.1f}',
                'Vasopressor in obs. window, %': f'{100 * d["vasopressor_any"].mean():.1f}',
            }
            if a['design'] == 'prospective':
                hypo = d[f'tgt_hypo_hours_{m}'] >= a['min_hypo_hours']
                r['Hypotension in target window, %'] = f'{100 * hypo.mean():.1f}'
            for grp, feats in [('vitals', S.VITAL_FEATURES), ('labs', S.LAB_FEATURES)]:
                r[f'Median missingness, {grp}, %'] = f'{100 * d[feats].isna().mean().median():.1f}'
            rows['MIMIC-III' if ds == 'mimic' else 'eICU'] = r
        out = pd.DataFrame(rows)
        out.index.name = 'characteristic'
        return out.reset_index()

    # ------------------------------------------------------------------ shift
    def shift(self):
        a = self.a
        ps, pt = self.pair(a['primary_target'], a['primary_feature_set'])
        smd = SH.smd_table(ps.X, pt.X)
        # Linear AUC = case-mix separability. Tree AUC can also exploit recording
        # resolution; the leave-one-feature-out search (trees) shows which features do.
        auc = SH.domain_propensity(ps.X, pt.X, self.seed, kind='tree')[2]
        auc_lin = SH.domain_propensity(ps.X, pt.X, self.seed, kind='linear')[2]
        smd['domain_auc_all_features'] = auc
        smd['domain_auc_linear'] = auc_lin
        self._save(smd, 'shift', 'feature_shift_smd')
        loo = [{'feature': c, 'domain_auc_without': SH.domain_propensity(
            ps.X.drop(columns=c), pt.X.drop(columns=c), self.seed, kind='tree')[2]}
            for c in ps.X.columns]
        self._save(pd.DataFrame(loo).assign(auc_drop=lambda d: auc - d['domain_auc_without'])
                   .sort_values('auc_drop', ascending=False), 'shift', 'domain_auc_leave_one_out')
        tgt = pd.DataFrame({'dataset': ['MIMIC-III', 'eICU'],
                            'target_mean': [ps.y.mean(), pt.y.mean()],
                            'target_sd': [ps.y.std(), pt.y.std()],
                            'n': [len(ps.y), len(pt.y)]})
        self._save(tgt, 'shift', 'target_shift')
        log.info('Domain classifier AUC (MIMIC vs eICU): trees %.3f, linear %.3f', auc, auc_lin)

    # ------------------------------------------------------- models/transfer
    def models_transfer(self):
        """R1.2 + R2.4: every model, internal / local / external / reverse, with CIs."""
        a = self.a
        all_rows, paired_rows, ci_rows = [], [], []
        pred_dir = self.out / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        for target in a['targets']:
            if target == 'hypotension' and a['design'] != 'prospective':
                continue
            for model in a['models']:
                fs = 'bp_history_only' if model == 'persistence' else a['primary_feature_set']
                ps, pt = self.pair(target, fs)
                if not self._model_ok(model, ps):
                    continue
                log.info('models_transfer: %s / %s', target, model)
                task = ps.task
                lab = {'target': target, 'model': model, 'feature_set': fs}
                preds = {}
                for r in range(a['cv_repeats']):
                    p = self.oof(model, ps, seed=self.seed + r)
                    if r == 0:
                        preds['mimic_internal'] = p
                    m = E.metrics_for(task)(ps.y, p)
                    if model == 'mean':
                        m = {k: (np.nan if k in ('AUROC', 'AUPRC', 'cal_slope', 'cal_intercept',
                                                 'pearson_r') else v) for k, v in m.items()}
                    all_rows.append(pd.DataFrame([{**lab, 'scenario': 'mimic_internal',
                                                   'repeat': r, **m}]))
                preds['eicu_local'] = self.oof(model, pt)
                preds['eicu_local_newhosp'] = self.oof(model, pt, group_by='site')
                preds['mimic_to_eicu'] = self.external(model, ps, pt)
                preds['eicu_to_mimic'] = self.external(model, pt, ps)

                for scen, p in preds.items():
                    on_mimic = scen in ('mimic_internal', 'eicu_to_mimic')
                    prep = ps if on_mimic else pt
                    clusters = prep.groups if on_mimic else prep.sites
                    ci = self._ci(prep.y, p, task, clusters, scenario=scen, **lab)
                    ci['n'] = len(prep.y)
                    ci['n_clusters'] = len(np.unique(clusters))
                    if model == 'mean':
                        # constant per fold: ranking/slope metrics are artifacts of pooling
                        disc = ci['metric'].isin(['AUROC', 'AUPRC', 'cal_slope',
                                                  'cal_intercept', 'pearson_r'])
                        ci.loc[disc, ['estimate', 'ci_low', 'ci_high']] = np.nan
                    ci_rows.append(ci)
                for local in ('eicu_local', 'eicu_local_newhosp'):
                    d = E.paired_diff_ci(pt.y, preds['mimic_to_eicu'], preds[local], task,
                                         pt.sites, a['n_bootstrap'], self.seed)
                    d.insert(0, 'comparison', f'mimic_to_eicu minus {local}')
                    for k, v in lab.items():
                        d.insert(0, k, v)
                    paired_rows.append(d)
                pd.DataFrame({'stay_id': pt.meta['stay_id'], 'site': pt.sites, 'y': pt.y,
                              **{k: v for k, v in preds.items() if k != 'mimic_internal'
                                 and k != 'eicu_to_mimic'}}).to_parquet(
                    pred_dir / f'eicu_{target}_{model}.parquet', index=False)
                pd.DataFrame({'stay_id': ps.meta['stay_id'], 'y': ps.y,
                              'mimic_internal': preds['mimic_internal'],
                              'eicu_to_mimic': preds['eicu_to_mimic']}).to_parquet(
                    pred_dir / f'mimic_{target}_{model}.parquet', index=False)
        self._save(pd.concat(ci_rows), 'models_transfer', 'metrics_ci')
        self._save(pd.concat(all_rows), 'models_transfer', 'internal_repeats')
        self._save(pd.concat(paired_rows), 'models_transfer', 'paired_differences')

    # ----------------------------------------------------------------- ladder
    def ladder(self, model=None):
        """Change one thing at a time, from the submitted task to the revised task."""
        model = model or self.a['primary_model']
        rows = []
        for step in self.cfg['ladder']:
            for target in ['sbp', 'dbp', 'mbp']:
                ps, pt = self.pair(target, step['feature_set'], step['design'],
                                   step['target_modality'])
                log.info('ladder: %s / %s (n=%d, %d)', step['name'], target, len(ps.y),
                         len(pt.y))
                preds = {'mimic_internal': (ps, self.oof(model, ps), ps.groups),
                         'mimic_to_eicu': (pt, self.external(model, ps, pt), pt.sites),
                         'eicu_local': (pt, self.oof(model, pt), pt.sites)}
                for scen, (prep, p, cl) in preds.items():
                    ci = self._ci(prep.y, p, 'regression', cl, scenario=scen, target=target,
                                  step=step['name'], model=model)
                    ci['n'] = len(prep.y)
                    ci['target_mean'] = prep.y.mean()
                    ci['target_sd'] = prep.y.std()
                    rows.append(ci)
        self._save(pd.concat(rows), 'ladder', 'ladder_metrics')

    # ---------------------------------------------------------- decomposition
    def decomposition(self, model=None):
        model = model or self.a['primary_model']
        rows, info = [], []
        for target in self.a['targets']:
            if target == 'hypotension' and self.a['design'] != 'prospective':
                continue
            ps, pt = self.pair(target, self.a['primary_feature_set'])
            # primary: logistic density-ratio weights; sensitivity: tree weights
            for kind in ['linear', 'tree']:
                p_s, _, auc = SH.domain_propensity(ps.X, pt.X, self.seed, kind=kind)
                w, ess = SH.importance_weights(p_s, len(ps.y), len(pt.y))
                res = SH.gap_decomposition(ps.task, ps.y, self.oof(model, ps), w, ps.groups,
                                           pt.y, self.external(model, ps, pt),
                                           self.oof(model, pt), pt.sites,
                                           self.a['n_bootstrap'], self.seed)
                res.insert(0, 'weights', kind)
                res.insert(0, 'target', target)
                rows.append(res)
                info.append({'target': target, 'weights': kind, 'domain_auc': auc,
                             'weights_ess': ess, 'n_source': len(ps.y),
                             'ess_fraction': ess / len(ps.y)})
        self._save(pd.concat(rows), 'decomposition', 'gap_decomposition')
        self._save(pd.DataFrame(info), 'decomposition', 'weights_info')

    # -------------------------------------------------------------- hospitals
    def hospitals(self, model=None):
        """R3: eICU is multi-centre, so evaluate per hospital and relate performance
        loss to measured shift instead of asserting it."""
        a = self.a
        model = model or a['primary_model']
        per_rows, meta_rows, corr_rows, group_rows, feat_rows = [], [], [], [], []
        targets = [t for t in a['targets']
                   if t != 'hypotension' or a['design'] == 'prospective']
        for target in targets:
            ps, pt = self.pair(target, a['primary_feature_set'])
            fn = E.metrics_for(pt.task)
            p_ext = self.external(model, ps, pt)
            p_loho = self.oof(model, pt, group_by='site')
            p_s, p_t, _ = SH.domain_propensity(ps.X, pt.X, self.seed, kind='linear')
            mu_s, sd_s = ps.X.mean(), ps.X.std().replace(0, np.nan)
            meta = pt.meta.reset_index(drop=True)
            site_smd = {}
            for site, idx in meta.groupby('site').indices.items():
                if len(idx) < a['min_hospital_n']:
                    continue
                y = pt.y[idx]
                if pt.task == 'classification' and len(np.unique(y)) < 2:
                    continue
                e_ext, e_loc = p_ext[idx] - y, p_loho[idx] - y
                m_ext, m_loc = fn(y, p_ext[idx]), fn(y, p_loho[idx])
                d_auc = roc_auc_score(np.r_[np.zeros(len(p_s)), np.ones(len(idx))],
                                      np.r_[p_s, p_t[idx]])
                signed = (pt.X.iloc[idx].mean() - mu_s) / sd_s
                site_smd[site] = signed
                smd = signed.abs().mean()
                row = {'target': target, 'site': site, 'n': len(idx),
                       'hospital_beds': meta.loc[idx[0], 'hospital_beds'],
                       'hospital_teaching': meta.loc[idx[0], 'hospital_teaching'],
                       'hospital_region': meta.loc[idx[0], 'hospital_region'],
                       'domain_auc_vs_mimic': d_auc, 'mean_abs_smd_vs_mimic': smd,
                       'target_mean': y.mean()}
                row.update({f'ext_{k}': v for k, v in m_ext.items()})
                row.update({f'loho_{k}': v for k, v in m_loc.items()})
                if pt.task == 'regression':
                    row.update({'ext_bias_se': e_ext.std(ddof=1) / np.sqrt(len(idx)),
                                'loho_bias_se': e_loc.std(ddof=1) / np.sqrt(len(idx)),
                                'ext_MAE_se': np.abs(e_ext).std(ddof=1) / np.sqrt(len(idx)),
                                'loho_MAE_se': np.abs(e_loc).std(ddof=1) / np.sqrt(len(idx))})
                per_rows.append(row)
            per = pd.DataFrame([r for r in per_rows if r['target'] == target])
            if per.empty:
                log.warning('hospitals: no hospital with n >= %d', a['min_hospital_n'])
                continue
            if pt.task == 'regression':
                per['transfer_loss_RMSE'] = per['ext_RMSE'] - per['loho_RMSE']
                for metric in ['bias', 'MAE']:
                    for src in ['ext', 'loho']:
                        r = SH.dersimonian_laird(per[f'{src}_{metric}'],
                                                 per[f'{src}_{metric}_se'])
                        meta_rows.append({'target': target, 'model_source': src,
                                          'metric': metric, **r})
                outcomes = ['ext_RMSE', 'ext_R2', 'transfer_loss_RMSE']
            else:
                per['transfer_loss_AUROC'] = per['ext_AUROC'] - per['loho_AUROC']
                outcomes = ['ext_AUROC', 'transfer_loss_AUROC']
            for shift_col in ['domain_auc_vs_mimic', 'mean_abs_smd_vs_mimic']:
                for oc in outcomes:
                    rho, pval = stats.spearmanr(per[shift_col], per[oc], nan_policy='omit')
                    corr_rows.append({'target': target, 'shift_measure': shift_col,
                                      'outcome': oc, 'spearman_rho': rho, 'p_value': pval,
                                      'n_hospitals': len(per)})
            # Which features' hospital-level shift tracks the transfer error? (R3)
            smd_df = pd.DataFrame(site_smd).T.reindex(per['site'].to_numpy())
            fo = (['ext_bias', 'transfer_loss_RMSE'] if pt.task == 'regression'
                  else ['transfer_loss_AUROC'])
            rows_t = []
            for feat in smd_df.columns:
                for oc in fo:
                    rho, pval = stats.spearmanr(smd_df[feat].to_numpy(), per[oc].to_numpy(),
                                                nan_policy='omit')
                    rows_t.append({'target': target, 'feature': feat, 'outcome': oc,
                                   'spearman_rho': rho, 'p_value': pval,
                                   'n_hospitals': len(per)})
            ft = pd.DataFrame(rows_t)
            ft['q_value_bh'] = np.nan
            for oc in fo:
                m = ft['outcome'] == oc
                ft.loc[m, 'q_value_bh'] = SH.benjamini_hochberg(ft.loc[m, 'p_value'])
            feat_rows.append(ft)
            for col in ['hospital_teaching', 'hospital_beds', 'hospital_region']:
                g = per.groupby(col)[outcomes].median()
                g['n_hospitals'] = per.groupby(col).size()
                g = g.reset_index().rename(columns={col: 'level'})
                g.insert(0, 'characteristic', col)
                g.insert(0, 'target', target)
                group_rows.append(g)
        if per_rows:
            self._save(pd.DataFrame(per_rows), 'hospitals', 'per_hospital')
            self._save(pd.DataFrame(meta_rows), 'hospitals', 'random_effects')
            self._save(pd.DataFrame(corr_rows), 'hospitals', 'shift_vs_performance')
            self._save(pd.concat(group_rows), 'hospitals', 'by_hospital_characteristics')
            self._save(pd.concat(feat_rows).sort_values(['target', 'outcome', 'p_value']),
                       'hospitals', 'feature_shift_vs_loss')

    # --------------------------------------------------------------- ablation
    def ablation(self, model=None):
        """R1.1: every feature set on the SAME patients (rows depend on target only),
        plus a complete-case sensitivity analysis on one fixed complete-case subset."""
        a = self.a
        model = model or a['primary_model']
        rows = []
        sets = [fs for fs in a['feature_sets']
                if a['design'] == 'prospective' or not S.uses_bp_history(fs)]
        targets = [t for t in a['targets']
                   if t != 'hypotension' or a['design'] == 'prospective']
        for target in targets:
            base_s, base_t = self.pair(target, a['primary_feature_set'])
            cc_cols = list(base_s.X.columns)
            cc_s = H.complete_case_mask(base_s, cc_cols)
            cc_t = H.complete_case_mask(base_t, cc_cols)
            for fs in sets:
                ps, pt = self.pair(target, fs)
                assert _stay_hash(ps) == _stay_hash(base_s), 'cohort changed with feature set'
                for cohort, (qs, qt) in {'all': (ps, pt),
                                         'complete_case': (ps.subset(np.where(cc_s)[0]),
                                                           pt.subset(np.where(cc_t)[0]))
                                         }.items():
                    log.info('ablation: %s / %s / %s (n=%d, %d)', target, fs, cohort,
                             len(qs.y), len(qt.y))
                    for scen, (prep, p, cl) in {
                            'mimic_internal': (qs, self.oof(model, qs), qs.groups),
                            'mimic_to_eicu': (qt, self.external(model, qs, qt), qt.sites),
                            'eicu_local': (qt, self.oof(model, qt), qt.sites)}.items():
                        ci = self._ci(prep.y, p, prep.task, cl, scenario=scen,
                                      cohort=cohort, feature_set=fs,
                                      n_features=qs.X.shape[1], model=model, target=target)
                        ci['n'] = len(prep.y)
                        rows.append(ci)
        self._save(pd.concat(rows), 'ablation', 'feature_set_ablation')
        target = a['primary_target']
        self._save(pd.DataFrame({'feature_set': sets,
                                 'features': [';'.join(self.pair(target, fs)[0].X.columns)
                                              for fs in sets]}), 'ablation', 'feature_lists')

    # -------------------------------------------------- recalibration / DA
    def _hospital_splits(self, pt):
        sites = np.unique(pt.sites)
        rng = np.random.default_rng(self.seed)
        for r in range(self.a['n_calibration_splits']):
            cal_sites = rng.choice(sites, max(1, int(round(self.a['calibration_fraction']
                                                           * len(sites)))), replace=False)
            cal = np.isin(pt.sites, cal_sites)
            yield r, np.where(cal)[0], np.where(~cal)[0]

    def recalibration(self, model=None):
        """Recalibrate on a subset of eICU HOSPITALS, evaluate on the other hospitals.
        (The paper said recalibration was fitted on MIMIC; the code fitted it on a random
        20% of eICU rows and evaluated a different subset than the headline result.)"""
        a = self.a
        model = model or a['primary_model']
        rows = []
        # all targets: the DBP/MAP transfer loss is mostly a constant offset
        targets = [t for t in a['targets']
                   if t != 'hypotension' or a['design'] == 'prospective']
        for target in targets:
            ps, pt = self.pair(target, a['primary_feature_set'])
            fn = E.metrics_for(pt.task)
            p = self.external(model, ps, pt)
            for r, cal, ev in self._hospital_splits(pt):
                yc, pc, pe, ye = pt.y[cal], p[cal], p[ev], pt.y[ev]
                out = {'none': pe}
                if pt.task == 'regression':
                    out['intercept_only'] = pe + (yc.mean() - pc.mean())
                    lr = LinearRegression().fit(pc.reshape(-1, 1), yc)
                    out['linear'] = lr.predict(pe.reshape(-1, 1))
                    out['isotonic'] = IsotonicRegression(out_of_bounds='clip').fit(
                        pc, yc).predict(pe)
                else:
                    lc, le = E._logit(pc).reshape(-1, 1), E._logit(pe).reshape(-1, 1)
                    offset = E._logit(np.array([yc.mean()]))[0] - E._logit(np.array([pc.mean()]))[0]
                    out['intercept_only'] = 1 / (1 + np.exp(-(le[:, 0] + offset)))
                    out['platt'] = LogisticRegression().fit(lc, yc).predict_proba(le)[:, 1]
                    out['isotonic'] = IsotonicRegression(out_of_bounds='clip', y_min=0,
                                                         y_max=1).fit(pc, yc).predict(pe)
                local = E.fit_full(model, pt.subset(cal), self.seed, self.n_jobs)
                out['local_model_on_calibration_hospitals'] = M.predict(local, pt.X.iloc[ev],
                                                                        pt.task)
                for method, pred in out.items():
                    rows.append({'target': target, 'split': r, 'method': method,
                                 'n_cal': len(cal), 'n_eval': len(ev), **fn(ye, pred)})
        df = pd.DataFrame(rows)
        self._save(df, 'recalibration', 'per_split')
        num = df.select_dtypes('number').columns.drop(['split', 'n_cal', 'n_eval'])
        summ = df.groupby(['target', 'method'])[list(num)].agg(['mean', 'std'])
        summ.columns = [f'{m}_{s}' for m, s in summ.columns]
        self._save(summ.reset_index(), 'recalibration', 'summary')

    def domain_adaptation(self, model=None):
        """Importance-weighted training vs. unweighted, same model, same evaluation rows.
        Weights use only UNLABELLED features from the calibration hospitals."""
        a = self.a
        model = model or a['primary_model']
        rows = []
        for target in [t for t in a['targets'] if t != 'hypotension']:
            ps, pt = self.pair(target, a['primary_feature_set'])
            fn = E.metrics_for(ps.task)
            base = self.external(model, ps, pt)
            for r, cal, ev in self._hospital_splits(pt):
                rows.append({'target': target, 'split': r, 'method': 'unweighted',
                             **fn(pt.y[ev], base[ev])})
                for kind in ['linear', 'tree']:
                    p_s, _, auc = SH.domain_propensity(ps.X, pt.X.iloc[cal], self.seed,
                                                       kind=kind)
                    w, ess = SH.importance_weights(p_s, len(ps.y), len(cal))
                    wm = E.fit_full(model, ps, self.seed, self.n_jobs, sample_weight=w)
                    p_w = M.predict(wm, pt.X.iloc[ev], pt.task)
                    rows.append({'target': target, 'split': r,
                                 'method': f'importance_weighted_{kind}', 'domain_auc': auc,
                                 'ess_fraction': ess / len(ps.y), **fn(pt.y[ev], p_w)})
        df = pd.DataFrame(rows)
        self._save(df, 'domain_adaptation', 'per_split')
        num = df.select_dtypes('number').columns.drop(['split'])
        summ = df.groupby(['target', 'method'])[list(num)].agg(['mean', 'std'])
        summ.columns = [f'{m}_{s}' for m, s in summ.columns]
        self._save(summ.reset_index(), 'domain_adaptation', 'summary')

    # ------------------------------------------------------ modality agreement
    def modality_agreement(self):
        """Is the eICU cuff-BP offset a measurement difference? (R3)

        In stays with BOTH cuff and arterial BP during the observation window, compare
        cuff minus arterial between the databases. Arterial pressures agree across the
        databases (ladder S2), so a between-database difference in cuff-minus-arterial
        points to how cuff pressure is measured or charted, not to the patients.
        Window means, not time-matched pairs: the comparison is within-patient but not
        within-minute."""
        a = self.a
        rows, diffs = [], {}
        for ds, df in self.data.items():
            cl_col = 'patient_id' if ds == 'mimic' else 'site'
            for b in S.BP_TYPES:
                ni, art = df[f'hist_{b}_ni_mean'], df[f'hist_{b}_art_mean']
                ok = (ni.notna() & art.notna()
                      & (df['hist_nhours_ni'] >= a['min_target_hours'])
                      & (df['hist_nhours_art'] >= a['min_target_hours'])).to_numpy()
                d = (ni - art).to_numpy()[ok]
                cl = df[cl_col].to_numpy()[ok]
                diffs[(ds, b)] = (d, cl)
                res = E.bootstrap(lambda i, d=d: {'mean_cuff_minus_arterial':
                                                  float(np.mean(d[i]))},
                                  [E.ClusterSampler(cl)], a['n_bootstrap'], self.seed)
                r = res.iloc[0]
                rows.append({'comparison': 'MIMIC-III' if ds == 'mimic' else 'eICU',
                             'bp': b, 'n_stays': int(ok.sum()),
                             'mean_cuff': float(ni[ok].mean()),
                             'mean_arterial': float(art[ok].mean()),
                             'sd_difference': float(np.std(d, ddof=1)),
                             'estimate': r['estimate'], 'ci_low': r['ci_low'],
                             'ci_high': r['ci_high']})
        for b in S.BP_TYPES:
            (dm, cm), (de, ce) = diffs[('mimic', b)], diffs[('eicu', b)]
            res = E.bootstrap(lambda i, j, dm=dm, de=de: {'delta': float(np.mean(de[j])
                                                                         - np.mean(dm[i]))},
                              [E.ClusterSampler(cm), E.ClusterSampler(ce)],
                              a['n_bootstrap'], self.seed)
            r = res.iloc[0]
            rows.append({'comparison': 'eICU minus MIMIC-III', 'bp': b,
                         'n_stays': len(dm) + len(de), 'estimate': r['estimate'],
                         'ci_low': r['ci_low'], 'ci_high': r['ci_high']})
        self._save(pd.DataFrame(rows), 'modality_agreement', 'cuff_vs_arterial')

    # ------------------------------------------------------------- imputation
    def imputation(self, model=None):
        a = self.a
        model = model or a['primary_model']
        ps, pt = self.pair(a['primary_target'], a['primary_feature_set'],
                           max_rows=a['sensitivity_max_rows'])
        rows = []
        for imp in ['median', 'knn', 'mice']:
            log.info('imputation: %s', imp)
            for scen, (prep, p, cl) in {
                    'mimic_internal': (ps, self.oof(model, ps, imputer=imp), ps.groups),
                    'mimic_to_eicu': (pt, self.external(model, ps, pt, imputer=imp),
                                      pt.sites)}.items():
                ci = self._ci(prep.y, p, prep.task, cl, scenario=scen, imputer=imp,
                              model=model)
                ci['n'] = len(prep.y)
                rows.append(ci)
        self._save(pd.concat(rows), 'imputation', 'imputation_sensitivity')

    # -------------------------------------------------------------------- all
    def run(self):
        ex = self.cfg['experiments']
        steps = ['cohort', 'shift', 'models_transfer', 'ladder', 'decomposition',
                 'hospitals', 'ablation', 'recalibration', 'domain_adaptation',
                 'modality_agreement', 'imputation']
        timings = {}
        for s in steps:
            if not ex.get(s, False):
                continue
            log.info('=== %s ===', s)
            t = time.time()
            getattr(self, s)()
            timings[s] = round(time.time() - t, 1)
        tpath = self.out / 'timings.json'   # merged, so partial reruns keep earlier stages
        merged = json.loads(tpath.read_text()) if tpath.exists() else {}
        merged.update(timings)
        tpath.write_text(json.dumps(merged, indent=2))
        return timings
