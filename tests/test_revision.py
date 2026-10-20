"""Fast checks of the pipeline invariants (docs/PIPELINE_NOTES.md). Run: python -m pytest tests -q"""

import numpy as np
import pandas as pd
import pytest

from revision import evaluation as E
from revision import harmonize as H
from revision import schema as S
from revision import sql
from revision.config import load_config
from revision.synthetic import _dataset


@pytest.fixture(scope='module')
def cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp('rev')
    return load_config(None, {'cache_dir': str(d / 'cache'), 'output_dir': str(d / 'out')})


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(1)
    return {'mimic': H.clean(_dataset(800, rng, 'mimic'), 'mimic'),
            'eicu': H.clean(_dataset(900, rng, 'eicu', 8), 'eicu')}


def test_sql_mentions_cuff_table_and_correct_itemids():
    q = sql.feature_query('eicu')
    assert 'vitalaperiodic' in q and 'noninvasivesystolic' in q
    m = sql.feature_query('mimic')
    assert '220179' in m and '455' in m          # NIBP systolic (MetaVision, CareVue)
    assert 'hadm_id = c.hadm_id' not in m        # labs restricted by time, not admission
    for ds in ('mimic', 'eicu'):
        q = sql.feature_query(ds)
        assert 'LIMIT 1)' in q and q.count('LIMIT') == q.count('LIMIT 1)')  # no cohort LIMIT
        assert 'stay_seq = 1' in q and 'ORDER BY c.stay_id' in q


def test_outcome_queries_do_not_change_feature_cache(cfg):
    from revision import extract
    key = extract.cache_key(cfg)
    assert 'hospital_expire_flag' in sql.outcome_query('mimic')
    assert "'Expired'" in sql.outcome_query('eicu')
    q = sql.severity_query(cfg['outcomes']['eicu'])
    assert "apacheversion = 'IVa'" in q and 'GROUP BY stay_id' in q
    assert extract.cache_key(cfg) == key


def test_gender_mapping_both_databases():
    g = H.map_gender(pd.Series(['M', 'F', 'Male', 'Female', 'Unknown', None]))
    assert g.tolist()[:4] == [0, 1, 0, 1] and g.isna().sum() == 2


def test_concurrent_design_refuses_bp_history(cfg, data):
    with pytest.raises(ValueError):
        H.prepare(data['mimic'], cfg, 'sbp', 'full', design='concurrent')


def test_hypotension_requires_prospective(cfg, data):
    with pytest.raises(ValueError):
        H.prepare(data['mimic'], cfg, 'hypotension', 'no_bp_core', design='concurrent')


def test_feature_sets_share_the_same_patients(cfg, data):
    ids = {fs: tuple(H.prepare(data['eicu'], cfg, 'sbp', fs).meta['stay_id'])
           for fs in S.FEATURE_SETS}
    assert len(set(ids.values())) == 1


def test_grouped_folds_do_not_split_groups():
    groups = np.repeat(np.arange(50), 3)
    for tr, te in E.grouped_folds(groups, 5, 0):
        assert not set(groups[tr]) & set(groups[te])
    all_te = np.concatenate([te for _, te in E.grouped_folds(groups, 5, 0)])
    assert sorted(all_te) == list(range(len(groups)))


def test_bp_history_matches_target_modality(cfg, data):
    p = H.prepare(data['mimic'], cfg, 'sbp', 'full', modality='art')
    raw = data['mimic'].set_index('stay_id').loc[p.meta['stay_id'], 'hist_sbp_art_mean']
    np.testing.assert_allclose(p.X['hist_sbp_mean'].to_numpy(), raw.to_numpy())


def test_row_cap_is_deterministic(cfg, data):
    a = H.prepare(data['eicu'], cfg, 'sbp', 'no_bp_core', max_rows=300).meta['stay_id']
    b = H.prepare(data['eicu'], cfg, 'sbp', 'no_bp_core', max_rows=300).meta['stay_id']
    assert len(a) == 300 and a.equals(b)


def test_regression_metrics_known_values():
    y = np.array([100., 110., 120., 130.])
    m = E.regression_metrics(y, y + 5)
    assert m['bias'] == pytest.approx(5) and m['RMSE'] == pytest.approx(5)
    assert m['cal_slope'] == pytest.approx(1)
    assert E.regression_metrics(y, np.full(4, y.mean()))['R2'] == pytest.approx(0)


def test_cluster_bootstrap_resamples_whole_clusters():
    s = E.ClusterSampler(np.repeat(['a', 'b', 'c'], [1, 2, 3]))
    rng = np.random.default_rng(0)
    for _ in range(20):
        counts = np.bincount(s.draw(rng), minlength=6)
        assert counts[1] == counts[2] and counts[3] == counts[4] == counts[5]


def test_benjamini_hochberg_matches_reference():
    from revision.shift import benjamini_hochberg
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5, np.nan])
    # reference values from R: p.adjust(c(.01,.04,.03,.5), "BH")
    np.testing.assert_allclose(q[:4], [0.04, 0.0533333, 0.0533333, 0.5], rtol=1e-5)
    assert np.isnan(q[4])


def test_linear_domain_model_ignores_value_resolution():
    """Same distribution, different recording resolution: trees can tell the
    databases apart, a linear model cannot."""
    from revision.shift import domain_propensity
    rng = np.random.default_rng(0)
    a = pd.DataFrame({'x': np.round(rng.normal(37, 0.6, 3000), 1)})          # charted
    b = pd.DataFrame({'x': (np.round(rng.normal(98.6, 1.1, 3000)) - 32) * 5 / 9})  # F->C
    auc_tree = domain_propensity(a, b, 0, kind='tree')[2]
    auc_lin = domain_propensity(a, b, 0, kind='linear')[2]
    assert auc_lin < 0.6 < auc_tree
