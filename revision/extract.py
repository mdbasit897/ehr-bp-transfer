"""Run the extraction queries on BigQuery and cache the results as parquet.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

from . import schema as S
from . import sql

log = logging.getLogger(__name__)

DATASETS = ['mimic', 'eicu']
SYNTHETIC_MARKER = '-- SYNTHETIC DATA'


def query_parameters(cfg):
    ex = cfg['extraction']
    return {k: int(ex[k]) for k in ['obs_hours', 'horizon_start_hours', 'horizon_end_hours',
                                    'lab_lookback_hours', 'hypotension_map', 'min_age',
                                    'max_age_exclusive', 'sample_permille']}


def cache_key(cfg):
    payload = json.dumps(query_parameters(cfg), sort_keys=True)
    payload += sql.feature_query('mimic') + sql.feature_query('eicu')
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def cache_paths(cfg, dataset):
    base = Path(cfg['cache_dir']) / f'{dataset}_{cache_key(cfg)}'
    return {'data': base.with_suffix('.parquet'),
            'flow': Path(str(base) + '_flow.json'),
            'sql': Path(str(base) + '.sql')}


def _run(client, query, params):
    from google.cloud import bigquery
    used = {k: v for k, v in params.items() if re.search(rf'@{k}\b', query)}
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter(k, 'INT64', v) for k, v in used.items()])
    return client.query(query, job_config=job_config).to_dataframe()


def extract(cfg, datasets=DATASETS, force=False):
    from google.cloud import bigquery
    project = cfg.get('project_id') or os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not project:
        raise ValueError('Set project_id in the config or GOOGLE_CLOUD_PROJECT')
    client = bigquery.Client(project=project)
    params = query_parameters(cfg)

    for ds in datasets:
        paths = cache_paths(cfg, ds)
        if paths['data'].exists() and not force:
            if is_synthetic(cfg, ds):
                raise RuntimeError(f'{paths["data"]} holds SYNTHETIC data; delete it or use '
                                   f'--force to replace it with real data')
            log.info('%s: cache exists (%s), skipping', ds, paths['data'])
            continue
        q = sql.feature_query(ds)
        paths['sql'].write_text(f'-- parameters: {json.dumps(params)}\n{q}\n')
        log.info('%s: running flow query', ds)
        flow = _run(client, sql.flow_query(ds), params).iloc[0].to_dict()
        flow = {k: int(v) for k, v in flow.items()}
        paths['flow'].write_text(json.dumps(flow, indent=2))
        log.info('%s: flow %s', ds, flow)
        log.info('%s: running feature query', ds)
        df = _run(client, q, params)
        missing = set(S.EXTRACT_COLUMNS) - set(df.columns)
        if missing:
            raise RuntimeError(f'{ds}: extraction is missing columns {sorted(missing)}')
        df.to_parquet(paths['data'], index=False)
        log.info('%s: %d rows -> %s', ds, len(df), paths['data'])


def is_synthetic(cfg, dataset):
    p = cache_paths(cfg, dataset)['sql']
    return p.exists() and p.read_text().startswith(SYNTHETIC_MARKER)


def load(cfg, dataset):
    paths = cache_paths(cfg, dataset)
    if not paths['data'].exists():
        raise FileNotFoundError(f'{paths["data"]} not found. Run the "extract" step first '
                                f'(or "synthetic" for an offline test).')
    if is_synthetic(cfg, dataset) and not cfg.get('allow_synthetic'):
        raise RuntimeError(f'{paths["data"]} is SYNTHETIC test data but this config does not '
                           f'set allow_synthetic: true. Run "extract" for real data.')
    df = pd.read_parquet(paths['data'])
    flow = json.loads(paths['flow'].read_text()) if paths['flow'].exists() else {}
    return df, flow


def print_sql(cfg, dataset):
    params = query_parameters(cfg)
    return f'-- parameters: {json.dumps(params)}\n{sql.feature_query(dataset)}'
