"""Command-line entry point.

  python -m revision.run_revision sql       --config configs/revision.yaml   # print SQL
  python -m revision.run_revision extract   --config configs/revision.yaml   # BigQuery -> cache
  python -m revision.run_revision analyze   --config configs/revision.yaml   # cache -> results
  python -m revision.run_revision report    --config configs/revision.yaml   # figures/tables
  python -m revision.run_revision synthetic --config configs/revision_smoke.yaml  # offline test
"""

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import extract, reporting
from .config import load_config


def _setup_logging(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    log_file = Path(out) / f'run_{datetime.now():%Y%m%d_%H%M%S}.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    for noisy in ('fontTools', 'matplotlib'):   # PDF font-subsetting chatter
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def _manifest(cfg, command, data_source):
    import numpy, pandas, sklearn, xgboost  # noqa: E401
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                cwd=repo).stdout.strip()
        # only modified tracked code/config files count (logs, outputs, archives do not)
        changed = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no',
                                  '--', 'revision', 'configs'], capture_output=True,
                                 text=True, cwd=repo).stdout.split('\n')
        changed = [c.strip() for c in changed if c.strip()]
    except OSError:
        commit, changed = None, None
    now = datetime.now()
    m = {'command': command, 'data_source': data_source, 'time': now.isoformat(),
         'experiments_run': [k for k, v in cfg['experiments'].items() if v],
         'git_commit': commit, 'git_dirty': bool(changed), 'git_changed_files': changed,
         'cache_key': extract.cache_key(cfg), 'python': platform.python_version(),
         'versions': {'numpy': numpy.__version__, 'pandas': pandas.__version__,
                      'scikit-learn': sklearn.__version__, 'xgboost': xgboost.__version__},
         'config': cfg}
    text = json.dumps(m, indent=2, default=str)
    out = Path(cfg['output_dir'])
    (out / f'run_manifest_{now:%Y%m%d_%H%M%S}.json').write_text(text)   # never overwritten
    (out / 'run_manifest.json').write_text(text)                        # latest run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['sql', 'extract', 'analyze', 'report', 'synthetic',
                                        'all'])
    ap.add_argument('--config', default='configs/revision.yaml')
    ap.add_argument('--force', action='store_true', help='re-extract even if cached')
    ap.add_argument('--dataset', choices=['mimic', 'eicu'], default='mimic',
                    help='for the sql command')
    args = ap.parse_args(argv)

    import sklearn
    if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) < (1, 2):
        sys.exit(f'scikit-learn >= 1.2 required (found {sklearn.__version__}): '
                 f'pip install -U "scikit-learn>=1.2"')

    cfg = load_config(args.config)
    if args.command == 'sql':
        print(extract.print_sql(cfg, args.dataset))
        return
    log_file = _setup_logging(cfg['output_dir'])
    log = logging.getLogger('revision')
    log.info('log: %s', log_file)

    if args.command == 'synthetic':
        from .synthetic import write_synthetic_cache
        paths = write_synthetic_cache(cfg)
        log.warning('Wrote SYNTHETIC data to %s -- for testing only', paths)
    if args.command in ('extract', 'all'):
        extract.extract(cfg, force=args.force)
    if args.command in ('analyze', 'all', 'synthetic'):
        from .experiments import Runner
        raw, flows = {}, {}
        for ds in extract.DATASETS:
            raw[ds], flows[ds] = extract.load(cfg, ds)
        synthetic = any(extract.is_synthetic(cfg, ds) for ds in extract.DATASETS)
        stamp = Path(cfg['output_dir']) / 'SYNTHETIC_DATA_DO_NOT_USE.txt'
        if synthetic:
            stamp.write_text('These outputs were produced from SYNTHETIC test data.\n')
            log.warning('Analysing SYNTHETIC data -- outputs are for testing only')
        elif stamp.exists():
            stamp.unlink()
        _manifest(cfg, ' '.join(sys.argv), 'synthetic' if synthetic else 'bigquery')
        timings = Runner(cfg, raw, flows).run()
        log.info('timings (s): %s', timings)
    if args.command in ('report', 'analyze', 'all', 'synthetic') and \
            cfg['experiments'].get('figures', True):
        reporting.make_all(cfg['output_dir'], cfg['analysis']['primary_target'],
                           cfg['analysis']['primary_model'])
        log.info('figures and tables written to %s', Path(cfg['output_dir']) / 'figures')


if __name__ == '__main__':
    main()
