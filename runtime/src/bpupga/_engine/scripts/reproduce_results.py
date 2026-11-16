"""Rebuild all numerical results from archived observations or fresh run.py outputs.

This script never fits a model. Use the configurations in experiments/runs.csv
with run.py to generate a fresh fit tree before selecting --fits-root.
"""
import os
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import subprocess
import sys

ROOT = Path(os.environ.get("BIPGA_ARTIFACT_ROOT", Path(__file__).resolve().parents[1]))
STANDARD = ['figure_05', 'figure_06', 'figure_07', 'figure_08', 'table_iv']
QUALITY = ['pair_f1', 'node_f1', 'hits1', 'mrr']
COST = ['time_s', 'ram_gib', 'gpu_gib']


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measurements(record):
    values = dict(pair_f1=record['pair']['f1'], node_f1=record['node_f1'],
                  hits1=record['ranking']['hits@1'], mrr=record['ranking']['mrr'],
                  **{key: record['runtime'][key] for key in COST})
    for key, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid measurement: ' + key)
        if key in QUALITY and value > 1:
            raise ValueError('Quality metrics must be fractions: ' + key)
    return values


def compare_summary(expected, actual):
    a, b = read(expected), read(actual)
    keys = [key for key in a[0] if key not in {'n', 'mean', 'sample_sd'}]
    aa = {tuple(row[k] for k in keys): row for row in a}
    bb = {tuple(row[k] for k in keys): row for row in b}
    if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys():
        raise ValueError('Summary groups differ: ' + str(expected))
    for key, row in aa.items():
        for field, value in row.items():
            other = bb[key][field]
            if value == other:
                continue
            if field not in {'n', 'mean', 'sample_sd'} or not value or not other:
                raise ValueError(f'Summary mismatch: {expected}, {key}, {field}')
            if not math.isclose(float(value), float(other), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Summary mismatch: {expected}, {key}, {field}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New results directory')
    parser.add_argument('--fits-root', type=Path, help='Fresh run.py outputs indexed by run_id')
    parser.add_argument('--device', default='cuda:0', help='CUDA assignment device for fresh diagnostic scores')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    templates = {name: read(ROOT / 'data' / name / 'values_by_run.csv') for name in STANDARD}
    fresh = {}
    if args.fits_root is not None:
        required = {row['run_id'] for rows in templates.values() for row in rows}
        required.update(row['run_id'] for row in read(ROOT / 'data/figure_09/runs.csv'))
        missing = [run_id for run_id in sorted(required)
                   if not (args.fits_root / run_id / 'metrics.json').is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} required fits are missing; first IDs: {missing[:5]}')
        for run_id in sorted(required):
            fresh[run_id] = measurements(json.loads((args.fits_root / run_id / 'metrics.json').read_text()))
        for row in read(ROOT / 'data/figure_09/runs.csv'):
            fit = args.fits_root / row['run_id'] / 'fit'
            if not (fit / 'fit_seal.json').is_file():
                raise FileNotFoundError(fit / 'fit_seal.json')
    args.output.mkdir(parents=True)
    data = args.output / 'data'
    for name, rows in templates.items():
        if args.fits_root is not None:
            for row in rows:
                values = fresh[row['run_id']]
                for key in QUALITY + COST:
                    if key in row:
                        row[key] = values[key]
        target = data / name
        write(target / 'values_by_run.csv', rows)
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/summarize.py'),
                        str(target / 'values_by_run.csv'), '--output', str(target / 'values.csv')], check=True)
    diagnostics = args.output / 'diagnostics'
    command = [sys.executable, '-B', str(ROOT / 'scripts/reproduce_diagnostics.py'),
               '--output', str(diagnostics)]
    if args.fits_root is not None:
        command += ['--fits-root', str(args.fits_root), '--device', args.device]
    subprocess.run(command, check=True)
    for name in ['figure_09', 'table_v', 'table_vi']:
        target = data / name
        target.mkdir(parents=True)
        shutil.copyfile(diagnostics / (name + '_by_run.csv'), target / 'values_by_run.csv')
        shutil.copyfile(diagnostics / (name + '_summary.csv'), target / 'values.csv')
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/reproduce_source_statistics.py'),
                    '--output', str(data / 'table_ii/values.csv')], check=True)
    if args.fits_root is None:
        for path in data.glob('*/values.csv'):
            compare_summary(ROOT / 'data' / path.parent.name / 'values.csv', path)
        print('All numerical summaries agree with the supplied results.')
    print('Results: ' + str(data.resolve()) + '; no model fitting was started.')


if __name__ == '__main__':
    main()
