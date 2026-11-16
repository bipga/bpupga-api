"""Recompute Figure 9 and Tables V/VI from observations or saved fits.

No model is fitted by this script. Reference labels are used only after fitting.
"""
import os
from pathlib import Path
import argparse
import csv
import json
import statistics
from collections import defaultdict
import numpy as np
ROOT = Path(os.environ.get("BIPGA_ARTIFACT_ROOT", Path(__file__).resolve().parents[1]))
SETTINGS = ['F', 'Us', 'Ut', 'B']

def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def summarize(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple((row[k] for k in keys))].append(row)
    result = []
    for key, values in sorted(groups.items()):
        for metric in metrics:
            x = [float(r[metric]) for r in values if r[metric] is not None]
            result.append(dict(zip(keys, key), metric=metric, n=len(x), mean=statistics.mean(x) if x else '', sample_sd=statistics.stdev(x) if len(x) > 1 else ''))
    return result

def analyze_saved_fits(sources, candidate, orders, device='cuda:0'):
    """sources[token]=(graph directory, sealed fit directory)."""
    source = np.asarray(candidate['source_ids'], dtype=np.int64)
    target = np.asarray(candidate['target_ids'], dtype=np.int64)
    pairs = [tuple(p) for p in candidate['reference_pairs']]
    assert list(source) == sorted((u for u, v in pairs))
    assert list(target) == sorted((v for u, v in pairs))
    target_index = {int(v): i for i, v in enumerate(target)}
    true_col = np.array([target_index[v] for u, v in pairs])
    k = len(source)
    result = dict(source_ids=source, target_ids=target, reference_target_ids=np.array([v for u, v in pairs], dtype=np.int64))
    names = ['native_correct', 'hits_source_correct', 'hits_target_correct', 'unique_top1_source', 'unique_top1_target', 'wrong_above_source', 'wrong_above_target']
    result.update({name: np.zeros((4, k), dtype=bool) for name in names})
    result['predicted_target_ids'] = np.empty((4, k), dtype=np.int64)
    result['assignment_objective'] = np.empty(4, dtype=np.float64)
    result['source_order_ids'] = orders['source_order_ids'].copy()
    result['target_order_ids'] = orders['target_order_ids'].copy()
    for index, token in enumerate(SETTINGS):
        graph, fit = map(Path, sources[token])
        ids = [sorted(map(int, (graph / f'G{side}.nodes').read_text().split())) for side in (1, 2)]
        inverse = [{u: i for i, u in enumerate(side)} for side in ids]
        native = json.loads((fit / 'native_pairs_local.json').read_text())['pairs']
        native = {(ids[0][u], ids[1][v]) for u, v in native}
        result['native_correct'][index] = [p in native for p in pairs]
        ii = np.array([inverse[0][int(u)] for u in source])
        jj = np.array([inverse[1][int(v)] for v in target])
        meta = json.loads((fit / 'fit_metadata.json').read_text())
        name = 'score_matrix.npy' if (fit / 'score_matrix.npy').exists() else 'cost_matrix.npy'
        matrix = np.load(fit / name, mmap_mode='r', allow_pickle=False)
        assert matrix.shape == (len(ids[0]), len(ids[1]))
        sign = -1 if name == 'cost_matrix.npy' or not meta['larger_is_better'] else 1
        gt = np.asarray(matrix[ii, jj[true_col]], dtype=np.float64) * sign
        gt_by_target = np.empty(k)
        gt_by_target[true_col] = gt
        col_max = np.full(k, -np.inf)
        col_arg = np.zeros(k, dtype=np.int64)
        col_ties = np.zeros(k, dtype=np.int64)
        for start in range(0, k, 256):
            q = np.asarray(matrix[ii[start:start + 256], :][:, jj], dtype=np.float64) * sign
            if not np.isfinite(q).all():
                raise ValueError('Nonfinite saved scores')
            count = len(q)
            row_max = q.max(axis=1)
            truth = gt[start:start + count]
            result['hits_source_correct'][index, start:start + count] = q.argmax(axis=1) == true_col[start:start + count]
            result['wrong_above_source'][index, start:start + count] = row_max > truth
            result['unique_top1_source'][index, start:start + count] = (row_max == truth) & ((q == truth[:, None]).sum(axis=1) == 1)
            best = q.argmax(axis=0)
            scores = q[best, np.arange(k)]
            better = scores > col_max
            col_arg[better] = start + best[better]
            col_max[better] = scores[better]
            col_ties += (q == gt_by_target[None, :]).sum(axis=0)
        result['hits_target_correct'][index] = col_arg[true_col] == np.arange(k)
        result['wrong_above_target'][index] = col_max[true_col] > gt
        result['unique_top1_target'][index] = (col_max[true_col] == gt) & (col_ties[true_col] == 1)
        q = np.asarray(matrix[np.ix_(ii, jj)], dtype=np.float64) * sign
        import sys
        sys.path.insert(0, str(ROOT))
        from gpu_backend.assignment import linear_sum_assignment as original_cuda_lap
        source_index = {int(u): i for i, u in enumerate(source)}
        target_index = {int(v): i for i, v in enumerate(target)}
        aa = np.array([source_index[int(u)] for u in orders['source_order_ids'][index]])
        bb = np.array([target_index[int(v)] for v in orders['target_order_ids'][index]])
        assert set(aa) == set(bb) == set(range(k))
        ordered = q[np.ix_(aa, bb)]
        result['hits_source_correct'][index, aa] = bb[ordered.argmax(axis=1)] == true_col[aa]
        truth_by_target = np.empty(k, dtype=np.int64)
        truth_by_target[true_col] = np.arange(k)
        target_correct = np.empty(k, dtype=bool)
        target_correct[bb] = aa[ordered.argmax(axis=0)] == truth_by_target[bb]
        result['hits_target_correct'][index] = target_correct[true_col]
        rr, cc = original_cuda_lap(ordered, maximize=True, device=device)
        rr, cc = (aa[rr], bb[cc])
        del ordered
        predicted = np.empty(k, dtype=np.int64)
        predicted[rr] = target[cc]
        del q
        assert set(map(int, predicted)) == set(map(int, target)) and len(set(map(int, predicted))) == k
        result['predicted_target_ids'][index] = predicted
        columns = np.array([inverse[1][int(v)] for v in predicted])
        result['assignment_objective'][index] = np.asarray(matrix[ii, columns], dtype=np.float64).sum() * sign
        del matrix
    return result

def derive(dataset, method, seed, data, orbits):
    k = len(data['source_ids'])
    fig = []
    for i, setting in enumerate(SETTINGS):
        source_correct = int(data['hits_source_correct'][i].sum())
        target_correct = int(data['hits_target_correct'][i].sum())
        correct = int((data['predicted_target_ids'][i] == data['reference_target_ids']).sum())
        fig.append(dict(dataset=dataset, method=method, seed=seed, setting=setting, core_pairs=k, correct_source_queries=source_correct, correct_target_queries=target_correct, hits1=(source_correct + target_correct) / (2 * k), correct_pairs=correct, pair_recall=correct / k, output_pairs=k, assignment_objective=float(data['assignment_objective'][i])))
    recovered = np.logical_and.reduce(data['native_correct'][:3], axis=0)
    lost = recovered & ~data['native_correct'][3]
    recovered_n = int(recovered.sum())
    lost_n = int(lost.sum())
    flipped_source = np.logical_and.reduce(data['unique_top1_source'][:3], axis=0) & data['wrong_above_source'][3]
    flipped_target = np.logical_and.reduce(data['unique_top1_target'][:3], axis=0) & data['wrong_above_target'][3]
    flip_n = int(flipped_source[lost].sum() + flipped_target[lost].sum())
    wrong_n = int(data['wrong_above_source'][3, lost].sum() + data['wrong_above_target'][3, lost].sum())
    losses = dict(dataset=dataset, method=method, seed=seed, core_pairs=k, recovered_pairs=recovered_n, lost_pairs=lost_n, lost_percent=100 * lost_n / recovered_n if recovered_n else None, lost_directional_queries=2 * lost_n, top1_reversal_queries=flip_n, top1_reversal_percent=100 * flip_n / (2 * lost_n) if lost_n else None, wrong_ranking_queries=wrong_n, wrong_ranking_percent=100 * wrong_n / (2 * lost_n) if lost_n else None)
    lookup = [dict(zip(orbits[f'G{s}_nodes'].tolist(), orbits[f'G{s}_orbit_sizes'].tolist())) for s in (1, 2)]
    ordinary = np.array([lookup[0][int(u)] == 1 and lookup[1][int(v)] == 1 for u, v in zip(data['source_ids'], data['reference_target_ids'])])
    symmetry = []
    for name, mask in [('all_shared', np.ones(k, dtype=bool)), (method + '_losses', lost)]:
        n = int(mask.sum())
        single = int(ordinary[mask].sum())
        symmetry.append(dict(dataset=dataset, method=method if name != 'all_shared' else '', seed=seed, pair_set=name, pairs=n, no_symmetric_endpoint_pairs=single, no_symmetric_endpoint_percent=100 * single / n if n else None))
    return (fig, losses, symmetry)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fits-root', type=Path, help='Read fresh sealed run.py outputs, one subdirectory per run_id; otherwise use packaged compact observations')
    parser.add_argument('--device', default='cuda:0', help='Original CUDA decoder device when reading fresh saved fits')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True)
    index = read_csv(ROOT / 'data/figure_09/inputs.csv')
    inputs = read_csv(ROOT / 'data/figure_09/runs.csv')
    graphs = {r['graph_id']: ROOT / r['path'] for r in read_csv(ROOT / 'datasets/index.csv')}
    figs = []
    losses = []
    symmetry = []
    all_sets = set()
    for row in index:
        dataset, method, seed = (row['dataset'], row['method'], int(row['seed']))
        if args.fits_root:
            settings = [r for r in inputs if (r['dataset'], r['method'], int(r['seed'])) == (dataset, method, seed)]
            sources = {r['setting']: (graphs[r['graph_id']], args.fits_root / r['run_id'] / 'fit') for r in settings}
            candidate = json.loads((ROOT / row['candidates']).read_text())
            with np.load(ROOT / row['observations'], allow_pickle=False) as stored:
                orders = {k: stored[k].copy() for k in ['source_order_ids', 'target_order_ids']}
            data = analyze_saved_fits(sources, candidate, orders=orders, device=args.device)
        else:
            with np.load(ROOT / row['observations'], allow_pickle=False) as stored:
                data = {name: stored[name] for name in stored.files}
        with np.load(ROOT / row['orbits'], allow_pickle=False) as orbits:
            fig, loss, sym = derive(dataset, method, seed, data, orbits)
        figs.extend(fig)
        losses.append(loss)
        for value in sym:
            key = (dataset, seed, value['pair_set'])
            if key not in all_sets:
                symmetry.append(value)
                all_sets.add(key)
    save_csv(args.output / 'figure_09_by_run.csv', figs)
    save_csv(args.output / 'figure_09_summary.csv', summarize(figs, ['dataset', 'method', 'setting'], ['hits1', 'pair_recall']))
    save_csv(args.output / 'table_v_by_run.csv', losses)
    save_csv(args.output / 'table_v_summary.csv', summarize(losses, ['dataset', 'method'], ['recovered_pairs', 'lost_pairs', 'lost_percent', 'top1_reversal_percent']))
    save_csv(args.output / 'table_vi_by_run.csv', symmetry)
    save_csv(args.output / 'table_vi_summary.csv', summarize(symmetry, ['dataset', 'pair_set'], ['pairs', 'no_symmetric_endpoint_percent']))
    print(f'Recomputed Figure9 ({len(figs)} settings), TableV ({len(losses)} runs), TableVI ({len(symmetry)} pair-set runs); no fitting')
if __name__ == '__main__':
    main()
