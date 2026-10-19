"""Independent model reload, metric recomputation, and top-K tie diagnostics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from gp_full_experiment import (evaluate, json_write, load_local_model, load_parts,
                                score_raw_shipments)


def tie_diagnostics(y, scores):
    rows = []
    n = len(y)
    total = float(np.sum(y))
    for fraction in [.01, .05, .1, .2]:
        k = max(1, int(round(fraction * n)))
        boundary = np.sort(scores)[-k]
        above = scores > boundary
        tied = scores == boundary
        remaining = k - int(above.sum())
        tie_positive = int(y[tied].sum())
        tie_count = int(tied.sum())
        hits_above = int(y[above].sum())
        expected = hits_above + remaining * tie_positive / tie_count
        minimum = hits_above + max(0, remaining - (tie_count - tie_positive))
        maximum = hits_above + min(remaining, tie_positive)
        rows.append({'fraction': fraction, 'k': k, 'boundary_tied_rows': tie_count,
                     'boundary_slots': remaining, 'expected_precision_random_ties': expected / k,
                     'expected_recall_random_ties': expected / total,
                     'recall_min_tie_order': minimum / total,
                     'recall_max_tie_order': maximum / total})
    return rows


def validate(root):
    directory = root / 'results' / 'gp_full_v1'
    seal = json.loads((directory / 'selection_sealed_before_test.json').read_text(encoding='utf-8'))
    train, val, test, _ = load_parts(root / 'dataset.parquet')
    rng = np.random.default_rng(2026)
    altered = test.copy()
    altered['is_lost'] = rng.permutation(altered['is_lost'].to_numpy())
    # Deliberately corrupt forbidden leakage fields; the persisted encoder must ignore them.
    altered['DATEPARAMETER'] = '2099-12-31'
    altered['CAUSA_BPP'] = altered['is_lost']
    diagnostics, checks = [], []
    for name in seal['evaluation_runs']:
        result = json.loads((directory / (name + '.json')).read_text(encoding='utf-8'))
        model = load_local_model(directory / result['model_path'])
        scores = score_raw_shipments(model, test.drop(columns='is_lost'))
        altered_scores = score_raw_shipments(model, altered)
        np.testing.assert_array_equal(scores, altered_scores)
        # Raw inference path must be independent of both label and forbidden attributes.
        for partition, frame in [('validation', val), ('test', test)]:
            values = score_raw_shipments(model, frame)
            measured = evaluate(frame['is_lost'].to_numpy(int), values, model['threshold'])
            for metric, value in measured.items():
                np.testing.assert_allclose(value, result[partition][metric], rtol=0, atol=1e-12)
        if not result['include_domain']:
            assert not any('DOM_DOMAIN_ID' in name for name in model['encoder'].names)
        assert result['fitness_rows'] == len(train) == 125082
        assert model['threshold'] == seal['thresholds'][name]
        assert (directory / 'selection_sealed_before_test.json').stat().st_mtime <= (directory / (name + '.json')).stat().st_mtime
        for row in tie_diagnostics(test['is_lost'].to_numpy(int), scores):
            diagnostics.append({'run': name, 'include_domain': result['include_domain'], **row})
        checks.append({'run': name, 'metrics_reproduced': True,
                       'labels_and_forbidden_columns_do_not_change_scores': True,
                       'fitness_rows': result['fitness_rows'],
                       'score_sha256': hashlib.sha256(scores.tobytes()).hexdigest()})
    pd.DataFrame(diagnostics).to_csv(directory / 'ranking_tie_audit.csv', index=False)
    purity = {}
    for name, frame in [('train', train), ('validation', val), ('test', test)]:
        stats = frame.groupby('DOM_DOMAIN_ID')['is_lost'].agg(['count', 'mean'])
        pure = stats[(stats['count'] >= 200) & (stats['mean'] == 1.0)]
        purity[name] = {'large_100pct_loss_categories': len(pure),
                        'rows_in_these_categories': int(pure['count'].sum()),
                        'row_share': float(pure['count'].sum() / len(frame))}
    json_write(directory / 'domain_purity_audit.json', purity)
    json_write(directory / 'independent_validation.json', {'checked_runs': checks, 'errors': 0})
    print({'validated_models': len(checks), 'errors': 0, 'tie_audit_rows': len(diagnostics)})


if __name__ == '__main__':
    validate(Path(__file__).resolve().parent)
