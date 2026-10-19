import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from elitereplay.evaluation import diversity


def samples():
    return [
        dict(model=model, question_id='task', sample_id=i,
             correct=label is not None, strategy=label)
        for model, labels in {'base': ['a', 'a', 'b', None],
                              'trained': ['a', 'a', 'a', None]}.items()
        for i, label in enumerate(labels)
    ]


def test_labels_compute_metrics_without_sql_parsing(monkeypatch):
    def no_sql(*args):
        raise AssertionError('Precomputed strategies must not invoke SQL descriptors')
    monkeypatch.setattr(diversity, 'descriptor_key', no_sql)
    monkeypatch.setattr(diversity, 'pair_similarity', no_sql)
    result = diversity.evaluate(samples(), [1, 2, 4], 'base', descriptor='precomputed', bootstrap=10)
    base, trained = result['models']['base'], result['models']['trained']
    assert base['u@1'] == .75
    assert base['u@2'] == pytest.approx(4 / 3)
    assert base['u@4'] == 2
    assert base['pass@1'] == .75 and base['pass@4'] == 1
    assert base['effective_diversity'] == pytest.approx(1 / 3)
    assert trained['effective_diversity'] == 0
    assert trained['retention'] == .5
    assert 'ci95' in trained
    assert result['effective_diversity_descriptor'] == 'precomputed'


@pytest.mark.parametrize('label', [None, '', '  ', 123, ['a']])
def test_correct_outputs_require_strategy_labels(label):
    rows = samples()
    rows[0]['strategy'] = label
    with pytest.raises(ValueError, match='nonempty strategy string'):
        diversity.evaluate(rows, [1], 'base', descriptor='precomputed')


def test_failed_outputs_remain_in_denominators_without_labels():
    rows = samples()
    for row in rows:
        if not row['correct']:
            del row['strategy']
    result = diversity.evaluate(rows, [1], 'base', descriptor='precomputed')
    assert result['models']['base']['pass@1'] == .75
    assert result['models']['base']['u@1'] == .75


def test_cli_defaults_to_categorical_effective_diversity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'metrics.json'
    subprocess.run([
        sys.executable, '-m', 'elitereplay', 'metrics',
        str(root / 'examples/scored_samples.jsonl'), '--descriptor', 'precomputed',
        '--base-model', 'base', '--k', '1', '2', '4',
        '--bootstrap', '0', '--output', str(output),
    ], check=True, env=dict(os.environ, PYTHONPATH=str(root / 'src')))
    result = json.loads(output.read_text())
    assert result['strategy_descriptor'] == 'precomputed'
    assert result['effective_diversity_descriptor'] == 'precomputed'
    assert result['models']['elitereplay']['retention'] == 1
    assert 'sqlglot_version' not in result
