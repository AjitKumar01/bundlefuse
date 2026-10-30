"""Experiment lifecycle tests: partial jobs must never replace canonical results."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.experiments import make_jobs, job_environment, read_result, promote_complete_results
from bundlefuse.artifacts import summarize_seeds, baseline_artifact, write_json


def test_default_plan_pairs_every_fullrank_split_and_retail_split():
    datasets = ('pog', 'pog_dense', 'spotify', 'spotify_sparse', 'electronic', 'food', 'clothing', 'dunnhumby')
    jobs = make_jobs(datasets, ('matrix', 'baselines', 'selection'), [0, 1, 2], 5)
    assert len(jobs) == 53
    for ds in datasets:
        for mode in ('matrix', 'baselines'):
            subset = [j for j in jobs if j['dataset'] == ds and j['mode'] == mode]
            assert [j['seed'] for j in subset] == ([0, 1, 2] if ds in datasets[:4] else [0])
            assert all(j['splits'] == (1 if ds in datasets[:4] else 5) for j in subset)


def test_job_environment_removes_ambient_experiment_modes(monkeypatch, tmp_path):
    monkeypatch.setenv('BASELINES', '1')
    monkeypatch.setenv('TRAIN_FRAC', '0.2')
    monkeypatch.setenv('VAL_N', '2')
    job = make_jobs(['pog'], ['matrix'], [2], 5)[0]
    env = job_environment(job, tmp_path)
    assert 'BASELINES' not in env and 'TRAIN_FRAC' not in env and 'VAL_N' not in env
    assert env['RAMEN_SEED'] == '2' and env['ENCODER'] == '0'
    assert env['SELDUMP'] == '1' and env['OMP_NUM_THREADS'] == '1'


def test_seed_summary_keeps_unrounded_means_and_dispersion():
    records = [{'seed': i, 'methods': {'Ours': {'R@20': value}}} for i, value in enumerate((.1, .3))]
    result = summarize_seeds(records)['Ours']
    assert result['R@20'] == pytest.approx(.2)
    assert result['R@20_std'] == pytest.approx(.1)
    assert result['seeds'] == 2
    with pytest.raises(ValueError, match='Inconsistent methods'):
        summarize_seeds(records + [{'seed': 2, 'methods': {}}])


def test_incomplete_job_cannot_publish(tmp_path):
    with pytest.raises(RuntimeError, match='Incomplete experiment'):
        promote_complete_results({'jobs': [{'status': 'failed'}]}, tmp_path, {})


def test_subset_job_does_not_publish_canonical_tables(tmp_path):
    # No writes outside the subset directory, even though every subset job completed.
    promote_complete_results({'datasets': ['electronic'], 'jobs': [{'status': 'complete'}]}, tmp_path, {})
    assert not list(tmp_path.iterdir())


def test_missing_required_result_fails_before_completion(tmp_path):
    job = {'mode': 'matrix', 'dataset': 'pog', 'objective': 'ndcg20', 'id': 'probe'}
    write_json(tmp_path / 'selcmp_pog_ndcg20.json', {'methods': {'Ours': {'R@20': .2}}})
    with pytest.raises(ValueError, match='missing methods'):
        read_result(job, tmp_path)


def test_baseline_artifact_preserves_selected_configs():
    record = {'seed': 2, 'methods': {'ItemKNN': {'hp': [50, 10], 'val_ndcg20': .2,
                                               'test': {20: (.3, .2)}}}}
    result = baseline_artifact('example', 100, [record])
    assert result['per_seed'][0]['methods']['ItemKNN']['hp'] == [50, 10]
    assert result['methods']['ItemKNN']['R@20'] == .3
