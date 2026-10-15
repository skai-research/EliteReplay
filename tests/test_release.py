import importlib.util
import itertools
import json
import hashlib
from pathlib import Path
import subprocess
import shutil

import pytest
from elitereplay.baselines.config import load, trinity_config
from elitereplay.evaluation.diversity import evaluate, no_hit
from elitereplay.analysis.sql_ast_modes import bound_alias_ast_key

ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_and_historical_counters_are_explicit():
    c = load(ROOT / 'configs/elitereplay.yaml')
    t = trinity_config(c)
    assert t['buffer']['total_steps'] == 100
    assert c['training']['prompts_per_iteration'] == 128
    assert c['training']['prompts_per_optimizer_update'] == 128
    assert t['algorithm']['policy_loss_fn_args']['sft_weight'] == .05
    assert t['algorithm']['policy_loss_fn_args']['sft_reduction'] == 'response_mean'
    historical = load(ROOT / 'configs/historical/elitereplay_schedule.yaml')
    assert historical['training']['rollout_iterations'] == 200
    assert historical['training']['prompts_per_optimizer_update'] == 64
    assert (ROOT / 'plugins/elitereplay_trinity.py').is_file()


@pytest.mark.parametrize('file', sorted((ROOT/'configs').glob('*.yaml')))
def test_all_training_configs_compile(file):
    trinity_config(load(file))


def test_released_overlay_hashes():
    lock = json.loads((ROOT/'experiments/baselines/runtime/runtime.lock.json').read_text())
    for record in lock['repositories']:
        assert (ROOT/record['overlay']/'LICENSE').is_file()
        for name, hashes in record['files'].items():
            assert hashlib.sha256((ROOT/record['overlay']/name).read_bytes()).hexdigest() == hashes['sha256']


def test_alias_binding_preserves_column_identity():
    assert bound_alias_ast_key('select x.id from t x') == bound_alias_ast_key('select y.id from t y')
    assert bound_alias_ast_key('select a.id from t a join t b on a.id=b.id') != bound_alias_ast_key('select b.id from t a join t b on a.id=b.id')


def test_hypergeometric_and_effective_diversity_against_enumeration():
    sql = ['SELECT a FROM t', 'SELECT a FROM t', 'SELECT b FROM t', '']
    rows = [dict(model='base', question_id='q', sample_id=i, sql=s, correct=bool(s)) for i,s in enumerate(sql)]
    result = evaluate(rows, [1,2,4], 'base', 'bound_alias_ast')['models']['base']
    labels = [bound_alias_ast_key(s) if s else None for s in sql]
    for k in [1,2,4]:
        subsets = list(itertools.combinations(labels,k))
        assert result[f'u@{k}'] == pytest.approx(sum(len(set(x)-{None}) for x in subsets)/len(subsets))
    pairs = list(itertools.combinations(labels,2))
    assert result['effective_diversity'] == pytest.approx(sum(a is not None and b is not None and a!=b for a,b in pairs)/len(pairs))


def test_invalid_metric_inputs_and_empty_retention():
    row = dict(model='base',question_id='q',sample_id=0,sql='',correct=False)
    rows = [row,dict(row,sample_id=1)]
    assert evaluate(rows,[1,2],'base', 'bound_alias_ast')['models']['base']['retention'] is None
    with pytest.raises(ValueError):evaluate(rows,[3],'base')
    with pytest.raises(ValueError):evaluate(rows,[0],'base')
    with pytest.raises(ValueError):evaluate([rows[0],dict(rows[1],dataset='spider')],[1],'base')


@pytest.mark.skipif(shutil.which("git") is None, reason="Git required for installer fixture")
def test_runtime_overlay_preserves_local_edits(tmp_path):
    spec = importlib.util.spec_from_file_location('runtime_setup', ROOT/'scripts/baselines/setup_runtime.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    repo = tmp_path/'upstream';repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    (repo/'module.py').write_text('original = True\n')
    subprocess.run(['git','-C',str(repo),'add','module.py'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.com','commit','-qm','fixture'],check=True)
    head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    overlay = tmp_path/'overlay';overlay.mkdir();(overlay/'module.py').write_text('adapted = True\n')
    record = {'path':'upstream','overlay':'overlay','commit':head,'files':{'module.py':{'upstream_sha256':module.digest(repo/'module.py'),'sha256':module.digest(overlay/'module.py')}}}
    module.install_overlay(tmp_path,record,True)
    module.install_overlay(tmp_path,record,False)
    (repo/'module.py').write_text('local_edit = True\n')
    with pytest.raises(ValueError,match='local edits'):
        module.install_overlay(tmp_path,record,True)
    assert (repo/'module.py').read_text() == 'local_edit = True\n'


def test_rollout_collection_retains_failures_and_rejects_missing_samples(tmp_path):
    spec = importlib.util.spec_from_file_location('collect_samples', ROOT/'scripts/evaluation/collect_samples.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    run = tmp_path/'run'; folder = run/'samples'/'q0000'; folder.mkdir(parents=True)
    (run/'summary.json').write_text(json.dumps(dict(status='complete',domain='bird',samples_per_question=2,questions=1)))
    generated = {'records': [dict(problem_id='q',sample_id=i,text='SELECT 1',finish_reason=finish) for i,finish in enumerate(['stop','length'])]}
    scores = {'generated_sha256': module.digest(generated),'scores':[{'correct':True},{'correct':False}]}
    (folder/'000.generated.json').write_text(json.dumps(generated))
    score_file = folder/'000.scored.json'; score_file.write_text(json.dumps(scores))
    output = tmp_path/'complete.jsonl'; module.collect([('base',run)],output,'bird')
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows)==2 and [row['correct'] for row in rows]==[True,False]
    generated['records'].pop(); scores['scores'].pop(); scores['generated_sha256']=module.digest(generated)
    (folder/'000.generated.json').write_text(json.dumps(generated)); score_file.write_text(json.dumps(scores))
    with pytest.raises(ValueError,match='Incomplete'):
        module.collect([('base',run)],tmp_path/'incomplete.jsonl','bird')
