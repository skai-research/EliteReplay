"""Exact DPH Spider split, explicit direct-SQL prompts and isolated EX scoring."""
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys

from .config import path, sha256
from .data import question_key, reference
from evorl.EliteReplay.bird import parse_bird_sql_strict
from evorl.EliteReplay.domain import DomainInfrastructureError, VerificationResult

DPH_SOURCE_SHA = 'fd8e1cfebc327b5d3ccb5b31216e25747fac5cb535b416ff4a19954419f34b46'
SCORER_SHA = '6bdc73aa82634867fee3ca80a26c92768bab0e171d803144702bd1a96f1bf1e5'
SYSTEM = ('You are solving a Spider text-to-SQL task against one SQLite database. '
          'Return exactly one complete read-only SQLite query that answers the question. '
          'Use the supplied schema. Output only SQL, without Markdown fences, analysis, or explanation.')


def check_source():
    file = Path(__file__).parent / 'vendor/dph_sql_executor.py'
    if sha256(file) != SCORER_SHA:
        raise ValueError('Pinned DPH Spider result comparator changed')


def spider_rows(c):
    source = path(c['data']['source_file'])
    if sha256(source) != DPH_SOURCE_SHA:
        raise ValueError('Spider source differs from exact DPH split')
    imported = json.loads((source.parent / 'manifest.json').read_text())
    raw = [json.loads(line) for line in source.read_text().splitlines()]
    if [r['id'] for r in raw] != list(range(1034)):
        raise ValueError('Spider official question IDs/count changed')
    databases, rows = {}, []
    dbroot = path(c['verifier']['spider_database_root'])
    for r in raw:
        gt = r['reward_model']['ground_truth']
        db = gt['db_id'].split('/')[0]
        if gt['db_id'] != f'{db}/{db}.sqlite' or db not in imported['database_files']:
            raise ValueError('Invalid Spider database identity')
        file = dbroot / gt['db_id']
        key = reference(file)
        if key not in databases:
            actual = sha256(file)
            if actual != imported['database_files'][db]['sha256']:
                raise ValueError(f'Spider database changed: {db}')
            databases[key] = actual
        original = r['prompt'][0]['content']
        context, instructions = original.rsplit('\n\nInstructions:', 1)
        question = context.split('\nQuestion:\n', 1)[1]
        # Retain the author's schema, sample values and question byte-for-byte.
        # Only replace the author's reasoning/fenced-SQL output instructions.
        rows.append({'problem_id': f"spider:official_dev:{r['id']}", 'question_key': question_key(question),
                     'prompt': [{'role':'system','content':SYSTEM}, {'role':'user','content':context}],
                     'reward_model': {'ground_truth': json.dumps({'db_id':db,'gold_sql':gt['gold_sql']})},
                     'extra_info': {'official_id':str(r['id']), 'db_id':db, 'difficulty':'unknown'}})
    return rows, {'source_file':reference(source), 'source_sha256':sha256(source),
                  'database_sha256':databases, 'import_manifest_sha256':sha256(source.parent/'manifest.json'),
                  'prompt_protocol':'dph-schema-direct-sql-v1', 'dph_scorer_sha256':SCORER_SHA}


class SpiderAdapter:
    verifier_version = 'dph-spider-ex:' + SCORER_SHA

    def __init__(self, *, database_root, timeout=15, max_response_bytes=131072):
        check_source()
        self.database_root = Path(database_root).resolve()
        self.timeout = float(timeout)
        self.max_response_bytes = max_response_bytes

    def format_prompt(self, item):
        return item['prompt']

    def parse_strict(self, text):
        return parse_bird_sql_strict(text, max_response_bytes=self.max_response_bytes)

    def verify(self, item, canonical_response):
        parsed = self.parse_strict(canonical_response)
        if not parsed.accepted or parsed.canonical_response != canonical_response:
            return VerificationResult.rejected('strict_parse')
        gt = item['reward_model']['ground_truth']
        if isinstance(gt, str):
            gt = json.loads(gt)
        db = gt['db_id']
        if not isinstance(db, str) or Path(db).name != db or db in ('.','..'):
            raise DomainInfrastructureError('Invalid Spider database ID')
        file = self.database_root / db / f'{db}.sqlite'
        if not file.is_file():
            raise DomainInfrastructureError(f'Missing Spider database: {file}')
        result = self._execute(str(file), gt['gold_sql'], canonical_response)
        correct = result['correct']
        return VerificationResult(score=float(correct), fully_correct=correct,
            executable=result['executable'], per_test_scores=(float(correct),), error=result.get('error'),
            metadata={'db_id':db,'verifier_version':self.verifier_version,
                      **{k:result[k] for k in ('predicted_rows','gold_rows','order_matters') if k in result}})

    @lru_cache(maxsize=2048)
    def _execute(self, database, gold, prediction):
        # Identical SQL may reuse a deterministic verifier outcome, but every
        # sampled attempt remains present in pass@k accounting.
        payload = {'db_path':database,'gold':gold,'prediction':prediction,'timeout':self.timeout}
        try:
            run = subprocess.run([sys.executable, '-m', 'evorl.evaluation.spider_worker'],
                input=json.dumps(payload), text=True, capture_output=True,
                timeout=2*self.timeout+10, check=False)
        except subprocess.TimeoutExpired:
            return {'correct':False,'executable':False,'error':'prediction_or_comparison_wall_timeout'}
        if run.returncode:
            raise DomainInfrastructureError(f'Spider worker exited {run.returncode}: {run.stderr[-500:]}')
        try:
            result = json.loads(run.stdout)
        except ValueError as exc:
            raise DomainInfrastructureError('Malformed Spider worker output') from exc
        if result.get('infrastructure_error'):
            raise DomainInfrastructureError(result['infrastructure_error'])
        return result
