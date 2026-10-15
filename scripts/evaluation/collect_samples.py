#!/usr/bin/env python3
"""Convert complete, hash-linked rollout blocks to the metric interchange JSONL.

Repeat --run MODEL=PATH for each model on the SAME dataset. Preserve failed and
truncated samples. Accepts original experiment output and this release's output.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from evorl.evaluation.config import digest


def collect(runs, output, dataset):
    expected = None
    with Path(output).open('x') as out:
        for model, root in runs:
            root = Path(root)
            summary = json.loads((root / 'summary.json').read_text())
            if summary['status'] != 'complete' or summary['domain'] != dataset:
                raise ValueError(f'Incomplete or wrong-domain evaluation: {root}')
            n, nq = summary['samples_per_question'], summary['questions']
            questions = set()
            for folder in sorted((root / 'samples').glob('q*')):
                seen = set(); qid = None
                for file in sorted(folder.glob('*.generated.json')):
                    generated = json.loads(file.read_text())
                    scored = json.loads(file.with_name(file.name.replace('.generated.json', '.scored.json')).read_text())
                    if scored['generated_sha256'] != digest(generated) or len(scored['scores']) != len(generated['records']):
                        raise ValueError(f'Generated/scored mismatch: {file}')
                    for row, score in zip(generated['records'], scored['scores']):
                        pid, sid = row['problem_id'], row['sample_id']
                        if qid is not None and pid != qid:
                            raise ValueError('Mixed questions in one folder')
                        qid = pid
                        if sid in seen or type(score['correct']) is not bool:
                            raise ValueError('Duplicate sample or non-Boolean correctness')
                        if row['finish_reason'] == 'length' and score['correct']:
                            raise ValueError('Truncated sample marked correct')
                        seen.add(sid)
                        out.write(json.dumps({'dataset': dataset, 'model': model, 'question_id': pid,
                                              'sample_id': sid, 'correct': score['correct'], 'sql': row['text']})+'\n')
                if seen != set(range(n)) or qid in questions:
                    raise ValueError(f'Incomplete/duplicate question: {folder}')
                questions.add(qid)
            if len(questions) != nq:
                raise ValueError('Incomplete question coverage')
            identity = (frozenset(questions), n)
            if expected is not None and identity != expected:
                raise ValueError('Models have different question/sample coverage')
            expected = identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, metavar='MODEL=PATH')
    p.add_argument('--dataset', choices=['bird', 'spider'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = [x.split('=', 1) for x in a.run]
    if any(len(x) != 2 for x in runs) or len({x[0] for x in runs}) != len(runs):
        p.error('Each --run must have a unique MODEL=PATH')
    temporary = a.output.with_suffix(a.output.suffix+'.partial')
    if a.output.exists() or temporary.exists():
        p.error('Output or partial output exists; choose a new file')
    try:
        collect(runs, temporary, a.dataset)
        temporary.replace(a.output)
    except FileExistsError:
        raise
    except Exception:
        if temporary.exists(): temporary.unlink()
        raise

if __name__ == '__main__':
    main()
