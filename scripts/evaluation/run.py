#!/usr/bin/env python3
"""Prepare, sample, verify and summarize BIRD/Spider using the experiment evaluator."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from evorl.evaluation.config import load, selected, run_dir, atomic_json, digest, sha256
from evorl.evaluation.data import prepare, read_prepared
from evorl.evaluation.engine import run, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model', default='base')
    p.add_argument('action', choices=['prepare', 'run', 'summarize'])
    a = p.parse_args()
    c = selected(load(a.config), a.model)
    if c['domain'] not in ('bird', 'spider'):
        p.error('This release supports BIRD and Spider SQL evaluation')
    if c['runtime']['launcher'] != 'native':
        p.error('Run this command inside the documented container with runtime.launcher=native')
    os.environ.update(BIRD_DATABASE_ROOT=c['verifier']['bird_database_root'],
                      BIRD_EVALUATOR_ROOT=c['verifier']['bird_evaluator_root'],
                      BIRD_SQL_TIMEOUT_SECONDS=str(c['verifier']['bird_timeout_seconds']))
    if a.action == 'prepare':
        print(json.dumps(prepare(c), indent=2)); return
    items, manifest = read_prepared(c)
    out = run_dir(c, a.model)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.release-eval.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = Path(c['models'][a.model]['path'])
        from evorl.evaluation.models import model_files
        files = model_files(c['models'][a.model])
        # Hash every model file and the scorer sources before permitting cache reuse.
        sources = Path(__file__).resolve().parents[2] / 'src'
        identity = {'config': c, 'data_manifest': manifest,
                    'model_files': {str(f): sha256(f) for f in files},
                    'source_files': {str(f.relative_to(sources)): sha256(f) for f in sources.rglob('*.py')}}
        receipt = out / 'release_identity.json'
        if receipt.exists():
            if json.loads(receipt.read_text()) != identity:
                raise ValueError('Evaluation inputs changed; choose a new output_root')
        else:
            if (out / 'samples').exists():
                raise ValueError('Untracked existing samples; use a new output_root')
            atomic_json(receipt, identity)
        if a.action == 'run':
            run(c, a.model, items, out)
        print(json.dumps(summarize(c, a.model, items, out), indent=2))

if __name__ == '__main__':
    main()
