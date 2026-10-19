#!/usr/bin/env python3
"""Fetch pinned public assets. BIRD database archives are acquired separately.

The default downloads only filtered questions and the evaluator. --models also
fetches the base model.
No credentials are stored in the repository; use your normal HF login if needed.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', action='store_true')
    a = p.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    file = hf_hub_download('birdsql/bird23-train-filtered', 'train-00000-of-00001.jsonl',
                           repo_type='dataset', revision='4068469807b255fcfc0816bdd520946fe460d256')
    data = Path(file).read_bytes()
    if hashlib.sha256(data).hexdigest() != '2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8':
        raise ValueError('Filtered BIRD source hash mismatch')
    dest = ROOT / 'data/raw/bird/train-00000-of-00001.jsonl'
    if dest.exists() and dest.read_bytes() != data:
        raise ValueError('Existing filtered source differs; refusing overwrite')
    dest.parent.mkdir(parents=True, exist_ok=True);dest.write_bytes(data)
    repo = ROOT / 'external/bird-mini-dev';commit = 'b3d4bcbbae9a96934ad812551eb400c7a3b23c12'
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        subprocess.run(['git', 'clone', 'https://github.com/bird-bench/mini_dev.git', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', commit], check=True)
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != commit:
        raise ValueError('Existing BIRD evaluator has a different revision')
    if hashlib.sha256((repo/'evaluation/evaluation_ex.py').read_bytes()).hexdigest() != 'da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89':
        raise ValueError('Existing BIRD evaluator differs')
    models = []
    if a.models:models.append(('Qwen/Qwen3.5-4B-Base', '1001bb4d826a52d1f399e183466143f4da7b741b'))
    for name, rev in models:
        print(snapshot_download(name, revision=rev, cache_dir=ROOT/'external/hf-cache/hub'))
    print('Pinned questions and evaluator ready. See README.md for database placement.')

if __name__ == '__main__':
    main()
