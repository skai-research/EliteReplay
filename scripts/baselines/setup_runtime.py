#!/usr/bin/env python3
"""Verify or install pinned upstream runtimes plus complete, licensed overlay files.

No patch utility or runtime string replacement is required. Local changes are
never overwritten: every target must match either its pinned upstream hash or
its released overlay hash. Run from the repository's training environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def install_overlay(root, record, install=False):
    dest = root / record['path']
    overlay = root / record['overlay']
    for name, hashes in record['files'].items():
        if digest(overlay / name) != hashes['sha256']:
            raise ValueError(f'Released overlay changed: {name}')
    if not dest.exists() and install:
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', record['url'], str(dest)], check=True)
        subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', record['commit']], check=True)
    if not (dest / '.git').exists():
        raise FileNotFoundError(f'Missing {dest}; run setup_runtime.py --install')
    head = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    if head != record['commit']:
        raise ValueError(f'Unexpected runtime commit at {dest}; checkout preserved')
    changed = subprocess.check_output(['git', '-C', str(dest), 'diff', '--name-only', 'HEAD'], text=True).splitlines()
    if set(changed) - set(record['files']):
        raise ValueError(f'Unexpected tracked changes at {dest}: {sorted(set(changed)-set(record["files"]))}')
    # Prevalidate every file before copying any file.
    for name, hashes in record['files'].items():
        actual = digest(dest / name)
        if actual not in (hashes['upstream_sha256'], hashes['sha256']):
            raise ValueError(f'Refusing to overwrite local edits: {dest / name}')
        if not install and actual != hashes['sha256']:
            raise ValueError(f'Overlay missing: {dest / name}; run --install')
    if install:
        for name in record['files']:
            shutil.copyfile(overlay / name, dest / name)
    for name, hashes in record['files'].items():
        if digest(dest / name) != hashes['sha256']:
            raise ValueError(f'Overlay verification failed: {name}')

def check(install=False):
    records = json.loads((ROOT / 'experiments/baselines/runtime/runtime.lock.json').read_text())['repositories']
    for record in records:
        install_overlay(ROOT, record, install)
    return {'status': 'verified', 'repositories': [r['name'] for r in records]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    print(json.dumps(check(parser.parse_args().install), indent=2))
