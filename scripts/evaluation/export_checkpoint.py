#!/usr/bin/env python3
"""Export one explicitly selected, pinned FSDP actor to an atomic HF directory.

Run inside the pinned training container with verl_fork on PYTHONPATH. Input
checkpoints must be selected/preserved before submission; this never follows
a live latest-checkpoint pointer or substitutes base-model weights.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from elitereplay.evaluation.config import atomic_json, sha256
from elitereplay.evaluation.models import model_files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    actor, output = args.actor.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.export.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        world = json.loads((actor / 'fsdp_config.json').read_text())['world_size']
        sources = [actor / f'model_world_size_{world}_rank_{rank}.pt' for rank in range(world)]
        for source in sources:
            with zipfile.ZipFile(source) as archive:
                if not archive.infolist():
                    raise ValueError(f'Empty model shard: {source}')
        sources += [actor / 'fsdp_config.json']
        sources += sorted(f for f in (actor / 'huggingface').rglob('*') if f.is_file())
        provenance = {'actor': str(actor), 'source_sha256': {str(f): sha256(f) for f in sources}}
        if output.exists():
            if json.loads((output / 'export_provenance.json').read_text()) != provenance:
                raise ValueError('Existing export belongs to different checkpoint inputs')
            model_files({'path': str(output), 'checkpoint_marker': None})
            print(f'Validated existing export: {output}', flush=True)
            return
        staging = output.with_name(output.name + f'.building-{os.getpid()}')
        subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                        '--local_dir', str(actor), '--target_dir', str(staging),
                        '--trust-remote-code'], check=True)
        model_files({'path': str(staging), 'checkpoint_marker': None})
        atomic_json(staging / 'export_provenance.json', provenance)
        staging.rename(output)
        print(f'Export complete: {output}', flush=True)


if __name__ == '__main__':
    main()
