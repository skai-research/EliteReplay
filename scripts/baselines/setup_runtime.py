#!/usr/bin/env python3
"""Recreate the pinned Qwen3.5 runtime, separately from pristine baseline repos.

Run without arguments to check only. --install clones missing runtime repos
and applies the recorded compatibility patch to clean pinned worktrees.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def check(install=False):
    records = json.loads((ROOT / "experiments/baselines/runtime/runtime.lock.json").read_text())["repositories"]
    for r in records:
        dest = ROOT / r["path"]
        patch = ROOT / r["patch"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != r["patch_sha256"]:
            raise ValueError(f"Runtime patch changed: {patch}")
        if not dest.exists() and install:
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", r["url"], str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", r["commit"]], check=True)
        if not (dest / ".git").exists():
            raise FileNotFoundError(f"Missing {dest}; run setup_runtime.py --install")
        head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if head != r["commit"]:
            raise ValueError(f"Unexpected runtime commit at {dest}; existing checkout is preserved")
        ready = all((dest / name).is_file() and hashlib.sha256((dest / name).read_bytes()).hexdigest() == digest
                    for name, digest in r["patched_files"].items())
        if not ready and install:
            if subprocess.check_output(["git", "-C", str(dest), "status", "--porcelain"], text=True).strip():
                raise ValueError(f"Refusing to patch existing changes at {dest}")
            subprocess.run(["git", "-C", str(dest), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(dest), "apply", str(patch)], check=True)
            ready = all(hashlib.sha256((dest / name).read_bytes()).hexdigest() == digest
                        for name, digest in r["patched_files"].items())
        if not ready:
            raise ValueError(f"Runtime compatibility changes missing/different at {dest}")
    return {"status": "verified", "repositories": [r["name"] for r in records]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    print(json.dumps(check(parser.parse_args().install), indent=2))
