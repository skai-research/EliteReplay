#!/usr/bin/env python3
"""Clone/check the public original repositories at the recorded commits.

Default is read-only validation. --clone downloads missing repositories only.
Existing clones are never reset, cleaned, pulled or overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def check(clone=False):
    manifest = json.loads((ROOT / "experiments/baselines/upstream.lock.json").read_text())
    results = []
    for record in manifest["repositories"]:
        directory = ROOT / record["path"]
        if not directory.exists() and clone:
            directory.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                            record["url"], str(directory)], check=True)
            subprocess.run(["git", "-C", str(directory), "checkout", "--detach", record["commit"]], check=True)
        if not (directory / ".git").exists():
            raise FileNotFoundError(f"Missing {directory}; run fetch_upstream.py --clone")
        if git(directory, "rev-parse", "HEAD") != record["commit"]:
            raise ValueError(f"Commit mismatch: {directory}; preserve this checkout and restore the pinned clone separately")
        if git(directory, "remote", "get-url", "origin").removesuffix(".git") != record["url"].removesuffix(".git"):
            raise ValueError(f"Unexpected origin: {directory}")
        if git(directory, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"Modified original source: {directory}; keep adaptations in EvoRL")
        for filename, expected in record["files"].items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {directory / filename}")
        results.append({"method": record["name"], "path": str(directory), "commit": record["commit"], "status": "verified"})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone", action="store_true")
    print(json.dumps(check(parser.parse_args().clone), indent=2))
