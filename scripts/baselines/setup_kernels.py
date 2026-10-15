#!/usr/bin/env python3
"""Install pinned kernels inside the training container, without changing torch.

Run on a compute node. Compilation uses MAX_JOBS=4 by default. The target is
immutable after its manifest is written; each run freezes its own package copy.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from elitereplay.baselines.kernels import DEFAULT_DIRECTORY, LOCK, check, record_build
from elitereplay.baselines.data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--directory", type=Path, default=ROOT / DEFAULT_DIRECTORY)
    args = parser.parse_args()
    root = args.directory.resolve()
    if (root / "manifest.json").exists() or not args.install:
        record = check(root)
        print(json.dumps({"status": "verified", "directory": str(root),
                          "torch": record["torch"], "cuda": record["cuda"]}))
        return
    if (root / "site-packages").exists():
        raise FileExistsError("Incomplete installation exists; choose a new directory and retain the build log")
    import torch
    if not torch.__version__.split("+")[0].startswith("2.11.") or torch.version.cuda != "13.0":
        raise RuntimeError("Run inside the pinned torch2.11/CUDA13 training container")
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    packages = []
    for package in json.loads(LOCK.read_text()):
        target = downloads / package["file"]
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".download")
            urllib.request.urlretrieve(package["url"], temporary)
            if sha256(temporary) != package["sha256"]:
                raise ValueError(f"Kernel download hash mismatch: {target.name}")
            temporary.replace(target)
        if sha256(target) != package["sha256"]:
            raise ValueError(f"Kernel download hash mismatch: {target.name}")
        packages.append(str(target))
    env = {**os.environ, "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
           "CAUSAL_CONV1D_FORCE_BUILD": "TRUE"}
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                    "--target", str(root / "site-packages"), *packages], env=env, check=True)
    # A fresh process ensures imports see the newly installed packages.
    env["PYTHONPATH"] = str(root / "site-packages") + ":" + str(ROOT / "src")
    subprocess.run([sys.executable, "-c", "import sys; from elitereplay.baselines.kernels import record_build; "
                    "record_build(sys.argv[1])", str(root)], env=env, check=True)
    print(json.dumps({"status": "installed", "directory": str(root)}))


if __name__ == "__main__":
    main()
