"""Read-only environment checks with bounded hardware probes."""

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from elitereplay.project import ROOT


def environment_checks(training=False):
    checks = []

    def add(name, ok, detail):
        checks.append(dict(name=name, ok=bool(ok), detail=detail))

    add(
        "Python",
        (
            sys.version_info[:2] == (3, 12)
            if training
            else (3, 10) <= sys.version_info[:2] < (3, 13)
        ),
        sys.version.split()[0] + (" (training requires 3.12)" if training else ""),
    )
    required = {
        "PyYAML": "6.0.2",
        "pyarrow": "19.0.1",
        "sqlglot": "30.17.0",
        "numpy": "2.2.6",
    }
    if training:
        required.update(
            {
                "torch": "2.11.0",
                "transformers": "5.8.0",
                "vllm": "0.20.2",
                "ray": "2.53.0",
                "TransferQueue": "0.1.7",
            }
        )
    for name, expected in required.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        add(name, actual.split("+")[0] == expected, f"{actual}; expected {expected}")
    if training:
        add(
            "Git",
            shutil.which("git") is not None,
            "required for pinned framework checkout",
        )
        add(
            "CUDA toolkit",
            shutil.which("nvcc") is not None,
            "nvcc is required to compile Qwen kernels",
        )
    return checks


def report(training=False, config=None):
    checks = environment_checks(training)
    kernel_args = []
    if config:
        from elitereplay.training.config import load
        from elitereplay.training.kernels import directory

        kernel_args = ["--directory", str(directory(load(config)))]
    if training:
        for name, script, args in [
            ("Framework sources", "runtime.py", []),
            ("Training kernels", "kernels.py", kernel_args),
        ]:
            file = ROOT / "scripts/setup" / script
            if file.is_file():
                try:
                    p = subprocess.run(
                        [sys.executable, str(file), *args],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    checks.append(
                        dict(
                            name=name,
                            ok=p.returncode == 0,
                            detail=(
                                "verified"
                                if p.returncode == 0
                                else (p.stderr or p.stdout)[-1200:]
                            ),
                        )
                    )
                except subprocess.TimeoutExpired:
                    checks.append(dict(name=name, ok=False, detail="check timed out"))
            else:
                checks.append(
                    dict(name=name, ok=False, detail="source checkout required")
                )
        try:
            p = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    'import json,torch; print(json.dumps({"gpus":torch.cuda.device_count(),"cuda":torch.version.cuda}))',
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            hardware = json.loads(p.stdout) if p.returncode == 0 else {}
            needed = 4
            if config:
                from elitereplay.training.config import load

                c = load(config)
                needed = c["runtime"]["trainer_gpus"] + c["runtime"]["rollout_gpus"]
            checks.append(
                dict(
                    name="GPU allocation",
                    ok=hardware.get("gpus") == needed
                    and hardware.get("cuda") == "13.0",
                    detail=f"{hardware}; need {needed} visible GPUs and PyTorch CUDA 13.0",
                )
            )
        except (subprocess.TimeoutExpired, ValueError):
            checks.append(
                dict(
                    name="GPU allocation",
                    ok=False,
                    detail="unable to inspect PyTorch CUDA",
                )
            )
    if config:
        from elitereplay.training.config import load
        from elitereplay.training.runtime import resolved, preflight

        try:
            result = preflight(resolved(load(config)))
            checks.append(
                dict(
                    name="Training inputs",
                    ok=result["ready"],
                    detail=(
                        "verified" if result["ready"] else "; ".join(result["errors"])
                    ),
                )
            )
        except (ValueError, OSError, KeyError) as exc:
            checks.append(dict(name="Training inputs", ok=False, detail=str(exc)))
    return {"ready": all(c["ok"] for c in checks), "checks": checks}
