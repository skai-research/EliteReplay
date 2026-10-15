"""Pinned, isolated Qwen3.5 training kernels; never alter shared dependencies."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from .config import ROOT, path
from .data import sha256

DEFAULT_DIRECTORY = "external/qwen35-kernels/torch2.11-cu130-v1"
LOCK = ROOT / "runtime/locks/kernels.lock.json"


def directory(c):
    return path(c["runtime"].get("kernel_directory", DEFAULT_DIRECTORY))


def hashes(root):
    return {str(f.relative_to(root)): sha256(f) for f in sorted(root.rglob("*"))
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"}


def check(root):
    root = Path(root)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise ValueError("Missing pinned Qwen3.5 kernels. Run scripts/setup/kernels.py --install "
                         "inside the training container; see docs/training.md")
    record = json.loads(manifest.read_text())
    if record["lock_sha256"] != sha256(LOCK):
        raise ValueError("Kernel installation does not match kernels.lock.json")
    if hashes(root / "site-packages") != record["files"]:
        raise ValueError("Pinned kernel installation was modified or is incomplete")
    return record


def record_build(root):
    """Run after --no-deps installation, in the pinned Python/CUDA environment."""
    import importlib.metadata
    import platform
    import sys
    import torch
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5
    root = Path(root)
    if not (torch.__version__.split("+")[0].startswith("2.11.")
            and torch.version.cuda == "13.0" and sys.version_info[:2] == (3, 12)
            and transformers.__version__ == "5.8.0"):
        raise RuntimeError("Kernel build requires the pinned Python3.12/torch2.11/CUDA13/Transformers5.8 container")
    for package in json.loads(LOCK.read_text()):
        actual = importlib.metadata.version(package["package"])
        if actual != package["version"]:
            raise RuntimeError(f"Unexpected {package['package']} version {actual}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 kernel verification requires a visible CUDA GPU")
    if not modeling_qwen3_5.is_fast_path_available:
        raise RuntimeError("Qwen3.5 still reports missing fast-path kernels")
    import fla, causal_conv1d
    for module in (fla, causal_conv1d):
        if not Path(module.__file__).resolve().is_relative_to((root / "site-packages").resolve()):
            raise RuntimeError("Kernel import escaped the isolated installation")
    record = {"lock_sha256": sha256(LOCK), "torch": torch.__version__, "cuda": torch.version.cuda,
              "transformers": transformers.__version__, "python": platform.python_version(),
              "files": hashes(root / "site-packages")}
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(root / "manifest.json")
    return record


def require_fast_path():
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5
    if not modeling_qwen3_5.is_fast_path_available:
        raise RuntimeError("Missing Qwen3.5 training kernels; refusing the slow fallback")
    print(json.dumps({"qwen35_fast_path": True, "torch": torch.__version__,
                      "cuda": torch.version.cuda}), flush=True)


@contextmanager
def placeholder_model_kernels():
    """The CPU controller's meta model must not initialize a CUDA norm.

    Actual trainer workers keep the fused norm. This temporary override is only
    for Trinity's parameter-free model used to calculate position IDs.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    original = module.FusedRMSNormGated
    module.FusedRMSNormGated = None
    try:
        yield
    finally:
        module.FusedRMSNormGated = original
