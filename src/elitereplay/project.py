"""Locate repository assets without depending on the caller's working directory."""

import os
from pathlib import Path

ROOT = Path(
    os.environ.get("ELITEREPLAY_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def require_checkout():
    if not (ROOT / "runtime/locks/runtime.lock.json").is_file():
        raise RuntimeError(
            "This command needs the source checkout. Clone the repository and run "
            "pip install -e '.[data]', or set ELITEREPLAY_REPO_ROOT to your checkout."
        )
    return ROOT
