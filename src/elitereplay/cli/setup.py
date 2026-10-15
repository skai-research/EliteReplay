"""One idempotent setup command; downloads and compilation are explicit."""

import shlex
import shutil
import subprocess
import sys

from elitereplay.project import require_checkout


def commands(root, *, skip_kernels=False, references=False):
    commands = [[sys.executable, str(root / "scripts/setup/runtime.py"), "--install"]]
    if references:
        commands.append(
            [sys.executable, str(root / "scripts/setup/upstream.py"), "--clone"]
        )
    if not skip_kernels:
        commands.append(
            [sys.executable, str(root / "scripts/setup/kernels.py"), "--install"]
        )
    return commands


def run(args):
    root = require_checkout()
    plan = commands(root, skip_kernels=args.skip_kernels, references=args.references)
    if not args.dry_run:
        if not shutil.which("git"):
            raise RuntimeError("Install Git before running setup.")
        if not args.skip_kernels:
            from .doctor import environment_checks

            failures = [c for c in environment_checks(training=True) if not c["ok"]]
            if failures:
                raise RuntimeError(
                    "Use the documented training container before setup: "
                    + "; ".join(c["name"] + ": " + c["detail"] for c in failures)
                )
    for command in plan:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)
    if not args.dry_run:
        print(
            "Framework sources ready."
            if args.skip_kernels
            else "Training runtime ready. Run elitereplay doctor --training on your GPU host."
        )
