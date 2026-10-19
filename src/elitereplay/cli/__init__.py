"""The public EliteReplay command-line interface; heavy imports stay lazy."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from elitereplay.project import ROOT, require_checkout


def script(name, args):
    root = require_checkout()
    subprocess.run(
        [sys.executable, str(root / "scripts" / name), *map(str, args)], check=True
    )


def config_path(value, default):
    return Path(value).expanduser().resolve() if value else ROOT / default


def parser():
    p = argparse.ArgumentParser(
        prog="elitereplay",
        description="Archive search, replay training, and SQL evaluation.",
    )
    commands = p.add_subparsers(dest="command", required=True)
    setup = commands.add_parser(
        "setup", help="Install pinned runtimes and compile training kernels"
    )
    setup.add_argument(
        "--skip-kernels",
        action="store_true",
        help="Install framework sources only; training still needs kernels",
    )
    setup.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without downloads or changes",
    )
    doctor = commands.add_parser(
        "doctor", help="Check dependencies and explain missing setup steps"
    )
    doctor.add_argument(
        "--training",
        action="store_true",
        help="Check pinned GPU environment, runtime sources, and kernels",
    )
    doctor.add_argument(
        "--config",
        type=Path,
        help="Also validate training assets for this configuration",
    )
    doctor.add_argument(
        "--json", action="store_true", help="Print a machine-readable report"
    )
    config = commands.add_parser(
        "config", help="Show the resolved training configuration or compute budget"
    )
    config.add_argument("--config", type=Path)
    config.add_argument("--plan", action="store_true")
    data = commands.add_parser(
        "data", help="Download assets and prepare BIRD/Spider data"
    )
    ds = data.add_subparsers(dest="action", required=True)
    for name, description in [
        ("fetch", "Download pinned questions, evaluator, and optional models"),
        ("spider", "Import official Spider databases"),
    ]:
        ds.add_parser(name, help=description, add_help=False)
    prepare = ds.add_parser(
        "prepare", help="Prepare the BIRD training and archive splits"
    )
    prepare.add_argument("--config", type=Path)
    archive = commands.add_parser(
        "archive", help="Import the released archive or run new search"
    )
    archive.add_argument("action", choices=["import", "build", "seeds"])
    archive.add_argument("--config", type=Path)
    train = commands.add_parser(
        "train", help="Train a policy, check assets, or resume a checkpoint"
    )
    train.add_argument("--config", type=Path)
    flags = train.add_mutually_exclusive_group()
    for name in ["resume", "check", "prepare"]:
        flags.add_argument(
            "--" + name,
            action="store_true",
            help={
                "resume": "Resume a complete checkpoint",
                "check": "Check inputs without launching training",
                "prepare": "Build the EliteReplay archive",
            }[name],
        )
    evaluate = commands.add_parser(
        "evaluate", help="Generate and verify BIRD/Spider SQL"
    )
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--model", default="base")
    flags = evaluate.add_mutually_exclusive_group()
    flags.add_argument("--prepare", action="store_true")
    flags.add_argument("--summarize", action="store_true")
    for name, desc in [
        ("metrics", "Compute pass@k and diversity from scored samples"),
        ("rollouts", "Convert verified rollout blocks to scored JSONL"),
        ("export", "Merge a training checkpoint into HF model weights"),
    ]:
        commands.add_parser(name, help=desc, add_help=False)
    return p


def main(argv=None):
    p = parser()
    args, extra = p.parse_known_args(argv)
    passthrough = args.command in ("metrics", "rollouts", "export") or (
        args.command == "data" and args.action in ("fetch", "spider")
    )
    if extra and not passthrough:
        p.error("unrecognized arguments: " + " ".join(extra))
    try:
        if args.command == "metrics":
            from elitereplay.evaluation.diversity import main as metrics

            old = sys.argv
            sys.argv = ["elitereplay metrics", *extra]
            try:
                metrics()
            finally:
                sys.argv = old
        elif args.command in ("rollouts", "export"):
            script(
                "evaluation/"
                + {"rollouts": "collect_samples.py", "export": "export_checkpoint.py"}[
                    args.command
                ],
                extra,
            )
        elif args.command == "setup":
            from .setup import run

            run(args)
        elif args.command == "doctor":
            from .doctor import report

            result = report(
                training=args.training or args.config is not None, config=args.config
            )
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for check in result["checks"]:
                    print(
                        f"{'OK' if check['ok'] else 'MISSING'}  {check['name']}: {check['detail']}"
                    )
            if not result["ready"]:
                raise SystemExit(1)
        elif args.command == "config":
            from elitereplay.training.config import load, plan
            import yaml

            c = load(config_path(args.config, "configs/training/elitereplay.yaml"))
            print(
                json.dumps(plan(c), indent=2)
                if args.plan
                else yaml.safe_dump(c, sort_keys=False)
            )
        elif args.command == "data":
            if args.action == "prepare":
                script(
                    "data/prepare_bird.py",
                    ["--config", config_path(args.config, "configs/data/bird.yaml")],
                )
            else:
                script(
                    "data/"
                    + {"fetch": "fetch.py", "spider": "prepare_spider.py"}[args.action],
                    extra,
                )
        elif args.command == "archive":
            c = config_path(args.config, "configs/training/elitereplay.yaml")
            if args.action == "import":
                script("archive/import_archive.py", ["--config", c])
            else:
                script(
                    "training/run.py",
                    [
                        "--config",
                        c,
                        "prepare-seeds" if args.action == "seeds" else "prepare",
                    ],
                )
        elif args.command == "train":
            action = next(
                (
                    name
                    for name in ["resume", "check", "prepare"]
                    if getattr(args, name.replace("-", "_"))
                ),
                "train",
            )
            script(
                "training/run.py",
                [
                    "--config",
                    config_path(args.config, "configs/training/elitereplay.yaml"),
                    action,
                ],
            )
        elif args.command == "evaluate":
            action = (
                "prepare"
                if args.prepare
                else "summarize" if args.summarize else "run"
            )
            script(
                "evaluation/run.py",
                ["--config", args.config.resolve(), "--model", args.model, action],
            )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        p.exit(2, f"Error: {exc}\n")
