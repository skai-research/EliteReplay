import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from elitereplay.cli import main
from elitereplay.cli.setup import commands
from elitereplay.training.config import ROOT, load, read_config, trinity_config


def test_inheritance_preserves_shared_settings_and_method_specific_options(tmp_path):
    original = load(ROOT / "configs/training/elitereplay.yaml")
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "extends": str(ROOT / "configs/training/elitereplay.yaml"),
                "training": {"learning_rate": 2e-6},
                "method_options": {"sft_weight": 0.1},
            }
        )
    )
    changed = load(config)
    assert changed["training"]["learning_rate"] == 2e-6
    assert (
        changed["training"]["rollouts_per_prompt"]
        == original["training"]["rollouts_per_prompt"]
    )
    assert changed["method_options"] == dict(original["method_options"], sft_weight=0.1)
    assert load(ROOT / "configs/training/elitereplay.yaml") == original
    assert load(ROOT / "configs/training/grpo.yaml")["method_options"] == {}
    assert "taco_image" not in changed["verifier"]
    config.write_text(
        yaml.safe_dump(
            {
                "extends": str(ROOT / "configs/training/elitereplay.yaml"),
                "training": {"learnig_rate": 1e-6},
            }
        )
    )
    with pytest.raises(ValueError, match="unknown"):
        load(config)


def test_inheritance_rejects_cycles_and_bad_parents(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\n")
    b.write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        read_config(a)
    a.write_text("extends: 3\n")
    with pytest.raises(ValueError, match="extends"):
        read_config(a)


def test_cli_plan_works_from_another_directory(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    p = subprocess.run(
        [sys.executable, "-m", "elitereplay", "config", "--plan"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(p.stdout)
    assert (
        plan["optimizer_updates"] == 100 and plan["total_replay_presentations"] == 12800
    )


def test_cli_routes_train_archive_and_evaluation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "elitereplay.cli.script",
        lambda name, args: calls.append((name, list(map(str, args)))),
    )
    main(["train", "--smoke"])
    assert calls[-1][0] == "training/run.py" and calls[-1][1][-1] == "smoke"
    main(["archive", "import"])
    assert calls[-1][0] == "archive/import_archive.py"
    main(
        [
            "evaluate",
            "--config",
            str(tmp_path / "eval.yaml"),
            "--model",
            "elitereplay",
            "--summarize",
        ]
    )
    assert calls[-1][1][-1] == "summarize"
    main(["evaluate", "--config", str(tmp_path / "eval.yaml"), "--smoke"])
    assert calls[-1][1][-1] == "smoke"
    with pytest.raises(SystemExit):
        main(["train", "--smoke", "--resume"])
    with pytest.raises(SystemExit):
        main(["train", "--typo"])


def test_cli_setup_dry_run_does_not_execute(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("dry-run must not start a subprocess")

    monkeypatch.setattr("subprocess.run", fail)
    main(["setup", "--dry-run"])
    output = capsys.readouterr().out
    assert "setup/runtime.py" in output and "setup/kernels.py" in output
    assert "upstream.py" not in output
    assert len(commands(ROOT, skip_kernels=True, references=True)) == 2


def test_cli_metrics_runs_on_cpu(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    output = tmp_path / "scores.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "elitereplay",
            "metrics",
            str(ROOT / "examples/scored_sql.jsonl"),
            "--k",
            "1",
            "2",
            "4",
            "--bootstrap",
            "0",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    assert json.loads(output.read_text())["models"]


def test_doctor_checks_configured_kernel_directory(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from elitereplay.cli import doctor

    c = load(ROOT / "configs/training/grpo.yaml")
    c["runtime"]["kernel_directory"] = str(tmp_path / "custom-kernels")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(c))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stdout='{"gpus":4,"cuda":"13.0"}', stderr=""
        )

    monkeypatch.setattr(doctor, "environment_checks", lambda training: [])
    monkeypatch.setattr(doctor.subprocess, "run", run)
    monkeypatch.setattr(
        "elitereplay.training.runtime.preflight", lambda c: {"ready": True}
    )
    assert doctor.report(training=True, config=config)["ready"]
    kernel_call = next(
        c for c in calls if any(str(x).endswith("kernels.py") for x in c)
    )
    assert kernel_call[-2:] == ["--directory", str(tmp_path / "custom-kernels")]
