"""BIRD text-to-SQL reward using the pinned official execution metric."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PINNED_BIRD_EVALUATOR_COMMIT = "b3d4bcbbae9a96934ad812551eb400c7a3b23c12"
PINNED_BIRD_EX_SHA256 = "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89"


class BirdInfrastructureError(RuntimeError):
    """A missing/mismatched database or evaluator that must stop a run."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@functools.lru_cache(maxsize=1)
def validate_official_evaluator() -> None:
    root = Path(
        os.environ.get(
            "BIRD_EVALUATOR_ROOT", _repo_root() / "external" / "bird-mini-dev"
        )
    ).expanduser().resolve()
    evaluator = root / "evaluation" / "evaluation_ex.py"
    if not evaluator.is_file():
        raise BirdInfrastructureError(f"Pinned BIRD evaluator missing: {evaluator}")
    digest = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    if digest != PINNED_BIRD_EX_SHA256:
        raise BirdInfrastructureError(
            "BIRD evaluator hash mismatch: "
            f"expected {PINNED_BIRD_EX_SHA256}, got {digest}"
        )
    git_dir = root / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            reference = head.partition(":")[2].strip()
            loose_ref = git_dir / reference
            if loose_ref.is_file():
                head = loose_ref.read_text(encoding="utf-8").strip()
            else:
                packed = (git_dir / "packed-refs").read_text(encoding="utf-8")
                matches = [
                    line.split(" ", 1)[0]
                    for line in packed.splitlines()
                    if line and not line.startswith(("#", "^")) and line.endswith(f" {reference}")
                ]
                if len(matches) != 1:
                    raise ValueError(f"could not resolve {reference}")
                head = matches[0]
    except (OSError, ValueError) as exc:
        raise BirdInfrastructureError(
            f"Could not verify BIRD evaluator checkout at {root}: {exc}"
        ) from exc
    if head != PINNED_BIRD_EVALUATOR_COMMIT:
        raise BirdInfrastructureError(
            "BIRD evaluator checkout is not pinned: "
            f"expected {PINNED_BIRD_EVALUATOR_COMMIT}, got {head}"
        )


_FENCED_SQL = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TAGGED_SQL = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.IGNORECASE | re.DOTALL)
_SQL_START = re.compile(r"\b(?:WITH|SELECT)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Extract one complete read query without silently shortening its body."""

    candidate = str(text or "").strip()
    if not candidate:
        return ""
    # Qwen can emit a stray answer while still in its reasoning channel and
    # then emit the actual answer after ``</think>``.  Executing the whole
    # suffix would turn two otherwise valid queries into a SQLite
    # multi-statement error.  The final post-reasoning answer is authoritative;
    # if the closing tag is terminal, retain the preceding text instead.
    if "</think>" in candidate.lower():
        parts = re.split(r"</think>", candidate, flags=re.IGNORECASE)
        candidate = next(
            (part.strip() for part in reversed(parts) if part.strip()), ""
        )
    tagged = _TAGGED_SQL.search(candidate)
    if tagged:
        candidate = tagged.group(1).strip()
    else:
        fenced = _FENCED_SQL.search(candidate)
        if fenced:
            candidate = fenced.group(1).strip()
        else:
            start = _SQL_START.search(candidate)
            if start:
                candidate = candidate[start.start() :].strip()

    # Drop only common answer wrappers after a terminal fence/tag.  We do not
    # split on semicolons: a complete SQL artifact may contain them in string
    # literals, and the immutable connection safely rejects write statements.
    candidate = re.sub(r"\s*(?:```|</sql>)\s*$", "", candidate).strip()
    return candidate


def _database_path(db_id: str) -> Path:
    root = Path(
        os.environ.get(
            "BIRD_DATABASE_ROOT",
            _repo_root() / "external" / "bird-data" / "train_databases",
        )
    ).expanduser().resolve()
    if not db_id or db_id in {".", ".."} or "/" in db_id or "\\" in db_id:
        raise BirdInfrastructureError(f"Invalid BIRD db_id: {db_id!r}")
    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if path.parent.parent != root:
        raise BirdInfrastructureError(f"BIRD database escaped root: {path}")
    if not path.is_file():
        raise BirdInfrastructureError(f"BIRD database missing: {path}")
    return path


def _worker_result(
    *, predicted_sql: str, ground_truth_sql: str, db_path: Path
) -> dict[str, Any]:
    timeout_seconds = float(os.environ.get("BIRD_SQL_TIMEOUT_SECONDS", "30"))
    reference_timeout_seconds = float(
        os.environ.get("BIRD_SQL_REFERENCE_TIMEOUT_SECONDS", "180")
    )
    payload = json.dumps(
        {
            "predicted_sql": predicted_sql,
            "ground_truth_sql": ground_truth_sql,
            "db_path": str(db_path),
            "timeout_seconds": timeout_seconds,
            "reference_timeout_seconds": reference_timeout_seconds,
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "elitereplay.evaluation.verifiers.bird_sql_worker"],
            input=payload,
            capture_output=True,
            text=True,
            # The isolated worker gives prediction and reference SQL separate
            # execution deadlines.  Leave enough wall time for both.
            timeout=timeout_seconds + reference_timeout_seconds + 5.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"executable": False, "execution_correct": False, "error": "timeout"}
    if completed.returncode != 0:
        return {
            "executable": False,
            "execution_correct": False,
            "error": f"worker_exit_{completed.returncode}",
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "executable": False,
            "execution_correct": False,
            "error": "malformed_worker_output",
        }
    if result.get("infrastructure_error"):
        raise BirdInfrastructureError(str(result["infrastructure_error"]))
    return result


def score_bird(
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one BIRD answer and retain executable/correctness diagnostics."""

    validate_official_evaluator()
    try:
        payload = json.loads(ground_truth)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BirdInfrastructureError("Invalid BIRD reward metadata") from exc
    info = dict(extra_info or {})
    db_id = str(info.get("db_id") or payload.get("db_id") or "")
    gold_sql = str(payload.get("gold_sql") or "")
    if not gold_sql:
        raise BirdInfrastructureError(f"Missing BIRD gold SQL for {db_id!r}")
    db_path = _database_path(db_id)
    predicted_sql = extract_sql(solution_str)
    if not predicted_sql:
        result: dict[str, Any] = {
            "executable": False,
            "execution_correct": False,
            "error": "no_sql_extracted",
        }
    else:
        result = _worker_result(
            predicted_sql=predicted_sql,
            ground_truth_sql=gold_sql,
            db_path=db_path,
        )
    executable = bool(result.get("executable", False))
    correct = bool(result.get("execution_correct", False))
    validity_bonus = float(os.environ.get("BIRD_VALIDITY_REWARD", "0"))
    if not 0.0 <= validity_bonus < 1.0:
        raise BirdInfrastructureError("BIRD_VALIDITY_REWARD must be in [0, 1)")
    score = 1.0 if correct else validity_bonus if executable else 0.0
    return {
        "score": score,
        "bird_execution_correct": float(correct),
        "bird_executable": float(executable),
        "bird_is_offspring": float(bool(info.get("is_offspring", False))),
        "bird_db_id": db_id,
        "bird_predicted_rows": int(result.get("predicted_rows", 0)),
        "bird_ground_truth_rows": int(result.get("ground_truth_rows", 0)),
        "evaluator_error": str(result.get("error") or result.get("worker_error") or ""),
    }
