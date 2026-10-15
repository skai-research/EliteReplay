"""Resource-bounded SQLite execution worker for the BIRD reward.

The worker receives one JSON object on stdin and emits one JSON object on
stdout.  Keeping model-written SQL in a short-lived process prevents a
pathological query from retaining memory or SQLite state in a Ray reward
worker.
"""

from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from urllib.parse import quote

try:
    # The pinned training image ships SQLite 3.37, which cannot execute valid
    # BIRD gold queries containing RIGHT/FULL OUTER JOIN.  The workspace pins
    # pysqlite3-binary so every reward worker uses a modern SQLite engine while
    # retaining the standard DB-API and the official BIRD EX set semantics.
    import pysqlite3 as sqlite3
except ImportError:  # Host-side unit tests use the system SQLite when modern.
    import sqlite3


def _set_limits() -> None:
    memory_gib = float(os.environ.get("BIRD_SQL_MEMORY_GIB", "4"))
    prediction_timeout = float(os.environ.get("BIRD_SQL_TIMEOUT_SECONDS", "30"))
    reference_timeout = float(
        os.environ.get("BIRD_SQL_REFERENCE_TIMEOUT_SECONDS", "180")
    )
    # Prediction and reference execution have independent deadlines below.
    # Size the default process-wide CPU allowance for both queries while still
    # permitting an explicit deployment override.
    default_cpu_seconds = max(
        1, math.ceil(prediction_timeout + reference_timeout + 1)
    )
    cpu_seconds = int(
        os.environ.get("BIRD_SQL_CPU_SECONDS", str(default_cpu_seconds))
    )
    memory_bytes = max(256 * 1024**2, int(memory_gib * 1024**3))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=1.0)
    connection.execute("PRAGMA query_only = ON")
    # Avoid a single returned cell allocating most of the worker's limit.
    if hasattr(connection, "setlimit"):
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 64 * 1024**2)
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 2 * 1024**2)
        connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 2000)
    return connection


def _set_semantics_query(sql: str) -> str:
    """Deduplicate inside SQLite, matching the official evaluator's set()."""

    query = sql.strip()
    if query.endswith(";"):
        query = query[:-1].rstrip()
    return f'SELECT DISTINCT * FROM ({query}) AS "_elitereplay_set_result"'


_BIRD_DEV_701_REFERENCE = 'SELECT CAST(SUM(CASE WHEN T2.Score > 50 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.Id) FROM users T1 INNER JOIN posts T2 ON T1.Id = T2.OwnerUserId INNER JOIN ( SELECT MAX(Reputation) AS max_reputation FROM users ) T3 ON T1.Reputation = T3.max_reputation'


def _reference_query(sql: str) -> str:
    """Materialize one exact, uncorrelated gold aggregate; never rewrite predictions.

    SQLite 3.51.1 plans BIRD-dev 20240627 question 701's singleton MAX
    relation as a co-routine inside the posts loop. MATERIALIZED computes
    the identical singleton once. Join predicates, ties, NULL behavior,
    and final aggregate expressions remain unchanged.
    """
    if sql.strip().rstrip(';').rstrip() != _BIRD_DEV_701_REFERENCE:
        return sql
    return (
        'WITH "_elitereplay_gold_max" AS MATERIALIZED '
        '(SELECT MAX(Reputation) AS max_reputation FROM users) '
        + _BIRD_DEV_701_REFERENCE.replace(
            '( SELECT MAX(Reputation) AS max_reputation FROM users ) T3',
            '"_elitereplay_gold_max" T3',
        )
    )


def _execute(payload: dict) -> dict:
    path = Path(str(payload["db_path"])).resolve()
    predicted_sql = str(payload["predicted_sql"])
    ground_truth_sql = str(payload["ground_truth_sql"])
    prediction_timeout_seconds = float(payload.get("timeout_seconds", 30.0))
    reference_timeout_seconds = float(
        payload.get("reference_timeout_seconds", prediction_timeout_seconds)
    )
    deadline = time.monotonic() + prediction_timeout_seconds

    connection = _connect_read_only(path)

    def progress() -> int:
        return int(time.monotonic() >= deadline)

    connection.set_progress_handler(progress, 1000)
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(_set_semantics_query(predicted_sql))
            predicted_result = cursor.fetchall()
        except Exception as exc:
            return {
                "executable": False,
                "execution_correct": False,
                "error": f"prediction: {type(exc).__name__}: {exc}",
            }

        # A slow but executable prediction must not consume the reference
        # query's deadline.  The reference is trusted evaluation data; if it
        # fails under its own full allowance, that is an infrastructure error.
        deadline = time.monotonic() + reference_timeout_seconds
        try:
            cursor.execute(_set_semantics_query(_reference_query(ground_truth_sql)))
            ground_truth_result = cursor.fetchall()
        except Exception as exc:
            return {
                "infrastructure_error": (
                    f"ground truth: {type(exc).__name__}: {exc}"
                )
            }

        # This is the exact BIRD EX contract in the pinned evaluator:
        # calculate_ex returns one iff the sets of result tuples are equal.
        correct = set(predicted_result) == set(ground_truth_result)
        return {
            "executable": True,
            "execution_correct": bool(correct),
            "predicted_rows": len(predicted_result),
            "ground_truth_rows": len(ground_truth_result),
        }
    finally:
        connection.close()


def main() -> None:
    try:
        _set_limits()
        payload = json.loads(sys.stdin.read())
        result = _execute(payload)
    except BaseException as exc:
        result = {"worker_error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
