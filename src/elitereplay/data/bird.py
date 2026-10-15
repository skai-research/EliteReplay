#!/usr/bin/env python3
"""Build frozen BIRD train-development splits and veRL parquet schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


SEED = 42
FILTERED_REVISION = "4068469807b255fcfc0816bdd520946fe460d256"
FILTERED_JSONL_SHA256 = "2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8"
FILTERED_COLUMN_MEANING_SHA256 = "7928ddb444fd6e1b96b13ef6a1a87f4b9cf138e8354873c0ed7ca2878d7a88db"
BIRD_EVALUATOR_COMMIT = "b3d4bcbbae9a96934ad812551eb400c7a3b23c12"
BIRD_EVALUATOR_SHA256 = "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89"

SYSTEM_PROMPT = """You are solving a BIRD text-to-SQL task against one SQLite database. Return exactly one complete read-only SQLite query that answers the question. Use the evidence and the complete schema below. Do not use a tool, ask a follow-up question, or invent tables or columns. Output only SQL, without Markdown fences or explanation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def derive_sql_difficulty(sql: str) -> str:
    """Create deterministic structural strata when BIRD train has no labels.

    BIRD publishes simple/moderate/challenging labels for development, but not
    for either the original or filtered training JSON.  These strata are used
    only to balance the frozen training sample; reported benchmark difficulty
    always comes from the official development labels.
    """

    normalized = re.sub(r"\s+", " ", sql.upper())
    joins = len(re.findall(r"\bJOIN\b", normalized))
    nested_selects = max(0, len(re.findall(r"\bSELECT\b", normalized)) - 1)
    high = nested_selects + len(
        re.findall(r"\b(?:UNION|INTERSECT|EXCEPT)\b|\bOVER\s*\(", normalized)
    )
    medium = joins + len(
        re.findall(
            r"\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bCASE\b|\bDISTINCT\b",
            normalized,
        )
    )
    if high > 0 or joins >= 2 or medium >= 3:
        return "challenging"
    if joins > 0 or medium > 0:
        return "moderate"
    return "simple"


def attach_difficulty(rows: list[dict], original_train_json: Path | None) -> str:
    if original_train_json is None or not original_train_json.is_file():
        for row in rows:
            row["difficulty"] = derive_sql_difficulty(str(row["SQL"]))
        return "derived_sql_structure"
    original = json.loads(original_train_json.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    question_lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in original:
        difficulty = str(row.get("difficulty") or "").lower()
        if not difficulty:
            continue
        lookup[(str(row["db_id"]), str(row["question"]), str(row["SQL"]))].append(difficulty)
        question_lookup[(str(row["db_id"]), str(row["question"]))].append(difficulty)
    if not lookup:
        for row in rows:
            row["difficulty"] = derive_sql_difficulty(str(row["SQL"]))
        return "derived_sql_structure_official_train_has_no_labels"
    unmatched = 0
    for row in rows:
        exact = lookup.get((row["db_id"], row["question"], row["SQL"]), [])
        loose = question_lookup.get((row["db_id"], row["question"]), [])
        candidates = exact or loose
        if candidates and len(set(candidates)) == 1:
            row["difficulty"] = candidates[0]
        else:
            row["difficulty"] = derive_sql_difficulty(str(row["SQL"]))
            unmatched += 1
    if unmatched:
        print(
            f"INFO: derived structural strata for {unmatched} rows because "
            "the official BIRD training release has no difficulty field"
        )
    return "official_train_labels" if not unmatched else "official_or_derived_sql_structure"


def _difficulty_distribution(rows: Iterable[dict]) -> dict[str, float]:
    counts = Counter(str(row.get("difficulty", "unknown")) for row in rows)
    total = sum(counts.values()) or 1
    return {key: value / total for key, value in counts.items()}


def choose_dev_databases(rows: list[dict], seed: int = SEED) -> list[str]:
    """Choose seven held-out DBs, balancing source size and difficulty mix."""

    by_db: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_db[str(row["db_id"])].append(row)
    names = sorted(by_db, key=lambda name: (len(by_db[name]), name))
    if len(names) < 10:
        raise ValueError("BIRD split requires at least ten databases")
    holdout_count = max(1, round(len(names) * 0.10))
    target_rows = len(rows) * holdout_count / len(names)
    global_dist = _difficulty_distribution(rows)

    # Seed one candidate from every source-size quantile, then perform a
    # deterministic random search. Database identity is the domain boundary;
    # the quantile coverage prevents the dev set being only tiny/easy sources.
    rng = random.Random(seed)
    buckets = [
        names[math.floor(i * len(names) / holdout_count) : math.floor((i + 1) * len(names) / holdout_count)]
        for i in range(holdout_count)
    ]

    def objective(selected: list[str]) -> tuple[float, str]:
        selected_rows = [row for name in selected for row in by_db[name]]
        dist = _difficulty_distribution(selected_rows)
        size_error = abs(len(selected_rows) - target_rows) / max(target_rows, 1)
        difficulty_error = sum(
            abs(dist.get(key, 0.0) - global_dist.get(key, 0.0))
            for key in set(dist) | set(global_dist)
        )
        return size_error + difficulty_error, "\x1f".join(sorted(selected))

    candidates: list[list[str]] = []
    for _ in range(50_000):
        chosen = [rng.choice(bucket) for bucket in buckets]
        if len(set(chosen)) == holdout_count:
            candidates.append(chosen)
    if not candidates:
        raise RuntimeError("could not construct a database-balanced BIRD dev split")
    return sorted(min(candidates, key=objective))


def stratified_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Sample without replacement across difficulty and database strata."""

    if count > len(rows):
        raise ValueError(f"cannot sample {count} distinct rows from {len(rows)}")
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("difficulty", "unknown")), str(row["db_id"]))].append(row)
    for key in sorted(groups):
        rng.shuffle(groups[key])

    difficulty_counts = Counter(str(row.get("difficulty", "unknown")) for row in rows)
    exact = {key: count * value / len(rows) for key, value in difficulty_counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda item: (-(exact[item] - quotas[item]), item))[: count - sum(quotas.values())]:
        quotas[key] += 1

    selected: list[dict] = []
    for difficulty in sorted(quotas):
        keys = [key for key in sorted(groups) if key[0] == difficulty]
        cursor = 0
        while quotas[difficulty] > 0:
            available = [key for key in keys if groups[key]]
            if not available:
                break
            key = available[cursor % len(available)]
            selected.append(groups[key].pop())
            quotas[difficulty] -= 1
            cursor += 1
    if len(selected) < count:
        remainder = [row for values in groups.values() for row in values]
        rng.shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    rng.shuffle(selected)
    return selected


def find_database_root(candidate: Path) -> Path:
    candidate = candidate.resolve()
    if any(candidate.glob("*/*.sqlite")):
        return candidate
    matches = sorted(
        path for path in candidate.rglob("train_databases") if any(path.glob("*/*.sqlite"))
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one train_databases directory under {candidate}, found {matches}"
        )
    return matches[0].resolve()


def render_schema(db_path: Path) -> str:
    """Render the complete executable SQLite schema without value annotations.

    The filtered release's generated column descriptions can be much larger
    than the database schema itself (over 90 kB for ``works_cycles``).  They
    are auxiliary value annotations rather than schema, and including them
    makes maintained prompts exceed Qwen's context window.  The canonical DDL
    retains every table, view, column, key, constraint, type, and default.
    Question-specific BIRD evidence remains in the prompt unchanged.
    """

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        objects = connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type, name"
        ).fetchall()
    finally:
        connection.close()
    if not objects:
        raise ValueError(f"database has no visible schema objects: {db_path}")
    rendered: list[str] = []
    for object_type, name, ddl in objects:
        if ddl:
            rendered.append(f"-- {object_type}: {name}\n{str(ddl).rstrip(';')};")
    return "\n\n".join(rendered)


def prompt_for(row: dict, schema: str) -> list[dict[str, str]]:
    evidence = str(row.get("evidence") or "").strip() or "(none)"
    user = (
        f"Database: {row['db_id']}\n\n"
        f"Complete schema:\n{schema}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question:\n{row['question']}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def to_records(
    rows: list[dict], schemas: dict[str, str], *, split: str
) -> list[dict]:
    records = []
    for schedule_index, row in enumerate(rows):
        payload = json.dumps(
            {"db_id": row["db_id"], "gold_sql": row["SQL"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        records.append(
            {
                "data_source": "bird",
                "prompt": prompt_for(row, schemas[row["db_id"]]),
                "ability": "bird_text_to_sql",
                "reward_model": {"style": "rule", "ground_truth": payload},
                "extra_info": {
                    "db_id": row["db_id"],
                    "difficulty": row.get("difficulty", "unknown"),
                    "filtered_row_id": int(row["filtered_row_id"]),
                    "schedule_index": schedule_index,
                    "split": split,
                    "bird_filtered_revision": FILTERED_REVISION,
                    "bird_evaluator_commit": BIRD_EVALUATOR_COMMIT,
                },
            }
        )
    return records


def write_parquet(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-jsonl", type=Path, default=Path("data/bird/raw/train-00000-of-00001.jsonl"))
    parser.add_argument("--column-meanings", type=Path, default=Path("data/bird/raw/train_column_meaning.json"))
    parser.add_argument("--database-root", type=Path, default=Path("external/bird-data"))
    parser.add_argument("--original-train-json", type=Path)
    parser.add_argument("--archive", type=Path, default=Path("external/bird-train.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/bird"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256(args.filtered_jsonl) != FILTERED_JSONL_SHA256:
        raise ValueError("filtered BIRD JSONL hash mismatch")
    if sha256(args.column_meanings) != FILTERED_COLUMN_MEANING_SHA256:
        raise ValueError("BIRD column-meaning hash mismatch")

    rows = load_jsonl(args.filtered_jsonl)
    if len(rows) != 6601:
        raise ValueError(f"expected 6,601 filtered rows, got {len(rows)}")
    for index, row in enumerate(rows):
        row["filtered_row_id"] = index
    difficulty_source = attach_difficulty(rows, args.original_train_json)
    db_root = find_database_root(args.database_root)
    db_ids = sorted({str(row["db_id"]) for row in rows})
    missing = [db_id for db_id in db_ids if not (db_root / db_id / f"{db_id}.sqlite").is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} filtered BIRD databases: {missing}")

    dev_db_ids = choose_dev_databases(rows)
    dev_set = set(dev_db_ids)
    dev_pool = [row for row in rows if row["db_id"] in dev_set]
    rl_pool = [row for row in rows if row["db_id"] not in dev_set]
    schedule = stratified_sample(rl_pool, 2000, SEED)
    competence = stratified_sample(dev_pool, 100, SEED + 1)
    schemas = {
        db_id: render_schema(db_root / db_id / f"{db_id}.sqlite")
        for db_id in db_ids
    }

    out = args.output_dir
    files = {
        "train_schedule": (out / "train_schedule.parquet", to_records(schedule, schemas, split="rl_schedule")),
        "competence_panel": (out / "competence_panel.parquet", to_records(competence, schemas, split="train_dev_competence")),
        "competence_panel_a": (out / "competence_panel_a.parquet", to_records(competence[:50], schemas, split="train_dev_competence_a")),
        "competence_panel_b": (out / "competence_panel_b.parquet", to_records(competence[50:], schemas, split="train_dev_competence_b")),
        "gate_a_panel": (out / "gate_a_panel.parquet", to_records(schedule[:4], schemas, split="gate_a")),
        "dev_panel": (out / "dev_panel.parquet", to_records(stratified_sample(dev_pool, min(256, len(dev_pool)), SEED + 2), schemas, split="train_dev")),
    }
    for path, records in files.values():
        write_parquet(path, records)

    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "filtered_release": {
            "repo": "birdsql/bird23-train-filtered",
            "revision": FILTERED_REVISION,
            "rows": len(rows),
            "jsonl_sha256": FILTERED_JSONL_SHA256,
            "column_meaning_sha256": FILTERED_COLUMN_MEANING_SHA256,
        },
        "prompt_schema": {
            "format": "complete_sqlite_master_ddl",
            "includes_generated_column_meanings": False,
            "reason": "auxiliary value annotations exceed the maintained model context; question evidence is retained",
        },
        "database_archive": {
            "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
            "content_length": 8919543554,
            "etag": "1FBA2891BD215A42CE1004736AB7F47F-400",
            "sha256": sha256(args.archive) if args.archive.is_file() else None,
            "database_root": str(db_root),
        },
        "official_evaluator": {
            "repo": "https://github.com/bird-bench/mini_dev.git",
            "commit": BIRD_EVALUATOR_COMMIT,
            "evaluation_ex_sha256": BIRD_EVALUATOR_SHA256,
        },
        "split": {
            "database_count": len(db_ids),
            "dev_database_count": len(dev_db_ids),
            "dev_databases": dev_db_ids,
            "dev_pool_rows": len(dev_pool),
            "rl_database_count": len(db_ids) - len(dev_db_ids),
            "rl_pool_rows": len(rl_pool),
            "database_overlap": sorted(dev_set & ({row["db_id"] for row in rl_pool})),
        },
        "schedule": {
            "rows": len(schedule),
            "distinct_filtered_rows": len({row["filtered_row_id"] for row in schedule}),
            "groups_per_step": 16,
            "steps": 125,
            "filtered_row_ids": [row["filtered_row_id"] for row in schedule],
            "difficulty_counts": dict(Counter(row["difficulty"] for row in schedule)),
            "difficulty_stratification_source": difficulty_source,
        },
        "competence": {
            "rows": len(competence),
            "filtered_row_ids": [row["filtered_row_id"] for row in competence],
            "database_ids": sorted({row["db_id"] for row in competence}),
        },
        "parquet_sha256": {name: sha256(path) for name, (path, _) in files.items()},
    }
    if manifest["split"]["database_overlap"]:
        raise AssertionError("BIRD RL/development database overlap")
    manifest_path = out / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "dev_databases": dev_db_ids, "counts": {name: len(records) for name, (_, records) in files.items()}}, indent=2))


if __name__ == "__main__":
    main()
