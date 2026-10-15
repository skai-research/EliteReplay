"""Shared deterministic training splits. No official evaluation data is used."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .config import ROOT, path, fingerprint


def sha256(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proportional_quotas(count, reference_counts):
    """Exact integer largest remainders; reference counts may precede filtering."""
    weights = {str(k): int(v) for k, v in reference_counts.items() if v > 0}
    total = sum(weights.values())
    if count < 0 or not total:
        raise ValueError("Need nonnegative sample size and positive reference counts")
    divided = {k: divmod(count * v, total) for k, v in weights.items()}
    quotas = {k: v[0] for k, v in divided.items()}
    for k in sorted(weights, key=lambda k: (-divided[k][1], k))[:count - sum(quotas.values())]:
        quotas[k] += 1
    return quotas


def stratified(rows, count, seed, key, *, reference_counts=None):
    """Random sampling within fixed proportional strata, without replacement."""
    if not 0 <= count <= len(rows):
        raise ValueError(f"Cannot select {count} from {len(rows)} rows")
    groups = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    rng = random.Random(seed)
    if count == 0:
        return [], list(rows)
    for k in sorted(groups):
        rng.shuffle(groups[k])
    sizes = proportional_quotas(count, reference_counts if reference_counts is not None
                               else {k: len(v) for k, v in groups.items()})
    for k, quota in sizes.items():
        if quota > len(groups.get(k, [])):
            raise ValueError(f"Stratum {k} needs {quota} eligible rows, has {len(groups.get(k, []))}; "
                             "refusing to change the requested proportions")
    for k in groups:
        sizes.setdefault(k, 0)
    selected = [row for k in sorted(groups) for row in groups[k][:sizes[k]]]
    remaining = [row for k in sorted(groups) for row in groups[k][sizes[k]:]]
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def bird_rows(c):
    spec = importlib.util.spec_from_file_location("baseline_bird_preprocessing", ROOT / "data/preprocess_bird.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    raw_path = path(c["source_file"])
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    for i, row in enumerate(raw):
        row["filtered_row_id"] = i
    difficulty_source = helper.attach_difficulty(raw, None)
    databases = set(c["heldout_databases"])
    eligible = [r for r in raw if r["db_id"] not in databases]
    validation = [r for r in raw if r["db_id"] in databases]
    if len(validation) != c["validation_questions"]:
        raise ValueError(f"BIRD heldout DBs contain {len(validation)} rows, expected {c['validation_questions']}")
    train, _ = stratified(eligible, c["train_questions"], c["seed"], lambda r: (r["db_id"], r["difficulty"]))
    used_dbs = {r["db_id"] for r in train + validation}
    dbroot = path(c["database_root"])
    schemas = {db: helper.render_schema(dbroot / db / f"{db}.sqlite") for db in used_dbs}
    def convert(rows, split):
        records = helper.to_records(rows, schemas, split=split)
        for record, raw_row in zip(records, rows):
            record["extra_info"]["problem_id"] = f"bird:{raw_row['filtered_row_id']:06d}"
        return records
    return convert(train, "train"), convert(validation, "validation"), {
        "raw_sha256": sha256(raw_path), "raw_rows": len(raw), "eligible_rows": len(eligible),
        "heldout_databases": sorted(databases), "source_revision": helper.FILTERED_REVISION,
        "difficulty_source": difficulty_source,
        "raw_difficulty_counts": dict(sorted(Counter(r["difficulty"] for r in raw).items())),
        "eligible_difficulty_counts": dict(sorted(Counter(r["difficulty"] for r in eligible).items()))}


def taco_rows(c):
    import pyarrow.parquet as pq
    if c["source_file"]:
        source = path(c["source_file"])
        sources = [source]
        provenance = {"source_file": str(source), "raw_sha256": sha256(source)}
    else:
        from huggingface_hub import HfApi, hf_hub_download
        info = HfApi().dataset_info("BAAI/TACO", revision=c["source_revision"])
        revision = info.sha
        names = sorted(s.rfilename for s in info.siblings)
        # ALL is the official all-difficulty configuration at the frozen revision.
        chosen = [n for n in names if n.startswith("ALL/train-") and n.endswith(".parquet")]
        if not chosen:
            chosen = [n for n in names if n.startswith("data/train-") and n.endswith(".parquet")]
        if not chosen:
            raise ValueError("No official all-difficulty TACO training shards found")
        sources = [Path(hf_hub_download("BAAI/TACO", n, repo_type="dataset", revision=revision,
                   cache_dir=ROOT / "external/dataset-cache/hub")) for n in chosen]
        provenance = {"source": "BAAI/TACO", "revision": revision, "split": "train",
                      "shard_sha256": {n: sha256(f) for n, f in zip(chosen, sources)}}

    def raw_rows():
        columns = ["question", "starter_code", "input_output", "difficulty", "source"]
        for source in sources:
            parquet = pq.ParquetFile(source)
            available = [k for k in columns if k in parquet.schema_arrow.names]
            for batch in parquet.iter_batches(batch_size=64, columns=available):
                yield from batch.to_pylist()

    unique, malformed, duplicate = {}, 0, 0
    raw_counts, excluded, raw_count = Counter(), [], 0
    # First pass retains question metadata only: the full raw test payload is
    # much larger than the selected training set and should not all live in RAM.
    for index, row in enumerate(raw_rows()):
        raw_count += 1
        difficulty = str(row.get("difficulty") or "unknown")
        raw_counts[difficulty] += 1
        try:
            tests = row["input_output"]
            tests = json.loads(tests) if isinstance(tests, str) else tests
            if (not isinstance(tests, dict) or not isinstance(tests.get("inputs"), list)
                    or not isinstance(tests.get("outputs"), list) or not tests["inputs"]
                    or len(tests["inputs"]) != len(tests["outputs"])):
                raise ValueError("Invalid tests")
            question = row["question"].strip()
            if not question:
                raise ValueError("Empty question")
        except (ValueError, KeyError, TypeError, AttributeError):
            malformed += 1
            excluded.append({"source_row_id": index, "difficulty": difficulty, "reason": "malformed"})
            continue
        # Ignore labels, tests and solutions when defining question identity.
        pid = "taco:" + hashlib.sha256(" ".join(question.split()).encode()).hexdigest()
        if pid in unique:
            duplicate += 1
            excluded.append({"source_row_id": index, "problem_id": pid, "difficulty": difficulty,
                             "reason": "duplicate_question"})
            continue
        starter = row.get("starter_code") or ""
        prompt = question + ("\n\nStarter code:\n" + starter if starter else "")
        unique[pid] = {"data_source": "taco", "ability": "code",
            "prompt": [{"role": "user", "content": prompt}],
            "extra_info": {"problem_id": pid, "difficulty": difficulty, "source_row_id": index,
                           "source": str(row.get("source", "unknown"))}}
    strata = lambda row: row["extra_info"]["difficulty"]
    train, remaining = stratified(list(unique.values()), c["train_questions"], c["seed"] + 1,
                                  strata, reference_counts=raw_counts)
    validation, _ = stratified(remaining, c["validation_questions"], c["seed"], strata,
                               reference_counts=raw_counts)
    selected = {r["extra_info"]["source_row_id"]: r for r in train + validation}
    for index, row in enumerate(raw_rows()):
        if index in selected:
            tests = row["input_output"]
            selected[index]["reward_model"] = {"style": "rule", "ground_truth": tests
                if isinstance(tests, str) else json.dumps(tests)}
    if any("reward_model" not in r for r in selected.values()):
        raise ValueError("TACO source changed between metadata and payload passes")
    provenance.update(raw_rows=raw_count, malformed_rows=malformed, duplicate_questions=duplicate,
        eligible_rows=len(unique), raw_difficulty_counts=dict(sorted(raw_counts.items())),
        eligible_difficulty_counts=dict(sorted(Counter(strata(r) for r in unique.values()).items())),
        sampling="uniform_without_replacement_within_raw_difficulty_quotas", exclusions=excluded)
    return train, validation, provenance


def validate_splits(splits):
    ids = {name: [r["extra_info"]["problem_id"] for r in rows] for name, rows in splits.items()}
    if any(len(v) != len(set(v)) for v in ids.values()):
        raise ValueError("Duplicate question IDs in a split")
    if set(ids["train"]) & set(ids["validation"]):
        raise ValueError("Training/validation question overlap")
    if not set(ids["aux_search"]) <= set(ids["train"]):
        raise ValueError("Archive questions must be a subset of RL training questions")
    return ids


def prepare(c):
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = path(c["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace prepared splits at {output}; select a new output_dir")
    if c["domain"] == "bird":
        train, validation, provenance = bird_rows(c)
    elif c["domain"] == "taco":
        train, validation, provenance = taco_rows(c)
    else:
        raise ValueError("Expected bird or taco")
    if c["domain"] == "taco":
        auxiliary, _ = stratified(train, c["archive_questions"], c["seed"] + 2,
            lambda r: r["extra_info"]["difficulty"], reference_counts=provenance["raw_difficulty_counts"])
    else:
        auxiliary, _ = stratified(train, c["archive_questions"], c["seed"] + 2,
            lambda r: (r["extra_info"]["db_id"], r["extra_info"]["difficulty"]))
    splits = {"train": train, "aux_search": auxiliary, "validation": validation}
    ids = validate_splits(splits)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        pq.write_table(pa.Table.from_pylist(rows), output / f"{name}.parquet",
                       compression="zstd", compression_level=9, row_group_size=256)
    manifest = {"schema_version": 1, "domain": c["domain"], "seed": c["seed"],
        "config": c, "config_sha256": fingerprint(c), "source": provenance,
        "archive_is_training_subset": True,
        "splits": {name: {"rows": len(rows), "problem_ids": ids[name],
            "record_sha256": [fingerprint(r) for r in rows],
            "difficulty_counts": dict(sorted(Counter(r["extra_info"]["difficulty"] for r in rows).items())),
            "source_counts": dict(sorted(Counter(r["extra_info"].get("db_id", r["extra_info"].get("source"))
                for r in rows).items()))} for name, rows in splits.items()},
        "parquet_sha256": {name: sha256(output / f"{name}.parquet") for name in splits}}
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output_dir": str(output), "rows": {k: len(v) for k, v in splits.items()}}
