"""Full official evaluation splits; no filtering, subsampling or gold in prompts."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .config import ROOT, atomic_json, digest, path, sha256


def question_key(question):
    return hashlib.sha256(" ".join(question.split()).encode()).hexdigest()


def reference(filename):
    """Keep release paths portable when inputs are inside the checkout."""
    filename = path(filename)
    return str(filename.relative_to(ROOT)) if filename.is_relative_to(ROOT) else str(filename)


def bird_helper():
    from elitereplay.data import bird
    return bird


def bird_rows(c):
    source = path(c["data"]["source_file"])
    raw = json.loads(source.read_text())
    if not isinstance(raw, list):
        raise ValueError("BIRD expects the original official question JSON array")
    helper = bird_helper()
    dbroot = path(c["verifier"]["bird_database_root"])
    schemas, databases, rows = {}, {}, []
    for index, r in enumerate(raw):
        db = r["db_id"]
        if not isinstance(db, str) or not db or Path(db).name != db or db in (".", "..") or "\\" in db:
            raise ValueError("Invalid BIRD database ID")
        if not r.get("SQL"):
            raise ValueError("BIRD gold SQL is unavailable. Hidden test scoring needs organizer assets; never substitute train data.")
        if not r.get("question", "").strip():
            raise ValueError("Empty BIRD question")
        if db not in schemas:
            dbpath = dbroot / db / f"{db}.sqlite"
            schemas[db] = helper.render_schema(dbpath)
            databases[reference(dbpath)] = sha256(dbpath)
        qid = str(r.get("question_id", index))
        rows.append({"problem_id": f"bird:{c['data']['split']}:{qid}", "question_key": question_key(r["question"]),
            "prompt": helper.prompt_for(r, schemas[db]),
            "reward_model": {"ground_truth": json.dumps({"db_id": db, "gold_sql": r["SQL"]})},
            "extra_info": {"db_id": db, "difficulty": r.get("difficulty", "unknown"), "official_id": qid}})
    return rows, {"source_file": reference(source), "source_sha256": sha256(source), "database_sha256": databases}


def audit_training(c, rows):
    import pyarrow.parquet as pq
    root = path(c["data"]["training_data_root"])
    if (root / "payload_manifest.json").exists():
        from elitereplay.training.shared_data import materialize_domain
        materialize_domain(root)
    manifest = json.loads((root / "split_manifest.json").read_text())
    eval_keys = {r["question_key"] for r in rows}
    eval_dbs = {r["extra_info"].get("db_id") for r in rows} if c["domain"] in ("bird", "spider") else set()
    hashes = {reference(root / "split_manifest.json"): sha256(root / "split_manifest.json")}
    for split in ("train", "aux_search", "validation"):
        filename = root / f"{split}.parquet"
        actual = sha256(filename)
        if actual != manifest["parquet_sha256"][split]:
            raise ValueError("Training split hash mismatch")
        hashes[reference(filename)] = actual
        for r in pq.read_table(filename).to_pylist():
            if c["domain"] in ("bird", "spider"):
                if r["extra_info"]["db_id"] in eval_dbs:
                    raise ValueError(f"Official SQL evaluation databases overlap {split}")
                if c["domain"] == "spider":
                    messages = r["prompt"]
                    text = "\n".join(m["content"] for m in messages)
                    question = text.rsplit("Question:\n", 1)[-1]
                    if question_key(question) in eval_keys:
                        raise ValueError(f"Spider evaluation question overlaps {split}")
    return hashes


def validate_rows(c, rows):
    if len(rows) != c["data"]["expected_questions"]:
        raise ValueError(f"Expected the full {c['data']['expected_questions']}-question split, found {len(rows)}")
    ids = [r["problem_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate official question IDs")


def recipe(c):
    data = dict(c["data"])
    for key in ("source_file", "prepared_dir", "training_data_root"):
        if data[key] is not None:
            data[key] = reference(data[key])
    return {"domain": c["domain"], "data": data,
            "bird_database_root": reference(c["verifier"]["bird_database_root"]) if c["domain"] == "bird" else None,
            **({"spider_database_root": reference(c["verifier"]["spider_database_root"]),
                "prompt_protocol": "dph-schema-direct-sql-v1"} if c["domain"] == "spider" else {})}


def materialize_payload(root, payload):
    """Restore a compressed payload distributed in Git-sized binary parts."""
    root = Path(root)
    if payload["filename"] != "questions.jsonl.gz":
        raise ValueError("Invalid evaluation payload filename")
    destination = root / payload["filename"]
    if not destination.exists() and payload.get("parts"):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".payload.", delete=False) as output:
                temporary = Path(output.name)
                for part in payload["parts"]:
                    if Path(part["filename"]).name != part["filename"]:
                        raise ValueError("Invalid evaluation payload part filename")
                    source = root / part["filename"]
                    if source.stat().st_size != part["bytes"] or sha256(source) != part["sha256"]:
                        raise ValueError("Shared evaluation gzip part changed")
                    with source.open("rb") as handle:
                        shutil.copyfileobj(handle, output)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != payload["bytes"] or sha256(temporary) != payload["sha256"]:
                raise ValueError("Restored evaluation gzip changed")
            if destination.exists():
                if sha256(destination) != payload["sha256"]:
                    raise ValueError("Another process wrote different evaluation gzip")
            else:
                temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    if destination.stat().st_size != payload["bytes"] or sha256(destination) != payload["sha256"]:
        raise ValueError("Shared evaluation gzip changed")
    return destination


def materialize(root, manifest):
    """Restore missing JSONL from the Git-distributed gzip, checking exact bytes."""
    root = Path(root)
    lock = root.parent / "release.json"
    if lock.exists():
        expected = json.loads(lock.read_text())["domains"][root.name]["manifest_sha256"]
        if sha256(root / "manifest.json") != expected:
            raise ValueError("Shared evaluation manifest has changed")
    destination = root / "questions.jsonl"
    if destination.exists():
        return
    payload = manifest.get("payload")
    if not payload or payload["filename"] != "questions.jsonl.gz":
        raise FileNotFoundError(f"Missing evaluation records: {destination}")
    source = materialize_payload(root, payload)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".questions.", delete=False) as output:
            temporary = Path(output.name)
            with gzip.open(source, "rb") as handle:
                shutil.copyfileobj(handle, output)
            output.flush()
            os.fsync(output.fileno())
        if sha256(temporary) != manifest["records_sha256"]:
            raise ValueError("Restored evaluation records changed")
        if destination.exists():
            if sha256(destination) != manifest["records_sha256"]:
                raise ValueError("Another process wrote different evaluation records")
        else:
            temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_prepared(c):
    root = path(c["data"]["prepared_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    materialize(root, manifest)
    if manifest["recipe_sha256"] != digest(recipe(c)) or manifest["records_sha256"] != sha256(root / "questions.jsonl"):
        raise ValueError("Prepared evaluation data changed; use a new prepared_dir")
    rows = [json.loads(s) for s in (root / "questions.jsonl").read_text().splitlines()]
    validate_rows(c, rows)
    if [r["problem_id"] for r in rows] != manifest["problem_ids"]:
        raise ValueError("Evaluation question order/identity changed")
    return rows, manifest


def prepare(c):
    root = path(c["data"]["prepared_dir"])
    if (root / "manifest.json").exists():
        rows, manifest = read_prepared(c)
        return {"status": "already_prepared", "questions": len(rows), "source": manifest["source"]}
    # The shared BIRD folder also carries the original public source files.
    source_files = {"README.md", "official_dev.json", "official_tables.json", "official_tied_append.json"}
    if root.exists() and any(p.name not in source_files for p in root.iterdir()):
        raise FileExistsError(f"Incomplete/nonempty evaluation data at {root}; inspect it or choose a new prepared_dir")
    if c["domain"] == "spider":
        from .spider import spider_rows
        rows, provenance = spider_rows(c)
    else:
        rows, provenance = bird_rows(c)
    validate_rows(c, rows)
    training = audit_training(c, rows)
    root.mkdir(parents=True, exist_ok=True)
    records = root / "questions.jsonl"
    records.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    atomic_json(root / "manifest.json", {"schema_version": 2, "recipe_sha256": digest(recipe(c)),
        "domain": c["domain"], "split": c["data"]["split"], "release": c["data"]["release"],
        "source": provenance, "training_sha256": training, "records_sha256": sha256(records),
        "problem_ids": [r["problem_id"] for r in rows], "questions": len(rows), "filtered_rows": 0})
    return {"status": "prepared", "questions": len(rows), "output": str(root), "source": provenance}
