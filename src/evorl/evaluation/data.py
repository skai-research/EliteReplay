"""Full official evaluation splits; no filtering, subsampling or gold in prompts."""
from __future__ import annotations

import importlib.util
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
    spec = importlib.util.spec_from_file_location("eval_bird_preprocessing", ROOT / "data/preprocess_bird.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def taco_rows(c):
    import pyarrow.parquet as pq
    d = c["data"]
    if d["source_file"]:
        source = path(d["source_file"])
        raw = pq.read_table(source).to_pylist()
        provenance = {"source_file": reference(source), "source_sha256": sha256(source), "revision": d["source_revision"]}
    else:
        from huggingface_hub import HfApi, hf_hub_download
        info = HfApi().dataset_info("BAAI/TACO", revision=d["source_revision"])
        revision = info.sha
        names = sorted(s.rfilename for s in info.siblings)
        shards = [n for n in names if n.startswith("ALL/test-") and n.endswith(".parquet")]
        if not shards:
            shards = [n for n in names if n.startswith("data/test-") and n.endswith(".parquet")]
        arrow = [n for n in names if n.startswith("test/") and n.endswith(".arrow")]
        chosen = shards or arrow
        if not chosen:
            raise ValueError("No official TACO test shards found at the selected revision")
        raw, hashes = [], {}
        for name in chosen:
            local = hf_hub_download("BAAI/TACO", name, repo_type="dataset", revision=revision,
                                    cache_dir=ROOT / "external/dataset-cache/hub")
            hashes[name] = sha256(local)
            if name.endswith(".parquet"):
                raw.extend(pq.read_table(local).to_pylist())
            else:
                from datasets import Dataset
                raw.extend(Dataset.from_file(local).to_list())
        provenance = {"repository": "BAAI/TACO", "revision": revision, "split": "test", "shard_sha256": hashes}
    rows = []
    for index, r in enumerate(raw):
        tests = r["input_output"]
        if isinstance(tests, str):
            tests = json.loads(tests)
        if not isinstance(tests, dict) or not isinstance(tests.get("inputs"), list) or not isinstance(tests.get("outputs"), list) or not tests["inputs"] or len(tests["inputs"]) != len(tests["outputs"]):
            raise ValueError(f"Malformed official TACO tests at row {index}; no silent exclusions")
        question = r["question"].strip()
        if not question:
            raise ValueError(f"Empty TACO question at row {index}")
        starter = r.get("starter_code") or ""
        prompt = question + ("\n\nStarter code:\n" + starter if starter else "")
        rows.append({"problem_id": f"taco:test:{index}", "question_key": question_key(question),
            "prompt": [{"role": "user", "content": prompt}], "reward_model": {"ground_truth": json.dumps(tests)},
            "extra_info": {"official_id": str(index), "difficulty": str(r.get("difficulty", "unknown")),
                           "source": str(r.get("source", "unknown"))}})
    return rows, provenance


def audit_training(c, rows):
    import pyarrow.parquet as pq
    root = path(c["data"]["training_data_root"])
    if (root / "payload_manifest.json").exists():
        from evorl.baselines.shared_data import materialize_domain
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
            else:
                # Training IDs already hash normalized original question text.
                pid = r["extra_info"]["problem_id"]
                if not pid.startswith("taco:") or len(pid.split(":", 1)[1]) != 64:
                    raise ValueError("Expected current-suite TACO question identities")
                if pid.split(":", 1)[1] in eval_keys:
                    raise ValueError(f"Official TACO evaluation questions overlap {split}")
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
    if c["domain"] == "livecodebench":
        from .livecodebench import livecodebench_rows
        rows, provenance = livecodebench_rows(c)
    elif c["domain"] == "spider":
        from .spider import spider_rows
        rows, provenance = spider_rows(c)
    else:
        rows, provenance = bird_rows(c) if c["domain"] == "bird" else taco_rows(c)
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
