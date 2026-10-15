"""Restore exact shared parquet bytes from ordinary Git payload parts."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile

from .config import fingerprint
from .data import proportional_quotas, sha256, validate_splits


def materialize_domain(directory):
    """Restore only missing payloads; never overwrite modified local datasets."""
    directory = Path(directory)
    lock = directory.parent / "release.json"
    if lock.exists():
        expected = json.loads(lock.read_text())["domains"][directory.name]
        if sha256(directory / "split_manifest.json") != expected["split_manifest_sha256"]:
            raise ValueError("Shared data release manifest has changed")
        if expected.get("payload_manifest_sha256") and sha256(directory / "payload_manifest.json") != expected["payload_manifest_sha256"]:
            raise ValueError("Shared data payload manifest has changed")
    metadata = directory / "payload_manifest.json"
    if not metadata.exists():
        return []
    payloads = json.loads(metadata.read_text())["files"]
    split_manifest = json.loads((directory / "split_manifest.json").read_text())
    restored = []
    for name, payload in payloads.items():
        if name not in {"train.parquet", "aux_search.parquet", "validation.parquet"}:
            raise ValueError("Invalid shared parquet filename")
        if payload["sha256"] != split_manifest["parquet_sha256"][name.removesuffix(".parquet")]:
            raise ValueError("Payload and split hashes disagree")
        destination = directory / name
        if destination.exists():
            if sha256(destination) != payload["sha256"]:
                raise ValueError(f"Local shared dataset was modified: {destination}")
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{name}.", delete=False) as output:
                temporary = Path(output.name)
                for part in payload["parts"]:
                    if Path(part["name"]).name != part["name"]:
                        raise ValueError("Invalid payload part filename")
                    source = directory / part["name"]
                    if source.stat().st_size != part["bytes"] or sha256(source) != part["sha256"]:
                        raise ValueError(f"Shared payload part changed: {source}")
                    with source.open("rb") as handle:
                        shutil.copyfileobj(handle, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != payload["bytes"] or sha256(temporary) != payload["sha256"]:
                raise ValueError("Restored parquet hash/size mismatch")
            if destination.exists():
                if sha256(destination) != payload["sha256"]:
                    raise ValueError(f"Another process wrote different data: {destination}")
            else:
                temporary.replace(destination)
            restored.append(str(destination))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return restored


def verify_domain(directory):
    """Verify bytes, ordered identities, contents, quotas, and split boundaries."""
    import pyarrow.parquet as pq
    directory = Path(directory)
    materialize_domain(directory)
    manifest = json.loads((directory / "split_manifest.json").read_text())
    splits = {}
    for name in ("train", "aux_search", "validation"):
        source = directory / f"{name}.parquet"
        if sha256(source) != manifest["parquet_sha256"][name]:
            raise ValueError(f"Parquet hash mismatch: {source}")
        records = pq.read_table(source).to_pylist()
        spec = manifest["splits"][name]
        if len(records) != spec["rows"] or [r["extra_info"]["problem_id"] for r in records] != spec["problem_ids"]:
            raise ValueError(f"Ordered identities/count mismatch: {name}")
        if [fingerprint(r) for r in records] != spec["record_sha256"]:
            raise ValueError(f"Record content mismatch: {name}")
        counts = dict(Counter(r["extra_info"]["difficulty"] for r in records))
        if counts != spec["difficulty_counts"]:
            raise ValueError(f"Difficulty counts mismatch: {name}")
        if manifest["domain"] == "taco" and counts != proportional_quotas(len(records), manifest["source"]["raw_difficulty_counts"]):
            raise ValueError(f"Original TACO difficulty proportions not preserved: {name}")
        splits[name] = records
    validate_splits(splits)
    train = {r["extra_info"]["problem_id"]: r for r in splits["train"]}
    if any(r != train[r["extra_info"]["problem_id"]] for r in splits["aux_search"]):
        raise ValueError("Archive questions are not identical training records")
    if manifest["domain"] == "bird":
        train_db = {r["extra_info"]["db_id"] for r in splits["train"]}
        val_db = {r["extra_info"]["db_id"] for r in splits["validation"]}
        if train_db & val_db:
            raise ValueError("BIRD internal validation databases overlap training")
    return {"domain": manifest["domain"], "verified": True,
            "rows": {name: len(rows) for name, rows in splits.items()},
            "difficulty_counts": {name: manifest["splits"][name]["difficulty_counts"] for name in splits},
            "manifest_sha256": sha256(directory / "split_manifest.json")}
