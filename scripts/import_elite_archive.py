#!/usr/bin/env python3
"""Verify and stage the packaged BIRD elite archive for the local run config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evorl.baselines.config import load, offline_dir, offline_fingerprint, path, fingerprint
from evorl.baselines.data import sha256
from evorl.baselines.replay import Replay
from evorl.baselines.runtime import resolved

ARCHIVE_SHA256 = "710cdbcd7a52d00497fc51cb0d077ad51c5de4ea7c281587d94e2b9d70357966"
TAXONOMY_SHA256 = "128a7137b5cad824120151988c493111102eae7f5b392bdaaae062454f731952"
MAP_ELITES_SHA256 = "06cbb6d25a07c7accc2474aeb4d25fda9e382b087296ee85aa294ceaa20859c2"
SOURCE_CONFIG_SHA256 = "4eb16cfc9318fd23b7810e00064919aa9900c8691a7c92e9828bde8d01e395cf"


def stage(config_file: Path) -> dict:
    c = resolved(load(config_file))
    if c["method"] != "EliteReplay" or c["domain"] != "bird":
        raise ValueError("The packaged archive is for EliteReplay on BIRD only")
    source = ROOT / "elite_archive"
    archive_file = source / "archive.jsonl"
    taxonomy_file = source / "taxonomy.json"
    full_archive_file = source / "map_elites_archive.json"
    original = json.loads((source / "source_complete.json").read_text())
    if (sha256(archive_file) != ARCHIVE_SHA256 or original["archive_sha256"] != ARCHIVE_SHA256
            or sha256(taxonomy_file) != TAXONOMY_SHA256
            or original["taxonomy_sha256"] != TAXONOMY_SHA256
            or sha256(full_archive_file) != MAP_ELITES_SHA256
            or original["map_elites_sha256"] != MAP_ELITES_SHA256
            or original["anonymized_copy"] is not True
            or original["config_sha256"] != SOURCE_CONFIG_SHA256
            or original["examples"] != 1500 or original["integrity_pass"] is not True):
        raise ValueError("Packaged archive or its provenance changed")

    data = path(c["data"]["root"])
    manifest = json.loads((data / "split_manifest.json").read_text())
    if manifest["domain"] != "bird" or manifest["archive_is_training_subset"] is not True:
        raise ValueError("Expected the paper's BIRD training/archive split")
    # Compare ordered semantic records, not writer-version-dependent Parquet bytes.
    import pyarrow.parquet as pq
    identity = json.loads((ROOT / "data/bird_split_identity.json").read_text())
    split_hashes = {}
    for split in ("train", "aux_search", "validation"):
        file = data / f"{split}.parquet"
        records = pq.read_table(file).to_pylist()
        expected = identity["splits"][split]
        if ([r["extra_info"]["problem_id"] for r in records] != expected["problem_ids"]
                or [fingerprint(r) for r in records] != expected["record_sha256"]):
            raise ValueError(f"{split} records differ from the frozen paper split")
        split_hashes[split] = sha256(file)
        if split_hashes[split] != manifest["parquet_sha256"][split]:
            raise ValueError(f"{split} manifest does not match its Parquet file")
    full_archive = json.loads(full_archive_file.read_text())
    if (full_archive["domain"] != "bird" or len(full_archive["problems"]) != 780
            or sum(len(cells) for cells in full_archive["problems"].values()) != 2708):
        raise ValueError("Full MAP-Elites archive dimensions changed")
    import pyarrow.parquet as pq
    allowed = {str(row["extra_info"]["problem_id"])
               for row in pq.read_table(data / "aux_search.parquet", columns=["extra_info"]).to_pylist()}
    rows = [json.loads(line) for line in archive_file.read_text().splitlines() if line.strip()]
    if len(rows) != 1500 or len(Replay(rows, method="qd", seed=c["seed"], sampling="mode_cycles").groups) != 780:
        raise ValueError("Archive row count or represented-question count changed")
    if not {str(row["metadata"]["problem_id"]) for row in rows} <= allowed:
        raise ValueError("Archive includes a question outside the matched training subset")
    taxonomy_hash = json.loads(taxonomy_file.read_text())["taxonomy_hash"]
    if any(row["metadata"]["taxonomy_hash"] != taxonomy_hash for row in rows):
        raise ValueError("Archive taxonomy identity changed")

    target = offline_dir(c)
    complete = {"archive_sha256": ARCHIVE_SHA256,
                "audit_summary_sha256": original["audit_summary_sha256"],
                "aux_sha256": split_hashes["aux_search"],
                "config_sha256": offline_fingerprint(c),
                "examples": 1500, "integrity_pass": True,
                "train_sha256": split_hashes["train"],
                "source_config_sha256": original["config_sha256"],
                "source_archive_sha256": original["source_archive_sha256"]}
    destination = target / "archive.jsonl"
    receipt = target / "complete.json"
    taxonomy_target = target / "map_elites/taxonomy/taxonomy.json"
    full_archive_target = target / "map_elites/archive/archive.json"
    if receipt.exists():
        if (json.loads(receipt.read_text()) != complete or sha256(destination) != ARCHIVE_SHA256
                or sha256(taxonomy_target) != TAXONOMY_SHA256
                or sha256(full_archive_target) != MAP_ELITES_SHA256):
            raise FileExistsError(f"Different offline archive already exists at {target}")
        return {"status": "already_staged", "path": str(target), "examples": 1500}
    if any(p.exists() for p in (destination, taxonomy_target, full_archive_target)):
        raise FileExistsError(f"Incomplete or different archive already exists at {target}")
    taxonomy_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive_file, destination)
    shutil.copyfile(taxonomy_file, taxonomy_target)
    full_archive_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(full_archive_file, full_archive_target)
    receipt.write_text(json.dumps(complete, sort_keys=True, indent=2) + "\n")
    return {"status": "staged", "path": str(target), "examples": 1500,
            "archive_sha256": ARCHIVE_SHA256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/elitereplay.yaml")
    args = parser.parse_args()
    print(json.dumps(stage(args.config), indent=2))


if __name__ == "__main__":
    main()
