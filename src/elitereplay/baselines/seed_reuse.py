"""Import verified seeds into a fresh search without rewriting source artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from .config import load, path, offline_dir, offline_fingerprint
from .data import sha256
from elitereplay.EliteReplay.manifests import (
    validate_output_hashes, write_immutable_json, write_immutable_text, write_stage_manifest,
)
from elitereplay.EliteReplay.schema import content_hash


def check_seed_configs(source, target):
    """Only search budget/export/location may differ for this reuse operation."""
    from .runtime import resolved
    source, target = resolved(source), resolved(target)
    for key in ("method", "domain", "seed", "model", "data", "verifier"):
        if source[key] != target[key] or source["method"] != "EliteReplay":
            raise ValueError(f"Seed reuse configuration mismatch: {key}")
    for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
        if source["training"][key] != target["training"][key]:
            raise ValueError(f"Seed reuse generation mismatch: {key}")
    for key in ("image", "rollout_gpus", "max_num_seqs", "rollout_memory_fraction"):
        if source["runtime"][key] != target["runtime"][key]:
            raise ValueError(f"Seed reuse runtime mismatch: {key}")
    excluded = {"source_dir", "seed_source_config", "rounds", "children_per_round",
                "candidates_per_question", "per_question_cap", "export_target", "minimum_examples"}
    if {k: v for k, v in source["offline"].items() if k not in excluded} != {
            k: v for k, v in target["offline"].items() if k not in excluded}:
        raise ValueError("Seed reuse offline generation/taxonomy settings differ")


def import_seed_journal(pipeline, rows, source_dir, expected_sha256):
    """Validate every seed request and import only seeds and an optional taxonomy.

    Mutation plans, candidates and elites are deliberately reconstructed by the
    new search. A completed taxonomy is transferred with a new provenance
    manifest; the original manifest remains untouched.
    """
    source_dir = Path(source_dir)
    if source_dir.resolve() == pipeline.output_dir.resolve():
        raise ValueError("Seed import requires a fresh search directory")
    journal = source_dir / "taxonomy/seed_observations.jsonl"
    if sha256(journal) != expected_sha256:
        raise ValueError("Source seed journal does not match its completion hash")
    old_resolved = json.loads((source_dir / "resolved_config.json").read_text())
    for key in ("seed", "domain", "base_model", "tokenizer", "generator_model", "labeler_model",
                "configured_labeler_model", "aux_search_split", "problem_id_field", "taxonomy_version",
                "verifier_version", "round_trip_gate_version", "round_trip_gate"):
        if old_resolved["resolved"][key] != pipeline.resolved_config["resolved"][key]:
            raise ValueError(f"Source seed pipeline mismatch: {key}")
    indexed = pipeline._index_problems(rows, allow_partial=False)
    observations = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    expected = [pipeline._seed_request(row, pid, index).to_dict()
                for pid, row in indexed.items()
                for index in range(pipeline.config.search.seed_samples_per_problem)]
    if len(observations) != len(expected):
        raise ValueError("Source seed journal is incomplete")
    for observation, request in zip(observations, expected, strict=True):
        if (observation["request"] != request or observation["slot_key"] != request["slot_key"]
                or observation["round_trip_gate_version"] != pipeline.round_trip_gate.version
                or observation["response"]["model"] != pipeline.generator.model_id):
            raise ValueError(f"Source seed request/model mismatch: {request['slot_key']}")
    inputs = {"problem_set_hash": pipeline._problem_set_hash(indexed),
              "aux_search_split": pipeline.config.aux_search_split,
              "round_trip_gate_version": pipeline.round_trip_gate.version, **pipeline._source_hashes()}
    taxonomy_manifest = source_dir / "taxonomy/manifest.json"
    previous = None
    if taxonomy_manifest.exists():
        previous = json.loads(taxonomy_manifest.read_text())
        validate_output_hashes(previous)
        if (previous["stage"] != "induce_taxonomy" or previous["config_hash"] != content_hash(old_resolved)
                or previous["inputs"] != inputs):
            raise ValueError("Source taxonomy provenance does not match the reused seeds/problems")
        if old_resolved["source_config"]["taxonomy"] != pipeline.resolved_config["source_config"]["taxonomy"]:
            raise ValueError("Source taxonomy settings differ")
    # All provenance checks above happen before importing anything.
    pipeline._prepare_output()
    write_immutable_text(pipeline.seed_observations_path, journal.read_text())
    provenance = {"schema_version": "EliteReplay.seed_reuse.v1", "source_dir": str(source_dir.resolve()),
                  "source_resolved_config_sha256": sha256(source_dir / "resolved_config.json"),
                  "seed_journal_sha256": expected_sha256, "seed_proposals": len(observations),
                  "taxonomy_reused": previous is not None}
    if previous is not None:
        taxonomy = source_dir / "taxonomy/taxonomy.json"
        provenance["source_taxonomy_manifest_sha256"] = sha256(taxonomy_manifest)
        write_immutable_text(pipeline.taxonomy_path, taxonomy.read_text())
        write_stage_manifest(pipeline.taxonomy_dir / "manifest.json", stage="induce_taxonomy",
            config_hash=pipeline.config_hash, inputs=inputs,
            outputs={"resolved_config": pipeline.output_dir / "resolved_config.json",
                     "seed_observations": pipeline.seed_observations_path, "taxonomy": pipeline.taxonomy_path},
            models=previous["models"], counts=previous["counts"], provenance=provenance)
    write_immutable_json(pipeline.output_dir / "seed_reuse.json", provenance)
    return provenance


def reuse_seeds(c, pipeline, rows):
    from .runtime import resolved
    source_config = path(c["offline"]["seed_source_config"])
    source = resolved(load(source_config))
    check_seed_configs(source, c)
    root = offline_dir(source)
    complete = json.loads((root / "seeds_complete.json").read_text())
    contract = json.loads((root / "contract.json").read_text())
    expected = {"config_sha256": offline_fingerprint(source),
                "train_sha256": sha256(path(c["data"]["root"]) / "train.parquet"),
                "aux_sha256": sha256(path(c["data"]["root"]) / "aux_search.parquet")}
    if contract != expected or any(complete.get(k) != v for k, v in expected.items()):
        raise ValueError("Seed source completion/config/data contract mismatch")
    if (complete["problems"] != c["data"]["archive_questions"]
            or complete["seed_proposals"] != c["data"]["archive_questions"] * c["offline"]["seed_samples"]):
        raise ValueError("Seed source completion has the wrong number of proposals")
    result = import_seed_journal(pipeline, rows, root / "map_elites", complete["seed_journal_sha256"])
    write_immutable_json(pipeline.output_dir / "seed_source_config.json", {
        "path": str(source_config), "sha256": sha256(source_config),
        "completion_sha256": sha256(root / "seeds_complete.json")})
    print(json.dumps({"reused_seeds": result}), flush=True)
    return result
