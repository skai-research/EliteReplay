"""Resumable offline taxonomy, QD search, archive freeze, and SFT export."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .archive import (
    AppendOnlyJournal,
    MapElitesArchive,
    choose_parent,
    choose_target_strategy,
    frozen_archive_document,
    load_frozen_archive,
    select_problem_balanced_elites,
)
from .domain import DomainAdapter, ParseResult, VerificationResult, item_value
from .manifests import (
    runtime_provenance,
    sha256_file,
    write_immutable_json,
    write_immutable_jsonl,
    write_immutable_text,
    write_stage_manifest,
    validate_output_hashes,
)
from .providers import (
    CachedSolutionGenerator,
    CachedStrategyLabeler,
    JsonRequestCache,
    MetadataBatchExhaustedError,
    SolutionGenerator,
    StrategyLabeler,
)
from .schema import (
    CandidateRecord,
    GenerationRequest,
    GenerationResponse,
    LabelDecision,
    PipelineConfig,
    SeedObservation,
    StrategyLabel,
    Taxonomy,
    canonical_json,
    content_hash,
    deterministic_seed,
    stable_candidate_id,
    validate_taxonomy_size,
)
from .tokenization import (
    TokenizerRoundTripGate,
    build_round_trip_gate,
)


LOGGER = logging.getLogger(__name__)


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _plain(value: Any) -> Any:
    """Normalise dataframe/Arrow values before immutable input hashing."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "as_py"):
        return _plain(value.as_py())
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _plain(value.item())
        except (TypeError, ValueError):
            pass
    return value


class EliteReplayPipeline:
    """Single-writer, fully resumable offline portion of EliteReplay."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        adapter: DomainAdapter,
        generator: SolutionGenerator,
        labeler: StrategyLabeler,
        round_trip_gate: TokenizerRoundTripGate | None = None,
        resume: bool = True,
        project_root: str | Path | None = None,
        trinity_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.output_dir = Path(config.output_dir)
        self.resume = resume
        cache = JsonRequestCache(self.output_dir / "request_cache")
        self.generator = (
            generator
            if isinstance(generator, CachedSolutionGenerator)
            else CachedSolutionGenerator(generator, cache)
        )
        self.labeler = (
            labeler
            if isinstance(labeler, CachedStrategyLabeler)
            else CachedStrategyLabeler(labeler, cache)
        )
        self.round_trip_gate = round_trip_gate or build_round_trip_gate(config)
        inferred_root = Path(__file__).resolve().parents[3]
        self.project_root = Path(project_root or inferred_root)
        self.trinity_root = Path(trinity_root or self.project_root / "Trinity-RFT")

        self.taxonomy_dir = self.output_dir / "taxonomy"
        self.archive_dir = self.output_dir / "archive"
        self.sft_dir = self.output_dir / "sft"
        self.taxonomy_path = self.taxonomy_dir / "taxonomy.json"
        self.seed_observations_path = self.taxonomy_dir / "seed_observations.jsonl"
        self.candidates_path = self.archive_dir / "candidates.jsonl"
        self.feasible_path = self.archive_dir / "feasible_strategies.jsonl"
        self.plans_path = self.archive_dir / "generation_plans.jsonl"
        self.archive_path = self.archive_dir / "archive.json"
        self.archive_hash_path = self.archive_dir / "archive.sha256"

    @property
    def resolved_config(self) -> dict[str, Any]:
        return {
            "schema_version": "EliteReplay.resolved_config.v1",
            "source_config": _plain(self.config.raw_config),
            "resolved": {
                "experiment_name": self.config.experiment_name,
                "seed": self.config.seed,
                "domain": self.config.domain,
                "base_model": self.config.base_model,
                "tokenizer": self.config.tokenizer,
                "configured_labeler_model": self.config.labeler_model,
                "generator_model": self.generator.model_id,
                "labeler_model": self.labeler.model_id,
                "aux_search_split": self.config.aux_search_split,
                "problem_id_field": self.config.problem_id_field,
                "output_dir": str(self.output_dir),
                "sft_publish_path": self.config.sft_publish_path,
                "taxonomy_version": self.config.taxonomy_version,
                "search": dataclasses.asdict(self.config.search),
                "verifier_version": self.adapter.verifier_version,
                "round_trip_gate": dict(self.round_trip_gate.descriptor),
                "round_trip_gate_version": self.round_trip_gate.version,
            },
        }

    @property
    def config_hash(self) -> str:
        return content_hash(self.resolved_config)

    def plan(
        self,
        problems: Sequence[Mapping[str, Any]],
        *,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        indexed = self._index_problems(problems, allow_partial=allow_partial)
        count = len(indexed)
        search = self.config.search
        return {
            "experiment": self.config.experiment_name,
            "domain": self.config.domain,
            "problems": count,
            "seed_generations": count * search.seed_samples_per_problem,
            "mutation_generations": (
                count * search.rounds * search.children_per_round_per_problem
            ),
            "total_generations": count * search.generations_per_problem,
            "taxonomy_description_cap": min(
                search.taxonomy_sample_cap,
                count * search.seed_samples_per_problem,
            ),
            "target_taxonomy_size": search.taxonomy_size,
            "target_archive_elites_per_problem": search.target_elites_per_problem,
            "maximum_sft_examples": min(
                search.export_max_examples,
                count * search.target_elites_per_problem,
            ),
            "generator_model": self.generator.model_id,
            "labeler_model": self.labeler.model_id,
            "max_response_tokens": search.decoding.max_tokens,
            "round_trip_gate_version": self.round_trip_gate.version,
            "output_dir": str(self.output_dir),
            "config_hash": self.config_hash,
        }

    def collect_seeds(self, problems: Sequence[Mapping[str, Any]], *, allow_partial: bool = False) -> dict[str, int]:
        """Generate and verify seeds without calling the external labeler.

        Uses the same journal as induce_taxonomy, so a later full preparation
        reuses every completed seed slot without spending another proposal.
        """
        indexed = self._index_problems(problems, allow_partial=allow_partial)
        self._prepare_output()
        observations = self._ensure_seed_observations(indexed)
        return {"problems": len(indexed), "seed_proposals": len(observations),
                "fully_correct_seeds": sum(o.fully_correct for o in observations)}

    def induce_taxonomy(
        self,
        problems: Sequence[Mapping[str, Any]],
        *,
        allow_partial: bool = False,
    ) -> Taxonomy:
        indexed = self._index_problems(problems, allow_partial=allow_partial)
        self._prepare_output()
        if (self.taxonomy_dir / "manifest.json").exists():
            self._require_stage_manifest(self.taxonomy_dir / "manifest.json", "induce_taxonomy")
            manifest = json.loads((self.taxonomy_dir / "manifest.json").read_text())
            if manifest["inputs"]["problem_set_hash"] != self._problem_set_hash(indexed):
                raise ValueError("Completed taxonomy belongs to different problems")
            if any(manifest["inputs"].get(k) != v for k, v in self._source_hashes().items()):
                raise ValueError("Completed taxonomy source data changed")
            return self._load_taxonomy()
        if self.config.domain == "optmath":
            from .optmath_strategies import frozen_taxonomy
            taxonomy = frozen_taxonomy(self.labeler.model_id)
            if taxonomy.version != self.config.taxonomy_version:
                raise ValueError("OptMATH taxonomy version disagrees with frozen definitions")
            validate_taxonomy_size(taxonomy, self.config.search.taxonomy_size)
            # Freeze descriptors before the first production generation request.
            write_immutable_json(self.taxonomy_path, taxonomy.to_dict())
            observations = self._ensure_seed_observations(indexed)
            correct = [o for o in observations if o.fully_correct]
            descriptions = ()
        else:
            observations = self._ensure_seed_observations(indexed)
            correct = [
                observation
                for observation in observations
                if observation.parse_accepted
                and observation.roundtrip_accepted
                and observation.executable
                and observation.fully_correct
                and observation.canonical_response is not None
            ]
            if not correct:
                raise RuntimeError("taxonomy induction found no fully correct seed solutions")
            ranked = sorted(
                correct,
                key=lambda item: content_hash(
                    {
                        "seed": self.config.seed,
                        "namespace": "taxonomy_sample",
                        "slot_key": item.slot_key,
                    }
                ),
            )[: self.config.search.taxonomy_sample_cap]

            calls: list[dict[str, Any]] = []
            for observation in ranked:
                item = indexed[observation.request.problem_id]
                calls.append(
                    {
                        "problem_id": observation.request.problem_id,
                        "labeler_input": self.adapter.format_labeler_input(
                            item, observation.canonical_response or ""
                        ),
                        "seed": deterministic_seed(
                            self.config.seed, "describe", observation.slot_key
                        ),
                    }
                )
            description_results = self._label_batch("describe", calls)
            descriptions = tuple(
                description
                for description in description_results
                if description is not None
            )
            if any(not isinstance(description, str) for description in descriptions):
                raise TypeError("labeler describe must return strings")
            if len(descriptions) < self.config.search.taxonomy_size:
                raise RuntimeError(
                    "taxonomy induction has too few valid strategy descriptions "
                    f"after metadata filtering: {len(descriptions)}/"
                    f"{self.config.search.taxonomy_size}"
                )
            labels = tuple(
                self.labeler.consolidate(
                    domain=self.config.domain,
                    descriptions=descriptions,
                    size=self.config.search.taxonomy_size,
                    seed=deterministic_seed(self.config.seed, "consolidate_taxonomy"),
                )
            )
            taxonomy = Taxonomy(
                domain=self.config.domain,
                version=self.config.taxonomy_version,
                labeler_model=self.labeler.model_id,
                labels=labels,
            )
        validate_taxonomy_size(taxonomy, self.config.search.taxonomy_size)
        write_immutable_json(self.taxonomy_path, taxonomy.to_dict())
        problem_hash = self._problem_set_hash(indexed)
        write_stage_manifest(
            self.taxonomy_dir / "manifest.json",
            stage="induce_taxonomy",
            config_hash=self.config_hash,
            inputs={
                "problem_set_hash": problem_hash,
                "aux_search_split": self.config.aux_search_split,
                "round_trip_gate_version": self.round_trip_gate.version,
                **self._source_hashes(),
            },
            outputs={
                "resolved_config": self.output_dir / "resolved_config.json",
                "seed_observations": self.seed_observations_path,
                "taxonomy": self.taxonomy_path,
            },
            models={
                "base_model": self.generator.model_id,
                "labeler_model": self.labeler.model_id,
                "tokenizer": self.config.tokenizer,
            },
            counts={
                "problems": len(indexed),
                "seed_attempts": len(observations),
                "correct_seed_solutions": len(correct),
                "taxonomy_examples": len(descriptions),
                "taxonomy_labels": len(taxonomy.labels),
            },
            provenance=runtime_provenance(
                project_root=self.project_root,
                trinity_root=self.trinity_root,
            ),
        )
        return taxonomy

    def build_archive(
        self,
        problems: Sequence[Mapping[str, Any]],
        *,
        allow_partial: bool = False,
    ) -> Mapping[str, Any]:
        indexed = self._index_problems(problems, allow_partial=allow_partial)
        self._prepare_output()
        self._require_stage_manifest(
            self.taxonomy_dir / "manifest.json", "induce_taxonomy"
        )
        taxonomy = self._load_taxonomy()
        observations = self._ensure_seed_observations(indexed)
        observation_by_slot = {item.slot_key: item for item in observations}

        candidate_journal = AppendOnlyJournal(
            self.candidates_path, key_field="slot_key"
        )
        expected_slots = self._expected_candidate_slots(tuple(indexed))
        actual_slots = [str(record["slot_key"]) for record in candidate_journal]
        if actual_slots != expected_slots[: len(actual_slots)]:
            raise ValueError(
                "candidate journal is not a prefix of the deterministic search schedule"
            )
        archive = MapElitesArchive()
        archive.replay(CandidateRecord.from_dict(row) for row in candidate_journal)

        seed_slots = expected_slots[
            : len(indexed) * self.config.search.seed_samples_per_problem
        ]
        missing_seed_slots = seed_slots[len(actual_slots) :]
        if missing_seed_slots:
            entries = [observation_by_slot[slot] for slot in missing_seed_slots]
            decisions = self._classify_observations(entries, indexed, taxonomy)
            for observation, (decision, label_error) in zip(
                entries, decisions, strict=True
            ):
                candidate = self._candidate_from_observation(
                    observation,
                    taxonomy=taxonomy,
                    decision=decision,
                    label_error=label_error,
                )
                accepted = archive.insert(candidate)
                candidate_journal.append(accepted.to_dict())

        feasible = self._ensure_feasible_plans(indexed, taxonomy, archive)
        plan_journal = AppendOnlyJournal(self.plans_path, key_field="slot_key")
        attempts = Counter((entry["request"]["problem_id"], entry["request"]["requested_strategy"]["id"])
                           for entry in plan_journal if entry["request"].get("requested_strategy") is not None)

        # One child index at a time preserves within-problem QD adaptation while
        # batching the independent requests for every problem in that wave.
        for round_index in range(self.config.search.rounds):
            for child_index in range(
                self.config.search.children_per_round_per_problem
            ):
                wave_slots = [
                    self._mutation_slot(problem_id, round_index, child_index)
                    for problem_id in indexed
                ]
                for problem_id, slot_key in zip(indexed, wave_slots, strict=True):
                    if plan_journal.get(slot_key) is not None:
                        continue
                    available = feasible[problem_id]
                    if self.config.domain == "optmath" and not available:
                        available = tuple(sorted({c.primary_strategy for c in archive.elites_for(problem_id)
                                                  if c.primary_strategy is not None}))
                    if self.config.domain == "optmath" and not available:
                        request = self._mutation_request(
                            item=indexed[problem_id], problem_id=problem_id,
                            round_index=round_index, child_index=child_index,
                            target=None, parent=None)
                        request, admission = self._admit_mutation_request(request, indexed[problem_id])
                        plan_journal.append({"slot_key": slot_key,
                                             "request_hash": self._request_hash(request),
                                             "request": request.to_dict(),
                                             "generation_kind": "untargeted_recovery",
                                             "reason": "no_evidenced_feasible_strategy",
                                             **admission})
                        continue
                    target_id = choose_target_strategy(
                        archive=archive,
                        problem_id=problem_id,
                        feasible=available,
                        seed=self.config.seed,
                        slot_key=slot_key,
                        attempt_counts=({label: attempts[(problem_id, label)] for label in available}
                                        if self.config.search.attempt_aware else None),
                    )
                    target = taxonomy.require_label(target_id)
                    parent = choose_parent(
                        archive=archive,
                        problem_id=problem_id,
                        target_strategy=target_id,
                        seed=self.config.seed,
                        slot_key=slot_key,
                    )
                    request = self._mutation_request(
                        item=indexed[problem_id],
                        problem_id=problem_id,
                        round_index=round_index,
                        child_index=child_index,
                        target=target,
                        parent=parent,
                    )
                    request, admission = self._admit_mutation_request(request, indexed[problem_id])
                    plan_journal.append(
                        {
                            **admission,
                            "slot_key": slot_key,
                            "request_hash": self._request_hash(request),
                            "request": request.to_dict(),
                        }
                    )
                    if request.requested_strategy is not None:
                        attempts[(problem_id, request.requested_strategy.id)] += 1

                pending_requests = [
                    GenerationRequest.from_dict(plan_journal.get(slot)["request"])
                    for slot in wave_slots
                    if candidate_journal.get(slot) is None
                ]
                if not pending_requests:
                    continue
                responses = self._generate_batch(pending_requests)
                pending_observations = self._evaluate_batch(
                    pending_requests, responses, indexed
                )
                decisions = self._classify_observations(
                    pending_observations, indexed, taxonomy
                )
                for observation, (decision, label_error) in zip(
                    pending_observations, decisions, strict=True
                ):
                    candidate = self._candidate_from_observation(
                        observation,
                        taxonomy=taxonomy,
                        decision=decision,
                        label_error=label_error,
                    )
                    accepted = archive.insert(candidate)
                    candidate_journal.append(accepted.to_dict())

        final_records = list(candidate_journal)
        if len(final_records) != len(expected_slots):
            raise RuntimeError(
                f"archive incomplete: {len(final_records)}/{len(expected_slots)} slots"
            )
        if not self.feasible_path.exists():
            write_immutable_text(self.feasible_path, "")
        if not self.plans_path.exists():
            write_immutable_text(self.plans_path, "")
        candidate_log_hash = sha256_file(self.candidates_path)
        archive_document = frozen_archive_document(
            domain=self.config.domain,
            taxonomy=taxonomy,
            archive=archive,
            candidate_log_sha256=candidate_log_hash,
            config_hash=self.config_hash,
        )
        write_immutable_json(self.archive_path, archive_document)
        write_immutable_text(
            self.archive_hash_path,
            f"{archive_document['archive_hash']}  {self.archive_path.name}\n",
        )
        total_elites = sum(
            len(archive.elites_for(problem_id)) for problem_id in archive.problem_ids
        )
        write_stage_manifest(
            self.archive_dir / "manifest.json",
            stage="build_archive",
            config_hash=self.config_hash,
            inputs={
                "problem_set_hash": self._problem_set_hash(indexed),
                "taxonomy_hash": taxonomy.hash,
                "seed_observations_sha256": sha256_file(
                    self.seed_observations_path
                ),
                "round_trip_gate_version": self.round_trip_gate.version,
                **self._source_hashes(),
            },
            outputs={
                "candidate_log": self.candidates_path,
                "feasible_strategies": self.feasible_path,
                "generation_plans": self.plans_path,
                "archive": self.archive_path,
                "archive_hash": self.archive_hash_path,
            },
            models={
                "base_model": self.generator.model_id,
                "labeler_model": self.labeler.model_id,
                "tokenizer": self.config.tokenizer,
            },
            counts={
                "problems": len(indexed),
                "candidate_attempts": len(final_records),
                "targeted_mutation_requests": sum(r["method"] == "qd_mutation" and r.get("requested_strategy") is not None for r in final_records),
                "untargeted_recovery_requests": sum(r["method"] == "qd_mutation" and r.get("requested_strategy") is None for r in final_records),
                "fully_correct_candidates": sum(
                    bool(record.get("fully_correct")) for record in final_records
                ),
                "occupied_primary_niches": sum(len({e.primary_strategy for e in archive.elites_for(pid)})
                                               for pid in archive.problem_ids),
                "occupied_descriptor_cells": total_elites,
            },
            provenance=runtime_provenance(
                project_root=self.project_root,
                trinity_root=self.trinity_root,
            ),
        )
        return archive_document

    def export_sft(
        self,
        problems: Sequence[Mapping[str, Any]],
        *,
        allow_partial: bool = False,
    ) -> Mapping[str, Any]:
        indexed = self._index_problems(problems, allow_partial=allow_partial)
        self._prepare_output()
        self._require_stage_manifest(
            self.archive_dir / "manifest.json", "build_archive"
        )
        taxonomy = self._load_taxonomy()
        archive, archive_document = load_frozen_archive(
            self.archive_path, expected_taxonomy_hash=taxonomy.hash
        )
        if archive_document.get("config_hash") != self.config_hash:
            raise ValueError("frozen archive was built with a different config")
        export_settings = self.config.raw_config.get("export", {})
        deduplication = export_settings.get("deduplication", "none")
        deduplication_key = None
        if deduplication == "canonical_ast":
            from .archive_diversity import canonical_artifact_key

            deduplication_key = lambda candidate: canonical_artifact_key(
                self.config.domain, candidate.canonical_response or ""
            )
        elif deduplication != "none":
            raise ValueError(f"unsupported export deduplication: {deduplication}")
        selected = select_problem_balanced_elites(
            archive,
            maximum=self.config.search.export_max_examples,
            maximum_per_problem=self.config.search.target_elites_per_problem,
            seed=self.config.seed,
            deduplication_key=deduplication_key,
        )
        if not selected:
            raise RuntimeError("frozen archive contains no exportable elites")

        def sanitize_and_verify(
            candidate: CandidateRecord,
        ) -> tuple[str, Mapping[str, Any], VerificationResult]:
            item = indexed.get(candidate.problem_id)
            if item is None:
                raise ValueError(
                    f"archive problem {candidate.problem_id!r} is absent from input split"
                )
            canonical = candidate.canonical_response
            if canonical is None:  # guarded by frozen archive validation
                raise ValueError("archive elite has no canonical response")
            sanitized = self.adapter.sanitize_for_sft(canonical)
            if sanitized != canonical:
                raise ValueError(
                    "SFT sanitizer changed the verifier artifact; refusing export"
                )
            verification = self.adapter.verify(item, sanitized)
            if not verification.executable or not verification.fully_correct:
                raise ValueError(
                    f"archive elite {candidate.candidate_id} failed export re-verification"
                )
            return sanitized, item, verification

        with ThreadPoolExecutor(
            max_workers=min(self.config.search.verifier_workers, len(selected))
        ) as executor:
            verified = tuple(executor.map(sanitize_and_verify, selected))

        dataset_rows: list[dict[str, Any]] = []
        metadata_rows: list[dict[str, Any]] = []
        for candidate, (sanitized, item, verification) in zip(selected, verified, strict=True):
            messages = [dict(message) for message in self.adapter.format_prompt(item)]
            if not messages or messages[-1].get("role") == "assistant":
                raise ValueError("domain prompt must end before the target assistant turn")
            messages.append({"role": "assistant", "content": sanitized})
            metadata = {
                "candidate_id": candidate.candidate_id,
                "problem_id": candidate.problem_id,
                "primary_strategy": candidate.primary_strategy,
                "secondary_strategy": None,
                "structural_subtype": candidate.structural_subtype,
                "mode_id": candidate.niche,
                "verifier_score": candidate.verifier_score,
                "fully_correct": True,
                "archive_method": "qd",
                "sample_source": "frozen_sft",
                "response_tokens": candidate.output_tokens,
                "taxonomy_hash": taxonomy.hash,
                "archive_hash": archive_document["archive_hash"],
            }
            if self.config.domain == "optmath":
                metadata.update(label_confidence=candidate.label_confidence,
                    label_justification=candidate.label_justification,
                    solver_verified=verification.fully_correct,
                    model_structure_hash=verification.metadata.get("model_structure_hash"),
                    verifier_receipt=dict(verification.metadata))
            dataset_rows.append({"messages": messages, "metadata": metadata})
            metadata_rows.append(metadata)

        if self.config.domain == "optmath":
            from .optmath_strategies import audit_strategy_evidence
            evidence = audit_strategy_evidence(metadata_rows)
            write_immutable_json(self.sft_dir / "strategy_evidence.json", evidence)
            if not evidence["evidence_checks_passed"]:
                raise ValueError("OptMATH formulation evidence audit failed; inspect strategy_evidence.json")

        dataset_path = self.sft_dir / "train.jsonl"
        metadata_path = self.sft_dir / "metadata.jsonl"
        write_immutable_jsonl(dataset_path, dataset_rows)
        write_immutable_jsonl(metadata_path, metadata_rows)
        published_path = (
            None
            if self.config.sft_publish_path is None
            else Path(self.config.sft_publish_path)
        )
        if published_path is not None:
            write_immutable_text(
                published_path, dataset_path.read_text(encoding="utf-8")
            )
        export_hash = content_hash(
            {
                "dataset_sha256": sha256_file(dataset_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
        export_hash_path = self.sft_dir / "export.sha256"
        write_immutable_text(export_hash_path, f"{export_hash}\n")
        output_files: dict[str, Path] = {
            "messages_dataset": dataset_path,
            "metadata": metadata_path,
            "export_hash": export_hash_path,
        }
        if published_path is not None:
            output_files["published_messages_dataset"] = published_path
        manifest = write_stage_manifest(
            self.sft_dir / "manifest.json",
            stage="export_sft",
            config_hash=self.config_hash,
            inputs={
                "problem_set_hash": self._problem_set_hash(indexed),
                "taxonomy_hash": taxonomy.hash,
                "archive_hash": archive_document["archive_hash"],
                "archive_file_sha256": sha256_file(self.archive_path),
                "round_trip_gate_version": self.round_trip_gate.version,
                **self._source_hashes(),
            },
            outputs=output_files,
            models={
                "base_model": self.generator.model_id,
                "labeler_model": self.labeler.model_id,
                "tokenizer": self.config.tokenizer,
            },
            counts={
                "archive_elites": sum(
                    len(archive.elites_for(problem_id))
                    for problem_id in archive.problem_ids
                ),
                "exported_examples": len(dataset_rows),
                "exported_problems": len(
                    {row["problem_id"] for row in metadata_rows}
                ),
            },
            provenance=runtime_provenance(
                project_root=self.project_root,
                trinity_root=self.trinity_root,
            ),
        )
        return manifest

    def _prepare_output(self) -> None:
        resolved_path = self.output_dir / "resolved_config.json"
        if not self.resume and self.output_dir.exists():
            populated = any(self.output_dir.iterdir())
            if populated:
                raise FileExistsError(
                    f"output directory is not empty and resume is disabled: {self.output_dir}"
                )
        write_immutable_json(resolved_path, self.resolved_config)

    def _index_problems(
        self,
        problems: Sequence[Mapping[str, Any]],
        *,
        allow_partial: bool,
    ) -> dict[str, Mapping[str, Any]]:
        expected = self.config.search.aux_prompt_count
        if not allow_partial and len(problems) != expected:
            raise ValueError(
                f"auxiliary split has {len(problems)} problems; expected exactly {expected}"
            )
        if not problems:
            raise ValueError("auxiliary split is empty")
        self._validate_split_manifest(len(problems), allow_partial=allow_partial)
        indexed: dict[str, Mapping[str, Any]] = {}
        for position, item in enumerate(problems):
            problem_id = self._problem_id(item, position)
            if problem_id in indexed:
                raise ValueError(f"duplicate problem id: {problem_id!r}")
            indexed[problem_id] = item
        return indexed

    def _source_hashes(self) -> dict[str, Any]:
        data = self.config.raw_config.get("data", {})
        result: dict[str, Any] = {}
        source = Path(self.config.aux_search_split)
        if source.is_file():
            result["aux_search_sha256"] = sha256_file(source)
        manifest_value = data.get("split_manifest") if isinstance(data, Mapping) else None
        if manifest_value:
            manifest_path = Path(str(manifest_value))
            if not manifest_path.is_file():
                raise FileNotFoundError(f"split manifest is missing: {manifest_path}")
            result["split_manifest_sha256"] = sha256_file(manifest_path)
        return result

    def _validate_split_manifest(self, rows: int, *, allow_partial: bool) -> None:
        data = self.config.raw_config.get("data", {})
        manifest_value = data.get("split_manifest") if isinstance(data, Mapping) else None
        if not manifest_value:
            return
        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_file():
            raise FileNotFoundError(f"split manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = int(manifest["splits"]["aux_search"]["rows"])
        if not allow_partial and rows != declared:
            raise ValueError(
                f"auxiliary split row count {rows} disagrees with manifest {declared}"
            )
        source = Path(self.config.aux_search_split)
        expected_hash = manifest.get("parquet_sha256", {}).get("aux_search")
        if source.is_file() and expected_hash and sha256_file(source) != expected_hash:
            raise ValueError("auxiliary split does not match its pinned manifest hash")

    def _require_stage_manifest(self, path: Path, expected_stage: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"required completed stage manifest is missing: {path}"
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("stage") != expected_stage:
            raise ValueError(f"unexpected stage manifest at {path}")
        if manifest.get("config_hash") != self.config_hash:
            raise ValueError(f"stage {expected_stage} used a different config")
        validate_output_hashes(manifest)

    def _problem_id(self, item: Mapping[str, Any], position: int) -> str:
        field = self.config.problem_id_field
        if field:
            value: Any = item
            for component in field.split("."):
                value = item_value(value, component, None)
                if value is None:
                    raise ValueError(
                        f"problem at position {position} lacks id field {field!r}"
                    )
            return str(value)
        for key in ("problem_id", "question_id", "id"):
            value = item_value(item, key, None)
            if value is not None:
                return str(value)
        extra = item_value(item, "extra_info", {})
        for key in ("problem_id", "filtered_row_id", "question_id", "index"):
            value = item_value(extra, key, None)
            if value is not None:
                return str(value)
        prompt_hash = content_hash(_plain(self.adapter.format_prompt(item)))[:20]
        return f"{self.config.domain}_{prompt_hash}"

    def _problem_set_hash(self, indexed: Mapping[str, Mapping[str, Any]]) -> str:
        return content_hash(
            [
                {"problem_id": problem_id, "item": _plain(item)}
                for problem_id, item in indexed.items()
            ]
        )

    def _seed_slot(self, problem_id: str, sample_index: int) -> str:
        return f"seed:{problem_id}:{sample_index:02d}"

    def _mutation_slot(
        self, problem_id: str, round_index: int, child_index: int
    ) -> str:
        return f"qd:{round_index:02d}:{child_index:02d}:{problem_id}"

    def _expected_candidate_slots(self, problem_ids: Sequence[str]) -> list[str]:
        slots = [
            self._seed_slot(problem_id, sample_index)
            for problem_id in problem_ids
            for sample_index in range(self.config.search.seed_samples_per_problem)
        ]
        slots.extend(
            self._mutation_slot(problem_id, round_index, child_index)
            for round_index in range(self.config.search.rounds)
            for child_index in range(
                self.config.search.children_per_round_per_problem
            )
            for problem_id in problem_ids
        )
        return slots

    def _seed_request(
        self, item: Mapping[str, Any], problem_id: str, sample_index: int
    ) -> GenerationRequest:
        slot_key = self._seed_slot(problem_id, sample_index)
        return GenerationRequest(
            slot_key=slot_key,
            problem_id=problem_id,
            method="seed",
            round=0,
            sample_index=sample_index,
            generation_seed=deterministic_seed(self.config.seed, slot_key),
            messages=tuple(dict(message) for message in self.adapter.format_prompt(item)),
            decoding=self.config.search.decoding,
            round_trip_gate_version=self.round_trip_gate.version,
        )

    def _mutation_request(
        self,
        *,
        item: Mapping[str, Any],
        problem_id: str,
        round_index: int,
        child_index: int,
        target: StrategyLabel | None,
        parent: CandidateRecord | None,
    ) -> GenerationRequest:
        slot_key = self._mutation_slot(problem_id, round_index, child_index)
        messages = [dict(message) for message in self.adapter.format_prompt(item)]
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("domain generation prompt must end with a user turn")
        if target is None:
            if self.config.domain != "optmath" or parent is not None:
                raise ValueError("Untargeted recovery is only supported for OptMATH without a parent")
            instruction = (
                "\n\nArchive recovery request: No evidenced applicable strategy is available. "
                "Construct a correct model of the original problem. Preserve all constraints, "
                "integrality and objective. Do not invent a target strategy or claim diversity. "
                "Return only complete raw Python source.")
        else:
            instruction = (
            "\n\nQD search instruction: Produce a correct solution using the "
            f"strategy '{target.name}' ({target.id}): {target.definition}. "
            "The solution must be substantially different from the verified "
            "parent when one is provided. Return only the executable artifact; "
            "do not include analysis, Markdown fences, reasoning tags, "
            "explanations, or alternatives."
        )
        if parent is not None:
            instruction += (
                "\n\nVerified parent artifact (use only as a starting point):\n"
                f"--- BEGIN PARENT ---\n{parent.canonical_response}\n"
                "--- END PARENT ---"
            )
        messages[-1]["content"] += instruction
        return GenerationRequest(
            slot_key=slot_key,
            problem_id=problem_id,
            method="qd_mutation",
            round=round_index + 1,
            sample_index=child_index,
            generation_seed=deterministic_seed(self.config.seed, slot_key),
            messages=tuple(messages),
            decoding=self.config.search.decoding,
            round_trip_gate_version=self.round_trip_gate.version,
            requested_strategy=target,
            parent_candidate_id=(None if parent is None else parent.candidate_id),
            parent_response=(None if parent is None else parent.canonical_response),
        )

    def _admit_mutation_request(self, request, item):
        """Count the actual rendered prompt before freezing its journal/cache ID.

        Whole parent omission preserves the question and target. If the target
        instruction itself does not fit, spend this slot on the exact original
        base prompt and record it as untargeted recovery. No token truncation.
        """
        if self.config.domain != "optmath":
            return request, {}
        try:
            limit = int(self.generator.max_prompt_tokens)
            count = self.generator.prompt_token_count
            original_tokens = count(request.messages)
        except (AttributeError, NotImplementedError) as exc:
            raise RuntimeError("OptMATH archive requires the generator's native prompt tokenizer and limit") from exc
        if limit <= 0:
            raise ValueError("Invalid native prompt token limit")
        audit = {"original_prompt_tokens": original_tokens, "prompt_limit": limit,
                 "parent_context_omitted": False, "prompt_fallback": None}
        admitted = request
        if original_tokens > limit and request.parent_candidate_id is not None:
            audit.update(parent_context_omitted=True,
                         omitted_parent_candidate_id=request.parent_candidate_id)
            admitted = self._mutation_request(item=item, problem_id=request.problem_id,
                round_index=request.round - 1, child_index=request.sample_index,
                target=request.requested_strategy, parent=None)
        final_tokens = count(admitted.messages)
        if final_tokens > limit:
            audit.update(prompt_fallback="original_base_prompt",
                         generation_kind="untargeted_recovery",
                         omitted_requested_strategy=None if request.requested_strategy is None else request.requested_strategy.id,
                         reason="mutation_prompt_exceeds_native_limit")
            admitted = dataclasses.replace(request,
                messages=tuple(dict(m) for m in self.adapter.format_prompt(item)),
                requested_strategy=None, parent_candidate_id=None, parent_response=None)
            final_tokens = count(admitted.messages)
        if final_tokens > limit:
            raise ValueError(f"OptMATH base prompt for {request.problem_id} has {final_tokens} native tokens, exceeds {limit}; release/tokenizer identity mismatch, no truncation allowed")
        audit["admitted_prompt_tokens"] = final_tokens
        return admitted, audit

    def _request_hash(self, request: GenerationRequest) -> str:
        return content_hash(
            {"generator_model": self.generator.model_id, "request": request.to_dict()}
        )

    def _ensure_seed_observations(
        self, indexed: Mapping[str, Mapping[str, Any]]
    ) -> tuple[SeedObservation, ...]:
        journal = AppendOnlyJournal(
            self.seed_observations_path, key_field="slot_key"
        )
        expected_requests = [
            self._seed_request(item, problem_id, sample_index)
            for problem_id, item in indexed.items()
            for sample_index in range(self.config.search.seed_samples_per_problem)
        ]
        expected_keys = [request.slot_key for request in expected_requests]
        actual_keys = [str(record["slot_key"]) for record in journal]
        if actual_keys != expected_keys[: len(actual_keys)]:
            raise ValueError(
                "seed observation journal is not a prefix of the configured split"
            )

        missing = expected_requests[len(actual_keys) :]
        for request_batch in _chunks(
            missing, self.config.search.generation_batch_size
        ):
            responses = self._generate_batch(request_batch)
            observations = self._evaluate_batch(request_batch, responses, indexed)
            for observation in observations:
                journal.append(observation.to_dict())
            print(f"archive seeds: {len(journal)}/{len(expected_requests)} proposals verified", flush=True)

        observations = tuple(SeedObservation.from_dict(record) for record in journal)
        if len(observations) != len(expected_requests):
            raise RuntimeError("seed generation did not complete every configured slot")
        return observations

    def _generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> tuple[GenerationResponse, ...]:
        if not requests:
            return ()
        responses = tuple(self.generator.generate_batch(requests))
        if len(responses) != len(requests):
            raise ValueError("generator returned the wrong batch cardinality")
        return responses

    def _evaluate_response(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
        item: Mapping[str, Any],
    ) -> SeedObservation:
        receipt_path = None
        receipt_identity = None
        if self.config.domain == "optmath":
            request_hash = self._request_hash(request)
            receipt_path = self.output_dir / "verifications" / f"{request_hash}.json"
            receipt_identity = {"config_hash": self.config_hash, "request_hash": request_hash,
                                "response_hash": content_hash(response.to_dict()),
                                "verifier_version": self.adapter.verifier_version}
            if receipt_path.exists():
                cached = json.loads(receipt_path.read_text())
                recorded_hash = cached.pop("receipt_hash", None)
                if cached.get("identity") != receipt_identity or content_hash(cached) != recorded_hash:
                    raise ValueError(f"OptMATH verification receipt identity/hash mismatch: {receipt_path}")
                return SeedObservation.from_dict(cached["observation"])
        over_limit = response.output_tokens > request.decoding.max_tokens
        if over_limit:
            parse = ParseResult.reject("response_token_limit_exceeded")
            roundtrip_accepted = False
            roundtrip_token_count = 0
            verification = VerificationResult.rejected(parse.error or "rejected")
        else:
            parse = self.adapter.parse_strict(response.raw_response)
            if parse.accepted:
                assert parse.canonical_response is not None
                roundtrip = self.round_trip_gate.check(parse.canonical_response)
                roundtrip_accepted = roundtrip.accepted
                roundtrip_token_count = roundtrip.token_count
                verification = (
                    self.adapter.verify(item, parse.canonical_response)
                    if roundtrip.accepted
                    else VerificationResult.rejected(
                        roundtrip.error or "tokenizer_roundtrip_rejected"
                    )
                )
            else:
                roundtrip_accepted = False
                roundtrip_token_count = 0
                verification = VerificationResult.rejected(
                    parse.error or "strict_parse"
                )
        observation = SeedObservation(
            slot_key=request.slot_key,
            request=request,
            response=response,
            parse_accepted=parse.accepted,
            roundtrip_accepted=roundtrip_accepted,
            roundtrip_token_count=roundtrip_token_count,
            round_trip_gate_version=self.round_trip_gate.version,
            canonical_response=parse.canonical_response,
            verifier_score=verification.score,
            fully_correct=verification.fully_correct,
            executable=verification.executable,
            verifier_error=verification.error or parse.error,
        )
        if receipt_path is not None:
            body = {"schema_version": "optmath.candidate_receipt.v1", "identity": receipt_identity,
                    "observation": observation.to_dict(), "verification": dataclasses.asdict(verification)}
            write_immutable_json(receipt_path, {**body, "receipt_hash": content_hash(body)})
        return observation

    def _evaluate_batch(
        self,
        requests: Sequence[GenerationRequest],
        responses: Sequence[GenerationResponse],
        indexed: Mapping[str, Mapping[str, Any]],
    ) -> tuple[SeedObservation, ...]:
        if len(requests) != len(responses):
            raise ValueError("request/response batch cardinality mismatch")
        calls = [
            (request, response, indexed[request.problem_id])
            for request, response in zip(requests, responses, strict=True)
        ]
        if not calls:
            return ()

        def evaluate(call: tuple[GenerationRequest, GenerationResponse, Mapping[str, Any]]):
            return self._evaluate_response(*call)

        with ThreadPoolExecutor(
            max_workers=min(self.config.search.verifier_workers, len(calls))
        ) as executor:
            # executor.map retains request order, so journals remain deterministic.
            return tuple(executor.map(evaluate, calls))

    def _label_batch(
        self, method: str, calls: Sequence[Mapping[str, Any]]
    ) -> tuple[Any | None, ...]:
        """Run labeler batches, retaining structural failures as empty items.

        The labeler's bounded retry layer distinguishes malformed model output
        from provider/infrastructure errors. Only the former is recoverable at
        item granularity; infrastructure failures continue to abort the stage.
        """

        results: list[Any | None] = []
        batch_method = getattr(self.labeler, f"{method}_batch")
        for batch in _chunks(calls, self.config.search.label_batch_size):
            try:
                generated = tuple(batch_method(batch))
            except MetadataBatchExhaustedError as exc:
                if not exc.structural_failure:
                    raise
                if exc.batch_size != len(batch):
                    raise RuntimeError(
                        "metadata failure batch cardinality mismatch"
                    ) from exc
                generated_list: list[Any | None] = [None] * len(batch)
                for index, result in exc.partial_results.items():
                    generated_list[index] = result
                generated = tuple(generated_list)
                LOGGER.warning(
                    "Treating %d/%d %s metadata replies as invalid after "
                    "bounded structural retries",
                    len(exc.failed_indexes),
                    len(batch),
                    method,
                )
            if len(generated) != len(batch):
                raise ValueError(
                    f"labeler {method}_batch returned the wrong number of results"
                )
            results.extend(generated)
        return tuple(results)

    def _classify_observations(
        self,
        observations: Sequence[SeedObservation],
        indexed: Mapping[str, Mapping[str, Any]],
        taxonomy: Taxonomy,
    ) -> tuple[tuple[LabelDecision | None, str | None], ...]:
        eligible_indexes: list[int] = []
        calls: list[dict[str, Any]] = []
        output: list[tuple[LabelDecision | None, str | None]] = [
            (None, None) for _ in observations
        ]
        for index, observation in enumerate(observations):
            if not (
                observation.parse_accepted
                and observation.roundtrip_accepted
                and observation.executable
                and observation.fully_correct
                and observation.canonical_response is not None
            ):
                continue
            eligible_indexes.append(index)
            calls.append(
                {
                    "problem_id": observation.request.problem_id,
                    "labeler_input": self.adapter.format_labeler_input(
                        indexed[observation.request.problem_id],
                        observation.canonical_response,
                    ),
                    "taxonomy": taxonomy,
                    "seed": deterministic_seed(
                        self.config.seed, "classify", observation.slot_key
                    ),
                    "deterministic": False,
                }
            )
        initial = self._label_batch("classify", calls)
        low_calls: list[dict[str, Any]] = []
        low_positions: list[int] = []
        decisions: dict[int, LabelDecision] = {}
        for position, decision in zip(eligible_indexes, initial, strict=True):
            if decision is None:
                output[position] = (None, "label_metadata_invalid_after_retries")
                continue
            if not isinstance(decision, LabelDecision):
                raise TypeError("labeler classify must return LabelDecision")
            decisions[position] = decision
            if decision.confidence < self.config.search.label_confidence_min:
                observation = observations[position]
                low_positions.append(position)
                low_calls.append(
                    {
                        "problem_id": observation.request.problem_id,
                        "labeler_input": self.adapter.format_labeler_input(
                            indexed[observation.request.problem_id],
                            observation.canonical_response or "",
                        ),
                        "taxonomy": taxonomy,
                        "seed": deterministic_seed(
                            self.config.seed, "classify_retry", observation.slot_key
                        ),
                        "deterministic": True,
                    }
                )
        retried = self._label_batch("classify", low_calls)
        for position, decision in zip(low_positions, retried, strict=True):
            if decision is None:
                continue
            if not isinstance(decision, LabelDecision):
                raise TypeError("labeler classify retry must return LabelDecision")
            decisions[position] = decision

        known_ids = {label.id for label in taxonomy.labels}
        for position, decision in decisions.items():
            error: str | None = None
            if decision.primary_strategy not in known_ids:
                error = "unknown_primary_strategy"
            elif (
                decision.secondary_strategy is not None
                and decision.secondary_strategy not in known_ids
            ):
                error = "unknown_secondary_strategy"
            elif decision.confidence < self.config.search.label_confidence_min:
                error = "label_confidence_below_threshold"
            output[position] = (decision, error)
        return tuple(output)

    def _candidate_from_observation(
        self,
        observation: SeedObservation,
        *,
        taxonomy: Taxonomy,
        decision: LabelDecision | None,
        label_error: str | None,
    ) -> CandidateRecord:
        request = observation.request
        request_hash = self._request_hash(request)
        primary = (
            None
            if decision is None or label_error is not None
            else decision.primary_strategy
        )
        error_parts = [
            part
            for part in (observation.verifier_error, label_error)
            if part is not None
        ]
        from .subtypes import structural_subtype, cell_id
        subtype = None
        if primary is not None and self.config.search.cell_descriptor == "primary_coarse_v1":
            subtype = structural_subtype(self.config.domain, observation.canonical_response or "")
        return CandidateRecord(
            candidate_id=stable_candidate_id(request_hash),
            slot_key=request.slot_key,
            request_hash=request_hash,
            problem_id=request.problem_id,
            method=request.method,
            round=request.round,
            parent_candidate_id=request.parent_candidate_id,
            requested_strategy=(
                None
                if request.requested_strategy is None
                else request.requested_strategy.id
            ),
            raw_response=observation.response.raw_response,
            canonical_response=observation.canonical_response,
            finish_reason=observation.response.finish_reason,
            input_tokens=observation.response.input_tokens,
            output_tokens=observation.response.output_tokens,
            verifier_score=observation.verifier_score,
            fully_correct=observation.fully_correct,
            executable=observation.executable,
            parse_accepted=observation.parse_accepted,
            roundtrip_accepted=observation.roundtrip_accepted,
            roundtrip_token_count=observation.roundtrip_token_count,
            primary_strategy=primary,
            secondary_strategy=None,
            label_confidence=(
                None if decision is None else decision.confidence
            ),
            label_justification=(
                None if decision is None else decision.justification
            ),
            niche=cell_id(primary, subtype),
            structural_subtype=subtype,
            accepted_into_archive=False,
            replaced_candidate_id=None,
            generator_model=observation.response.model,
            labeler_model=self.labeler.model_id,
            generation_seed=request.generation_seed,
            taxonomy_hash=taxonomy.hash,
            verifier_version=self.adapter.verifier_version,
            round_trip_gate_version=self.round_trip_gate.version,
            error=";".join(error_parts) or None,
        )

    def _ensure_feasible_plans(
        self,
        indexed: Mapping[str, Mapping[str, Any]],
        taxonomy: Taxonomy,
        archive: MapElitesArchive,
    ) -> dict[str, tuple[str, ...]]:
        journal = AppendOnlyJournal(self.feasible_path, key_field="problem_id")
        actual = [str(row["problem_id"]) for row in journal]
        expected = list(indexed)
        if actual != expected[: len(actual)]:
            raise ValueError(
                "feasible-strategy journal is not a prefix of the problem split"
            )
        missing = expected[len(actual) :]
        calls: list[dict[str, Any]] = []
        for problem_id in missing:
            observed = tuple(sorted({
                candidate.primary_strategy
                for candidate in archive.elites_for(problem_id)
                if candidate.primary_strategy is not None
            }))
            calls.append(
                {
                    "problem_id": problem_id,
                    "problem_context": canonical_json(
                        self.adapter.format_prompt(indexed[problem_id])
                    ),
                    "taxonomy": taxonomy,
                    "observed": observed,
                    "limit": self.config.search.taxonomy_feasible_limit,
                    "seed": deterministic_seed(
                        self.config.seed, "feasible", problem_id
                    ),
                }
            )
        planned = self._label_batch("plan_feasible", calls)
        known = {label.id for label in taxonomy.labels}
        for problem_id, proposed in zip(missing, planned, strict=True):
            observed = sorted({
                candidate.primary_strategy
                for candidate in archive.elites_for(problem_id)
                if candidate.primary_strategy is not None
            })
            metadata_fallback = proposed is None
            if proposed is None:
                # Observed niches are known feasible for this problem. If none
                # exist, use a stable subset of the frozen taxonomy; generated
                # candidates still pass the normal strict verifier/classifier.
                proposed_ids = list(dict.fromkeys(observed))
                if not proposed_ids and self.config.domain != "optmath":
                    proposed_ids = sorted(
                        (label.id for label in taxonomy.labels),
                        key=lambda strategy_id: content_hash(
                            {
                                "seed": self.config.seed,
                                "namespace": "feasible_metadata_fallback",
                                "problem_id": problem_id,
                                "strategy_id": strategy_id,
                            }
                        ),
                    )[: self.config.search.taxonomy_feasible_limit]
            else:
                proposed_ids = [str(item) for item in proposed]
            unknown = [item for item in proposed_ids if item not in known]
            if unknown:
                raise ValueError(
                    f"feasibility labeler returned unknown strategies: {unknown}"
                )
            combined: list[str] = []
            for strategy in [*observed, *proposed_ids]:
                if strategy not in combined:
                    combined.append(strategy)
            if len(combined) > self.config.search.taxonomy_feasible_limit:
                required = list(dict.fromkeys(observed))
                optional = [item for item in combined if item not in required]
                combined = required + optional[
                    : max(0, self.config.search.taxonomy_feasible_limit - len(required))
                ]
            if not combined and self.config.domain != "optmath":
                raise ValueError(
                    f"feasibility labeler returned no strategy for {problem_id!r}"
                )
            record = {
                "problem_id": problem_id,
                "strategy_ids": combined,
                "taxonomy_hash": taxonomy.hash,
            }
            if metadata_fallback:
                record["metadata_fallback"] = "invalid_after_retries"
            journal.append(record)
        return {
            str(row["problem_id"]): tuple(str(item) for item in row["strategy_ids"])
            for row in journal
        }

    def _load_taxonomy(self) -> Taxonomy:
        if not self.taxonomy_path.is_file():
            raise FileNotFoundError(
                f"taxonomy is not frozen; run induce_taxonomy first: {self.taxonomy_path}"
            )
        taxonomy = Taxonomy.from_dict(
            json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        )
        if taxonomy.domain != self.config.domain:
            raise ValueError("taxonomy domain does not match pipeline domain")
        validate_taxonomy_size(taxonomy, self.config.search.taxonomy_size)
        return taxonomy
