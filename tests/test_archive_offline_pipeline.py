from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from elitereplay.archive.archive import MapElitesArchive
from elitereplay.archive.domain import ParseResult, VerificationResult
from elitereplay.archive.pipeline import EliteReplayPipeline
from elitereplay.archive.providers import MetadataBatchExhaustedError
from elitereplay.archive.schema import (
    GenerationRequest,
    GenerationResponse,
    LabelDecision,
    PipelineConfig,
    StrategyLabel,
    Taxonomy,
    load_pipeline_config,
)
from elitereplay.archive.trinity_support import UniformProblemNicheArchive
from elitereplay.archive.tokenization import FunctionalTokenizerRoundTripGate


class FakeAdapter:
    verifier_version = "fake-verifier-v1"

    def __init__(self) -> None:
        self.verified_responses: list[str] = []

    def format_prompt(self, item: Mapping[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "Return one artifact."},
            {"role": "user", "content": str(item["prompt_text"])},
        ]

    def parse_strict(self, raw_response: str) -> ParseResult:
        if not raw_response.startswith("artifact::"):
            return ParseResult.reject("bad_artifact")
        return ParseResult.accept(raw_response)

    def verify(
        self, item: Mapping[str, Any], canonical_response: str
    ) -> VerificationResult:
        self.verified_responses.append(canonical_response)
        return VerificationResult(
            score=1.0,
            fully_correct=True,
            executable=True,
            per_test_scores=(1.0,),
        )

    def format_labeler_input(
        self, item: Mapping[str, Any], canonical_response: str
    ) -> str:
        return f"{item['problem_id']}\n{canonical_response}"

    def sanitize_for_sft(self, canonical_response: str) -> str:
        if not self.parse_strict(canonical_response).accepted:
            raise ValueError("not canonical")
        return canonical_response


class FakeGenerator:
    model_id = "fake-base"

    def __init__(self) -> None:
        self.batch_calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return self.generate_batch([request])[0]

    def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResponse]:
        self.batch_calls += 1
        return tuple(
            GenerationResponse(
                raw_response=f"artifact::{request.slot_key}",
                finish_reason="stop",
                input_tokens=10,
                output_tokens=3 + request.sample_index,
                model=self.model_id,
            )
            for request in requests
        )


class FakeLabeler:
    model_id = "fake-labeler"

    def __init__(self) -> None:
        self.calls = 0

    def describe(self, *, problem_id: str, labeler_input: str, seed: int) -> str:
        self.calls += 1
        return f"strategy for {problem_id} seed {seed}"

    def consolidate(
        self,
        *,
        domain: str,
        descriptions: Sequence[str],
        size: int,
        seed: int,
    ) -> Sequence[StrategyLabel]:
        self.calls += 1
        labels = (
            StrategyLabel("strategy_a", "A", "Use approach A."),
            StrategyLabel("strategy_b", "B", "Use approach B."),
        )
        return labels[:size]

    def classify(
        self,
        *,
        problem_id: str,
        labeler_input: str,
        taxonomy: Taxonomy,
        seed: int,
        deterministic: bool,
    ) -> LabelDecision:
        self.calls += 1
        label = taxonomy.labels[seed % len(taxonomy.labels)].id
        return LabelDecision(label, 0.99, f"uses {label}")

    def plan_feasible(
        self,
        *,
        problem_id: str,
        problem_context: str,
        taxonomy: Taxonomy,
        observed: Sequence[str],
        limit: int,
        seed: int,
    ) -> Sequence[str]:
        self.calls += 1
        return tuple(label.id for label in taxonomy.labels[:limit])


class PartiallyMalformedLabeler(FakeLabeler):
    """Fail one item once in each metadata-only batch operation."""

    def __init__(self) -> None:
        super().__init__()
        self.failed_describe = False
        self.failed_classify = False
        self.failed_feasible = False
        self.consolidated_descriptions: tuple[str, ...] = ()

    @staticmethod
    def _structural_failure(valid: Any, operation: str) -> None:
        raise MetadataBatchExhaustedError(
            f"{operation} malformed fixture",
            partial_results={0: valid},
            failed_indexes=(1,),
            batch_size=2,
            structural_failure=True,
        )

    def describe_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[str]:
        assert len(requests) == 2
        if not self.failed_describe:
            self.failed_describe = True
            self._structural_failure(self.describe(**requests[0]), "describe")
        return tuple(self.describe(**request) for request in requests)

    def consolidate(
        self,
        *,
        domain: str,
        descriptions: Sequence[str],
        size: int,
        seed: int,
    ) -> Sequence[StrategyLabel]:
        self.consolidated_descriptions = tuple(descriptions)
        return super().consolidate(
            domain=domain, descriptions=descriptions, size=size, seed=seed
        )

    def classify_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[LabelDecision]:
        assert len(requests) == 2
        if not self.failed_classify:
            self.failed_classify = True
            self._structural_failure(self.classify(**requests[0]), "classify")
        return tuple(self.classify(**request) for request in requests)

    def plan_feasible_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[Sequence[str]]:
        assert len(requests) == 2
        if not self.failed_feasible:
            self.failed_feasible = True
            self._structural_failure(
                self.plan_feasible(**requests[0]), "plan_feasible"
            )
        return tuple(self.plan_feasible(**request) for request in requests)


class PartiallyFailedProviderLabeler(FakeLabeler):
    def describe_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[str]:
        assert len(requests) == 2
        raise MetadataBatchExhaustedError(
            "fixture provider failure",
            partial_results={0: self.describe(**requests[0])},
            failed_indexes=(1,),
            batch_size=2,
            structural_failure=False,
        )


def _config(tmp_path: Path) -> PipelineConfig:
    raw = {
        "experiment": {
            "name": "fake_qd_chord",
            "seed": 7,
            "output_dir": str(tmp_path / "offline"),
        },
        "models": {
            "base_model": "fake-base",
            "tokenizer": "fake-base",
            "tokenizer_revision": "fake-revision",
            "labeler_model": "fake-labeler",
        },
        "data": {
            "domain": "fake",
            "aux_search_split": "fake.jsonl",
            "aux_prompt_count": 2,
            "problem_id_field": "problem_id",
        },
        "taxonomy": {
            "version": "fake-v1",
            "size": 2,
            "sample_cap": 4,
            "feasible_limit": 2,
            "confidence_min": 0.6,
        },
        "search": {
            "seed_samples_per_problem": 2,
            "rounds": 1,
            "children_per_round_per_problem": 2,
            "target_elites_per_problem": 2,
            "temperature": 0.9,
            "top_p": 0.95,
            "max_response_tokens": 32,
            "generation_batch_size": 2,
            "label_batch_size": 2,
        },
        "export": {
            "max_examples": 2,
            "publish_path": str(tmp_path / "sft_archive.jsonl"),
        },
    }
    return PipelineConfig.from_mapping(raw)


def _identity_gate() -> FunctionalTokenizerRoundTripGate:
    return FunctionalTokenizerRoundTripGate(lambda text: text, name="test_identity")


def test_tiny_pipeline_is_cached_resumable_and_trinity_compatible(tmp_path: Path):
    problems = [
        {"problem_id": "p0", "prompt_text": "problem zero"},
        {"problem_id": "p1", "prompt_text": "problem one"},
    ]
    config = _config(tmp_path)
    first_generator = FakeGenerator()
    first_labeler = FakeLabeler()
    first = EliteReplayPipeline(
        config=config,
        adapter=FakeAdapter(),
        generator=first_generator,
        labeler=first_labeler,
        round_trip_gate=_identity_gate(),
    )

    taxonomy = first.induce_taxonomy(problems)
    archive = first.build_archive(problems)
    manifest = first.export_sft(problems)

    assert len(taxonomy.labels) == 2
    assert archive["archive_hash"]
    assert manifest["counts"]["exported_examples"] == 2
    assert first_generator.batch_calls == 4  # two seed + two mutation waves
    assert first_labeler.calls > 0

    published = tmp_path / "sft_archive.jsonl"
    rows = [json.loads(line) for line in published.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["metadata"]["fully_correct"] is True for row in rows)
    assert all(row["metadata"]["sample_source"] == "frozen_sft" for row in rows)
    archive_reader = UniformProblemNicheArchive(rows, seed=7)
    assert archive_reader.problem_count == 2
    seed_row = json.loads(first.seed_observations_path.read_text().splitlines()[0])
    assert seed_row["request"]["round_trip_gate_version"] == _identity_gate().version
    taxonomy_manifest = json.loads(
        (first.taxonomy_dir / "manifest.json").read_text()
    )
    assert (
        taxonomy_manifest["inputs"]["round_trip_gate_version"]
        == _identity_gate().version
    )

    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    second_generator = FakeGenerator()
    second_labeler = FakeLabeler()
    resumed = EliteReplayPipeline(
        config=config,
        adapter=FakeAdapter(),
        generator=second_generator,
        labeler=second_labeler,
        round_trip_gate=_identity_gate(),
    )
    resumed.induce_taxonomy(problems)
    resumed.build_archive(problems)
    resumed.export_sft(problems)
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert before == after
    assert second_generator.batch_calls == 0
    assert second_labeler.calls == 0


def test_round_trip_mismatch_is_logged_and_never_archived(tmp_path: Path):
    problems = [
        {"problem_id": "p0", "prompt_text": "problem zero"},
        {"problem_id": "p1", "prompt_text": "problem one"},
    ]
    rejected_text = "artifact::seed:p0:00"
    gate = FunctionalTokenizerRoundTripGate(
        lambda text: text + " " if text == rejected_text else text,
        name="test_one_mismatch",
    )
    adapter = FakeAdapter()
    pipeline = EliteReplayPipeline(
        config=_config(tmp_path),
        adapter=adapter,
        generator=FakeGenerator(),
        labeler=FakeLabeler(),
        round_trip_gate=gate,
    )

    pipeline.induce_taxonomy(problems)
    pipeline.build_archive(problems)

    candidates = [
        json.loads(line)
        for line in pipeline.candidates_path.read_text().splitlines()
    ]
    rejected = next(row for row in candidates if row["raw_response"] == rejected_text)
    assert rejected["canonical_response"] == rejected_text
    assert rejected["parse_accepted"] is True
    assert rejected["roundtrip_accepted"] is False
    assert rejected["round_trip_gate_version"] == gate.version
    assert rejected["executable"] is False
    assert rejected["fully_correct"] is False
    assert rejected["accepted_into_archive"] is False
    assert rejected["error"] == "serialize_tokenize_decode_mismatch"
    assert rejected_text not in adapter.verified_responses


def test_valid_length_finished_response_is_verified_and_keeps_reason(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    pipeline = EliteReplayPipeline(
        config=_config(tmp_path),
        adapter=adapter,
        generator=FakeGenerator(),
        labeler=FakeLabeler(),
        round_trip_gate=_identity_gate(),
    )
    item = {"problem_id": "p0", "prompt_text": "problem zero"}
    request = pipeline._seed_request(item, "p0", 0)
    response = GenerationResponse(
        raw_response="artifact::complete-at-limit",
        finish_reason="length",
        input_tokens=10,
        output_tokens=request.decoding.max_tokens,
        model="fake-base",
    )

    observation = pipeline._evaluate_response(request, response, item)

    assert observation.parse_accepted is True
    assert observation.roundtrip_accepted is True
    assert observation.executable is True
    assert observation.fully_correct is True
    assert observation.verifier_error is None
    assert adapter.verified_responses == ["artifact::complete-at-limit"]
    assert observation.response.finish_reason == "length"

    taxonomy = Taxonomy(
        domain="fake",
        version="fake-v1",
        labeler_model="fake-labeler",
        labels=(StrategyLabel("strategy_a", "A", "Use approach A."),),
    )
    candidate = pipeline._candidate_from_observation(
        observation,
        taxonomy=taxonomy,
        decision=LabelDecision("strategy_a", 0.99, "uses strategy A"),
        label_error=None,
    )
    assert candidate.finish_reason == "length"
    assert candidate.fully_correct is True
    archived = MapElitesArchive().insert(candidate)
    assert archived.accepted_into_archive is True
    assert archived.niche == "strategy_a"


def test_response_over_token_limit_is_rejected_before_parse_or_verify(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()

    def unexpected_parse(_: str) -> ParseResult:
        raise AssertionError("over-limit response must not be parsed")

    adapter.parse_strict = unexpected_parse  # type: ignore[method-assign]
    pipeline = EliteReplayPipeline(
        config=_config(tmp_path),
        adapter=adapter,
        generator=FakeGenerator(),
        labeler=FakeLabeler(),
        round_trip_gate=_identity_gate(),
    )
    item = {"problem_id": "p0", "prompt_text": "problem zero"}
    request = pipeline._seed_request(item, "p0", 0)
    response = GenerationResponse(
        raw_response="artifact::over-limit",
        finish_reason="stop",
        input_tokens=10,
        output_tokens=request.decoding.max_tokens + 1,
        model="fake-base",
    )

    observation = pipeline._evaluate_response(request, response, item)

    assert observation.parse_accepted is False
    assert observation.roundtrip_accepted is False
    assert observation.executable is False
    assert observation.fully_correct is False
    assert observation.verifier_error == "response_token_limit_exceeded"
    assert adapter.verified_responses == []


def test_structurally_invalid_metadata_is_skipped_per_item(tmp_path: Path):
    problems = [
        {"problem_id": "p0", "prompt_text": "problem zero"},
        {"problem_id": "p1", "prompt_text": "problem one"},
    ]
    labeler = PartiallyMalformedLabeler()
    pipeline = EliteReplayPipeline(
        config=_config(tmp_path),
        adapter=FakeAdapter(),
        generator=FakeGenerator(),
        labeler=labeler,
        round_trip_gate=_identity_gate(),
    )

    taxonomy = pipeline.induce_taxonomy(problems)
    pipeline.build_archive(problems)

    # Four correct seed solutions were available; one malformed description
    # was omitted while all independently valid descriptions were retained.
    assert len(labeler.consolidated_descriptions) == 3
    taxonomy_manifest = json.loads(
        (pipeline.taxonomy_dir / "manifest.json").read_text()
    )
    assert taxonomy_manifest["counts"]["taxonomy_examples"] == 3

    candidates = [
        json.loads(line) for line in pipeline.candidates_path.read_text().splitlines()
    ]
    unclassified = [
        candidate
        for candidate in candidates
        if candidate["error"] == "label_metadata_invalid_after_retries"
    ]
    assert len(unclassified) == 1
    assert unclassified[0]["fully_correct"] is True
    assert unclassified[0]["primary_strategy"] is None
    assert unclassified[0]["accepted_into_archive"] is False

    feasible_rows = [
        json.loads(line) for line in pipeline.feasible_path.read_text().splitlines()
    ]
    fallback = [row for row in feasible_rows if "metadata_fallback" in row]
    assert len(fallback) == 1
    assert fallback[0]["metadata_fallback"] == "invalid_after_retries"
    known = {label.id for label in taxonomy.labels}
    assert fallback[0]["strategy_ids"]
    assert set(fallback[0]["strategy_ids"]).issubset(known)


def test_metadata_provider_failure_is_not_downgraded(tmp_path: Path):
    pipeline = EliteReplayPipeline(
        config=_config(tmp_path),
        adapter=FakeAdapter(),
        generator=FakeGenerator(),
        labeler=PartiallyFailedProviderLabeler(),
        round_trip_gate=_identity_gate(),
    )
    calls = [
        {"problem_id": "p0", "labeler_input": "input zero", "seed": 1},
        {"problem_id": "p1", "labeler_input": "input one", "seed": 2},
    ]

    with pytest.raises(MetadataBatchExhaustedError) as captured:
        pipeline._label_batch("describe", calls)

    assert captured.value.structural_failure is False
