"""Stable schemas and configuration for the offline EliteReplay pipeline.

The offline search is intentionally independent from Trinity-RFT.  Objects in
this module are JSON serialisable and hashed canonically so every expensive
request and every frozen artifact can be identified across restarts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


SCHEMA_VERSION = "EliteReplay.v1"
_STRATEGY_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def _jsonable(value: Any) -> Any:
    """Return a deterministic, JSON-compatible representation of *value*."""

    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not permit non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serialisable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialise *value* using the canonical representation used for hashes."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def deterministic_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable non-negative 31-bit seed without mutable RNG state."""

    digest = content_hash({"base_seed": base_seed, "parts": parts})
    return int(digest[:16], 16) % (2**31)


@dataclass(frozen=True)
class StrategyLabel:
    id: str
    name: str
    definition: str
    positive_cues: tuple[str, ...] = ()
    negative_cues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _STRATEGY_ID.fullmatch(self.id):
            raise ValueError(
                f"invalid strategy id {self.id!r}; expected stable_snake_case"
            )
        if not self.name.strip() or not self.definition.strip():
            raise ValueError("strategy name and definition must be non-empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyLabel":
        return cls(
            id=str(value["id"]),
            name=str(value["name"]),
            definition=str(value["definition"]),
            positive_cues=tuple(str(v) for v in value.get("positive_cues", ())),
            negative_cues=tuple(str(v) for v in value.get("negative_cues", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class Taxonomy:
    domain: str
    version: str
    labeler_model: str
    labels: tuple[StrategyLabel, ...]
    schema_version: str = "EliteReplay.taxonomy.v1"

    def __post_init__(self) -> None:
        if not self.domain.strip() or not self.version.strip():
            raise ValueError("taxonomy domain and version must be non-empty")
        ids = [label.id for label in self.labels]
        if not ids:
            raise ValueError("taxonomy must contain at least one strategy")
        if len(ids) != len(set(ids)):
            raise ValueError("taxonomy strategy ids must be unique")

    @property
    def hash(self) -> str:
        return content_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "version": self.version,
            "labeler_model": self.labeler_model,
            "labels": [label.to_dict() for label in self.labels],
        }
        if include_hash:
            result["taxonomy_hash"] = self.hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Taxonomy":
        taxonomy = cls(
            domain=str(value["domain"]),
            version=str(value["version"]),
            labeler_model=str(value["labeler_model"]),
            labels=tuple(StrategyLabel.from_dict(v) for v in value["labels"]),
            schema_version=str(
                value.get("schema_version", "EliteReplay.taxonomy.v1")
            ),
        )
        recorded_hash = value.get("taxonomy_hash")
        if recorded_hash is not None and recorded_hash != taxonomy.hash:
            raise ValueError("taxonomy content does not match taxonomy_hash")
        return taxonomy

    def require_label(self, strategy_id: str) -> StrategyLabel:
        for label in self.labels:
            if label.id == strategy_id:
                return label
        raise ValueError(f"unknown strategy id: {strategy_id!r}")


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class SearchConfig:
    aux_prompt_count: int = 500
    taxonomy_size: int = 24
    taxonomy_sample_cap: int = 1000
    taxonomy_feasible_limit: int = 8
    label_confidence_min: float = 0.60
    seed_samples_per_problem: int = 8
    rounds: int = 3
    children_per_round_per_problem: int = 4
    target_elites_per_problem: int = 4
    export_max_examples: int = 500
    generation_batch_size: int = 128
    label_batch_size: int = 128
    verifier_workers: int = 16
    cell_descriptor: str = "primary"
    attempt_aware: bool = False
    decoding: DecodingConfig = field(default_factory=DecodingConfig)

    def __post_init__(self) -> None:
        positive = {
            "aux_prompt_count": self.aux_prompt_count,
            "taxonomy_size": self.taxonomy_size,
            "taxonomy_sample_cap": self.taxonomy_sample_cap,
            "taxonomy_feasible_limit": self.taxonomy_feasible_limit,
            "seed_samples_per_problem": self.seed_samples_per_problem,
            "target_elites_per_problem": self.target_elites_per_problem,
            "export_max_examples": self.export_max_examples,
            "generation_batch_size": self.generation_batch_size,
            "label_batch_size": self.label_batch_size,
            "verifier_workers": self.verifier_workers,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rounds < 0 or self.children_per_round_per_problem < 0:
            raise ValueError("round and child counts must be non-negative")
        if not 0 <= self.label_confidence_min <= 1:
            raise ValueError("label_confidence_min must be in [0, 1]")
        if self.cell_descriptor not in ("primary", "primary_coarse_v1"):
            raise ValueError("Unknown archive cell descriptor")
        if type(self.attempt_aware) is not bool:
            raise ValueError("attempt_aware must be a boolean")

    @property
    def generations_per_problem(self) -> int:
        return self.seed_samples_per_problem + (
            self.rounds * self.children_per_round_per_problem
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SearchConfig":
        taxonomy = value.get("taxonomy", {})
        search = value.get("search", value)
        export = value.get("export", {})
        return cls(
            aux_prompt_count=int(
                value.get("data", {}).get("aux_prompt_count", 500)
            ),
            taxonomy_size=int(taxonomy.get("size", 24)),
            taxonomy_sample_cap=int(taxonomy.get("sample_cap", 1000)),
            taxonomy_feasible_limit=int(taxonomy.get("feasible_limit", 8)),
            label_confidence_min=float(taxonomy.get("confidence_min", 0.60)),
            seed_samples_per_problem=int(
                search.get("seed_samples_per_problem", 8)
            ),
            rounds=int(search.get("rounds", 3)),
            children_per_round_per_problem=int(
                search.get("children_per_round_per_problem", 4)
            ),
            target_elites_per_problem=int(
                search.get("target_elites_per_problem", 4)
            ),
            export_max_examples=int(export.get("max_examples", 500)),
            generation_batch_size=int(search.get("generation_batch_size", 128)),
            label_batch_size=int(search.get("label_batch_size", 128)),
            verifier_workers=int(search.get("verifier_workers", 16)),
            cell_descriptor=str(search.get("cell_descriptor", "primary")),
            attempt_aware=search.get("attempt_aware", False),
            decoding=DecodingConfig(
                temperature=float(search.get("temperature", 0.9)),
                top_p=float(search.get("top_p", 0.95)),
                max_tokens=int(search.get("max_response_tokens", 1024)),
            ),
        )


@dataclass(frozen=True)
class PipelineConfig:
    experiment_name: str
    seed: int
    domain: str
    base_model: str
    tokenizer: str
    labeler_model: str
    aux_search_split: str
    problem_id_field: str | None
    output_dir: str
    sft_publish_path: str | None
    taxonomy_version: str
    search: SearchConfig
    raw_config: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, output_dir: str | None = None
    ) -> "PipelineConfig":
        experiment = value.get("experiment", {})
        models = value.get("models", {})
        data = value.get("data", {})
        taxonomy = value.get("taxonomy", {})
        export = value.get("export", {})
        search_section = value.get("search", {})
        if search_section.get("method", "qd") != "qd":
            raise ValueError("offline EliteReplay pipeline supports only search.method=qd")
        if export.get("deduplication", "none") not in {"none", "canonical_ast"}:
            raise ValueError("export.deduplication must be none or canonical_ast")
        target = value.get("archive_target")
        if target is not None:
            lower = int(target["min_examples"])
            upper = int(target["max_examples"])
            if not 0 < lower <= upper <= int(export.get("max_examples", 500)):
                raise ValueError("archive_target must fit within export.max_examples")
        required = {
            "experiment.name": experiment.get("name"),
            "models.base_model": models.get("base_model"),
            "models.tokenizer": models.get("tokenizer"),
            "models.labeler_model": models.get("labeler_model"),
            "data.domain": data.get("domain"),
            "data.aux_search_split": data.get("aux_search_split"),
        }
        missing = [name for name, item in required.items() if not item]
        if missing:
            raise ValueError("missing required config values: " + ", ".join(missing))
        resolved_output = output_dir or experiment.get("output_dir")
        if not resolved_output:
            raise ValueError(
                "output directory is required via experiment.output_dir or CLI"
            )
        return cls(
            experiment_name=str(experiment["name"]),
            seed=int(experiment.get("seed", 42)),
            domain=str(data["domain"]),
            base_model=str(models["base_model"]),
            tokenizer=str(models["tokenizer"]),
            labeler_model=str(models["labeler_model"]),
            aux_search_split=str(data["aux_search_split"]),
            problem_id_field=(
                None
                if data.get("problem_id_field") is None
                else str(data["problem_id_field"])
            ),
            output_dir=str(resolved_output),
            sft_publish_path=(
                None
                if export.get("publish_path") is None
                else str(export["publish_path"])
            ),
            taxonomy_version=str(taxonomy.get("version", "v1")),
            search=SearchConfig.from_mapping(value),
            raw_config=dict(value),
        )


def load_pipeline_config(path: str | Path, *, output_dir: str | None = None) -> PipelineConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - PyYAML is in runtime images
            raise RuntimeError("PyYAML is required to load a YAML config") from exc
        raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError("EliteReplay config must contain a mapping")
    return PipelineConfig.from_mapping(raw, output_dir=output_dir)


@dataclass(frozen=True)
class GenerationRequest:
    slot_key: str
    problem_id: str
    method: Literal["seed", "qd_mutation"]
    round: int
    sample_index: int
    generation_seed: int
    messages: tuple[Mapping[str, str], ...]
    decoding: DecodingConfig
    round_trip_gate_version: str
    requested_strategy: StrategyLabel | None = None
    parent_candidate_id: str | None = None
    parent_response: str | None = None

    def __post_init__(self) -> None:
        if not self.slot_key or not self.problem_id:
            raise ValueError("generation slot and problem id must be non-empty")
        if self.method not in {"seed", "qd_mutation"}:
            raise ValueError(f"unsupported generation method: {self.method!r}")
        if self.round < 0 or self.sample_index < 0 or self.generation_seed < 0:
            raise ValueError("generation indexes and seed must be non-negative")
        if not self.messages:
            raise ValueError("generation request must contain prompt messages")
        if not self.round_trip_gate_version:
            raise ValueError("generation request must pin the round-trip gate")
        if (self.parent_candidate_id is None) != (self.parent_response is None):
            raise ValueError("parent id and response must either both be set or both absent")

    @property
    def hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, request: Mapping[str, Any]) -> "GenerationRequest":
        requested = request.get("requested_strategy")
        return cls(
            slot_key=str(request["slot_key"]),
            problem_id=str(request["problem_id"]),
            method=str(request["method"]),  # type: ignore[arg-type]
            round=int(request["round"]),
            sample_index=int(request["sample_index"]),
            generation_seed=int(request["generation_seed"]),
            messages=tuple(dict(message) for message in request["messages"]),
            decoding=DecodingConfig(**request["decoding"]),
            round_trip_gate_version=str(request["round_trip_gate_version"]),
            requested_strategy=(
                None if requested is None else StrategyLabel.from_dict(requested)
            ),
            parent_candidate_id=request.get("parent_candidate_id"),
            parent_response=request.get("parent_response"),
        )


@dataclass(frozen=True)
class GenerationResponse:
    raw_response: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    model: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("generation token counts must be non-negative")
        if not self.model:
            raise ValueError("generation response must record its model")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationResponse":
        return cls(
            raw_response=str(value["raw_response"]),
            finish_reason=str(value.get("finish_reason", "other")),
            input_tokens=int(value.get("input_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
            model=str(value["model"]),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class LabelDecision:
    primary_strategy: str | None
    confidence: float
    justification: str
    secondary_strategy: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("label confidence must be in [0, 1]")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LabelDecision":
        primary = value.get("primary_strategy")
        secondary = value.get("secondary_strategy")
        return cls(
            primary_strategy=None if primary is None else str(primary),
            secondary_strategy=None if secondary is None else str(secondary),
            confidence=float(value.get("confidence", 0.0)),
            justification=str(value.get("justification", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, kw_only=True)
class CandidateRecord:
    candidate_id: str
    slot_key: str
    request_hash: str
    problem_id: str
    method: Literal["seed", "qd_mutation"]
    round: int
    parent_candidate_id: str | None
    requested_strategy: str | None
    raw_response: str
    canonical_response: str | None
    finish_reason: str
    input_tokens: int
    output_tokens: int
    verifier_score: float
    fully_correct: bool
    executable: bool
    parse_accepted: bool
    roundtrip_accepted: bool
    roundtrip_token_count: int
    primary_strategy: str | None
    secondary_strategy: str | None
    label_confidence: float | None
    label_justification: str | None
    niche: str | None
    accepted_into_archive: bool
    replaced_candidate_id: str | None
    generator_model: str
    labeler_model: str
    generation_seed: int
    taxonomy_hash: str
    verifier_version: str
    round_trip_gate_version: str
    error: str | None = None
    structural_subtype: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.slot_key or not self.problem_id:
            raise ValueError("candidate identifiers must be non-empty")
        if not math.isfinite(self.verifier_score):
            raise ValueError("candidate verifier score must be finite")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("candidate token counts must be non-negative")
        if self.roundtrip_token_count < 0:
            raise ValueError("round-trip token count must be non-negative")
        if self.fully_correct and not self.executable:
            raise ValueError("a fully correct candidate must be executable")
        from .subtypes import cell_id
        if self.niche != cell_id(self.primary_strategy, self.structural_subtype):
            raise ValueError("niche must match its realized primary strategy and subtype")
        if self.accepted_into_archive and (
            not self.fully_correct
            or not self.executable
            or not self.parse_accepted
            or not self.roundtrip_accepted
            or self.niche is None
        ):
            raise ValueError("an accepted archive candidate must be fully verified")
        if self.label_confidence is not None and not 0 <= self.label_confidence <= 1:
            raise ValueError("candidate label confidence must be in [0, 1]")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateRecord":
        fields = {item.name for item in dataclasses.fields(cls)}
        required = fields - {"error", "structural_subtype"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError("candidate record is missing: " + ", ".join(missing))
        kwargs = {key: value.get(key) for key in fields}
        kwargs.update(
            round=int(value["round"]),
            input_tokens=int(value.get("input_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
            verifier_score=float(value.get("verifier_score", 0.0)),
            fully_correct=bool(value.get("fully_correct", False)),
            executable=bool(value.get("executable", False)),
            parse_accepted=bool(value.get("parse_accepted", False)),
            roundtrip_accepted=bool(value.get("roundtrip_accepted", False)),
            roundtrip_token_count=int(value.get("roundtrip_token_count", 0)),
            accepted_into_archive=bool(value.get("accepted_into_archive", False)),
            generation_seed=int(value["generation_seed"]),
        )
        confidence = value.get("label_confidence")
        kwargs["label_confidence"] = (
            None if confidence is None else float(confidence)
        )
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class SeedObservation:
    slot_key: str
    request: GenerationRequest
    response: GenerationResponse
    parse_accepted: bool
    roundtrip_accepted: bool
    roundtrip_token_count: int
    round_trip_gate_version: str
    canonical_response: str | None
    verifier_score: float
    fully_correct: bool
    executable: bool
    verifier_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SeedObservation":
        request = value["request"]
        return cls(
            slot_key=str(value["slot_key"]),
            request=GenerationRequest.from_dict(request),
            response=GenerationResponse.from_dict(value["response"]),
            parse_accepted=bool(value.get("parse_accepted", False)),
            roundtrip_accepted=bool(value.get("roundtrip_accepted", False)),
            roundtrip_token_count=int(value.get("roundtrip_token_count", 0)),
            round_trip_gate_version=str(value["round_trip_gate_version"]),
            canonical_response=value.get("canonical_response"),
            verifier_score=float(value.get("verifier_score", 0.0)),
            fully_correct=bool(value.get("fully_correct", False)),
            executable=bool(value.get("executable", False)),
            verifier_error=value.get("verifier_error"),
        )


def stable_candidate_id(request_hash: str) -> str:
    """A compact deterministic identifier; reruns never allocate a new UUID."""

    return f"cand_{request_hash[:24]}"


def validate_taxonomy_size(taxonomy: Taxonomy, expected: int) -> None:
    if len(taxonomy.labels) != expected:
        raise ValueError(
            f"labeler returned {len(taxonomy.labels)} labels; expected {expected}"
        )


def validate_problem_ids(items: Sequence[Mapping[str, Any]], field_name: str) -> None:
    ids = [str(item[field_name]) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate problem ids in {field_name!r}")
