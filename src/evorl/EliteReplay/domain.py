"""Shared domain contracts for offline EliteReplay candidate verification.

The offline search layer deliberately passes only canonical, strictly parsed
artifacts to a verifier.  Keeping parsing and verification as separate typed
results makes that boundary explicit in candidate logs and prevents a lenient
reward extractor from blessing text that would later be exported to SFT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


class DomainInfrastructureError(RuntimeError):
    """A missing or malformed verifier asset that must stop the pipeline."""


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Result of applying a domain's exact output contract."""

    accepted: bool
    canonical_response: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.accepted:
            if not self.canonical_response:
                raise ValueError("an accepted parse must contain a canonical response")
            if self.error is not None:
                raise ValueError("an accepted parse cannot contain an error")
        elif self.canonical_response is not None:
            raise ValueError("a rejected parse cannot contain a canonical response")

    @classmethod
    def accept(cls, canonical_response: str) -> "ParseResult":
        return cls(accepted=True, canonical_response=canonical_response)

    @classmethod
    def reject(cls, error: str) -> "ParseResult":
        return cls(accepted=False, error=error)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Grounded verifier result shared by every EliteReplay domain."""

    score: float
    fully_correct: bool
    executable: bool
    per_test_scores: tuple[float, ...] = ()
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"verification score must be in [0, 1], got {self.score}")
        if any(value not in (0.0, 1.0) for value in self.per_test_scores):
            raise ValueError("per-test scores must be binary")
        if self.fully_correct and not self.executable:
            raise ValueError("a fully correct artifact must be executable")
        if self.fully_correct and self.score != 1.0:
            raise ValueError("a fully correct artifact must have score 1")

    @classmethod
    def rejected(cls, error: str) -> "VerificationResult":
        return cls(score=0.0, fully_correct=False, executable=False, error=error)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """The parse and verifier records for one raw model response."""

    raw_response: str
    parse: ParseResult
    verification: VerificationResult

    @property
    def canonical_response(self) -> str | None:
        return self.parse.canonical_response


@runtime_checkable
class DomainAdapter(Protocol):
    """Minimal interface consumed by QD search, export, and evaluation."""

    verifier_version: str

    def format_prompt(self, item: Any) -> list[dict[str, str]]: ...

    def parse_strict(self, raw_response: str) -> ParseResult: ...

    def verify(self, item: Any, canonical_response: str) -> VerificationResult: ...

    def format_labeler_input(self, item: Any, canonical_response: str) -> str: ...

    def sanitize_for_sft(self, canonical_response: str) -> str: ...


_FORBIDDEN_ENVELOPE = re.compile(
    r"```|</?think\b|<\|(?:im_start|im_end|assistant|user|system|endoftext)\|>"
    r"|^\s*(?:assistant|system)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


def strict_envelope_error(
    raw_response: str,
    *,
    max_response_bytes: int,
) -> str | None:
    """Return a shared strict-format error without extracting a substring."""

    if not isinstance(raw_response, str):
        return "response_not_string"
    if not raw_response.strip():
        return "empty_response"
    if "\x00" in raw_response:
        return "nul_byte"
    if len(raw_response.encode("utf-8")) > max_response_bytes:
        return "response_too_large"
    if _FORBIDDEN_ENVELOPE.search(raw_response):
        return "forbidden_wrapper_or_role_marker"
    if _has_pathological_repetition(raw_response):
        return "pathological_repetition"
    return None


def _has_pathological_repetition(text: str) -> bool:
    """Catch obvious degeneration without rejecting normal repeated syntax."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    run = 1
    for previous, current in zip(lines, lines[1:]):
        run = run + 1 if current == previous and len(current) >= 4 else 1
        if run >= 6:
            return True

    tokens = re.findall(r"\S+", text)
    run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous and len(current) >= 2 else 1
        if run >= 16:
            return True

    # Repeated multi-token loops are common in collapsed reasoning traces.
    for width in (2, 4, 8):
        repeats = 0
        previous: tuple[str, ...] | None = None
        for offset in range(0, len(tokens) - width + 1, width):
            block = tuple(tokens[offset : offset + width])
            repeats = repeats + 1 if block == previous else 1
            previous = block
            if repeats >= 8:
                return True
    return False


def item_value(item: Any, key: str, default: Any = None) -> Any:
    """Read a field from dict-like, pandas-like, or attribute-backed rows."""

    if isinstance(item, Mapping):
        return item.get(key, default)
    try:
        return item[key]
    except (KeyError, IndexError, TypeError):
        return getattr(item, key, default)


def normalized_messages(value: Any) -> list[dict[str, str]]:
    """Copy a parquet/message value into plain role/content dictionaries."""

    if value is None:
        raise DomainInfrastructureError("item is missing prompt messages")
    messages: list[dict[str, str]] = []
    for message in list(value):
        if isinstance(message, Mapping):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if role not in {"system", "user", "assistant"} or not isinstance(
            content, str
        ):
            raise DomainInfrastructureError("item contains malformed prompt messages")
        messages.append({"role": str(role), "content": content})
    if not messages:
        raise DomainInfrastructureError("item has an empty prompt")
    return messages
