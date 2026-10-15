"""Exact serialize/tokenize/decode gate for verified EliteReplay artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .schema import content_hash


ROUND_TRIP_GATE_SCHEMA_VERSION = "EliteReplay.tokenizer_roundtrip.v1"


class TokenizerRoundTripInfrastructureError(RuntimeError):
    """The pinned tokenizer could not be loaded or applied reliably."""


@dataclass(frozen=True)
class RoundTripResult:
    accepted: bool
    token_count: int
    error: str | None = None

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("round-trip token count cannot be negative")
        if self.accepted and self.error is not None:
            raise ValueError("accepted round trip cannot contain an error")
        if not self.accepted and not self.error:
            raise ValueError("rejected round trip must contain an error")


class TokenizerRoundTripGate(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def descriptor(self) -> Mapping[str, Any]: ...

    def check(self, canonical_response: str) -> RoundTripResult: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_tokenizer_assets(path: Path) -> dict[str, str]:
    """Fingerprint tokenizer metadata without hashing multi-GB model weights."""

    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "tokenizer.model",
        "config.json",
    )
    return {
        name: _sha256_file(path / name)
        for name in names
        if (path / name).is_file()
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class HFExactTokenizerRoundTripGate:
    """Require the pinned Hugging Face tokenizer to decode the exact input text."""

    def __init__(
        self,
        *,
        tokenizer_source: str,
        revision: str | None,
        source_origin: str,
        trust_remote_code: bool = True,
    ) -> None:
        source = str(tokenizer_source).strip()
        if not source:
            raise ValueError("tokenizer source must be non-empty")
        local = Path(source).expanduser()
        if source_origin == "ELITEREPLAY_TOKENIZER_PATH":
            if not local.exists():
                raise FileNotFoundError(
                    f"ELITEREPLAY_TOKENIZER_PATH does not exist: {local}"
                )
            source = str(local.resolve())
            revision = None
        self.tokenizer_source = source
        self.revision = revision
        self.source_origin = source_origin
        self.trust_remote_code = bool(trust_remote_code)
        local_source = Path(source)
        assets = (
            _local_tokenizer_assets(local_source)
            if local_source.is_dir()
            else {}
        )
        self._descriptor = {
            "schema_version": ROUND_TRIP_GATE_SCHEMA_VERSION,
            "implementation": "hf_exact_encode_decode",
            "tokenizer_source": source,
            "tokenizer_revision": revision,
            "source_origin": source_origin,
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
            "local_asset_sha256": assets,
            "transformers_version": _package_version("transformers"),
            "tokenizers_version": _package_version("tokenizers"),
        }
        self._version = (
            f"{ROUND_TRIP_GATE_SCHEMA_VERSION}:"
            f"{content_hash(self._descriptor)}"
        )
        self._tokenizer: Any | None = None
        self._load_lock = threading.Lock()
        self._codec_lock = threading.Lock()

    @property
    def version(self) -> str:
        return self._version

    @property
    def descriptor(self) -> Mapping[str, Any]:
        return dict(self._descriptor)

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._load_lock:
            if self._tokenizer is not None:
                return self._tokenizer
            try:
                from transformers import AutoTokenizer

                kwargs: dict[str, Any] = {
                    "trust_remote_code": self.trust_remote_code,
                }
                if self.revision is not None:
                    kwargs["revision"] = self.revision
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_source, **kwargs
                )
            except Exception as exc:
                raise TokenizerRoundTripInfrastructureError(
                    "failed to load pinned tokenizer "
                    f"{self.tokenizer_source!r} revision={self.revision!r}: {exc}"
                ) from exc
        return self._tokenizer

    def check(self, canonical_response: str) -> RoundTripResult:
        if not isinstance(canonical_response, str):
            raise TypeError("canonical response must be a string")
        tokenizer = self._get_tokenizer()
        try:
            with self._codec_lock:
                token_ids = tokenizer.encode(
                    canonical_response, add_special_tokens=False
                )
                if hasattr(token_ids, "tolist"):
                    token_ids = token_ids.tolist()
                token_ids = [int(token_id) for token_id in token_ids]
                decoded = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
        except Exception as exc:
            raise TokenizerRoundTripInfrastructureError(
                f"pinned tokenizer failed during exact round-trip check: {exc}"
            ) from exc
        if not token_ids and canonical_response:
            return RoundTripResult(
                accepted=False,
                token_count=0,
                error="tokenizer_produced_empty_sequence",
            )
        if decoded != canonical_response:
            return RoundTripResult(
                accepted=False,
                token_count=len(token_ids),
                error="serialize_tokenize_decode_mismatch",
            )
        return RoundTripResult(accepted=True, token_count=len(token_ids))


class FunctionalTokenizerRoundTripGate:
    """Lightweight injectable gate for unit tests and audited custom tokenizers."""

    def __init__(
        self,
        round_trip: Callable[[str], str],
        *,
        name: str = "functional_exact",
    ) -> None:
        self._round_trip = round_trip
        self._descriptor = {
            "schema_version": ROUND_TRIP_GATE_SCHEMA_VERSION,
            "implementation": name,
        }
        self._version = (
            f"{ROUND_TRIP_GATE_SCHEMA_VERSION}:"
            f"{content_hash(self._descriptor)}"
        )

    @property
    def version(self) -> str:
        return self._version

    @property
    def descriptor(self) -> Mapping[str, Any]:
        return dict(self._descriptor)

    def check(self, canonical_response: str) -> RoundTripResult:
        decoded = self._round_trip(canonical_response)
        if not isinstance(decoded, str):
            raise TokenizerRoundTripInfrastructureError(
                "functional tokenizer gate must return a string"
            )
        if decoded != canonical_response:
            return RoundTripResult(
                accepted=False,
                token_count=len(canonical_response),
                error="serialize_tokenize_decode_mismatch",
            )
        return RoundTripResult(
            accepted=True,
            token_count=len(canonical_response),
        )


def build_round_trip_gate(config: Any) -> HFExactTokenizerRoundTripGate:
    """Resolve the production tokenizer path without silently changing revisions."""

    environment_path = os.environ.get("ELITEREPLAY_TOKENIZER_PATH")
    models = config.raw_config.get("models", {})
    if environment_path:
        return HFExactTokenizerRoundTripGate(
            tokenizer_source=environment_path,
            revision=None,
            source_origin="ELITEREPLAY_TOKENIZER_PATH",
        )
    revision_value = models.get("tokenizer_revision")
    revision = None if revision_value is None else str(revision_value)
    if revision is None:
        raise ValueError(
            "models.tokenizer_revision is required when "
            "ELITEREPLAY_TOKENIZER_PATH is unset"
        )
    return HFExactTokenizerRoundTripGate(
        tokenizer_source=config.tokenizer,
        revision=revision,
        source_origin="config.models.tokenizer+revision",
    )
