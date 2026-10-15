"""Pluggable generation/labeling providers with deterministic request caching."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

from .schema import (
    DecodingConfig,
    GenerationRequest,
    GenerationResponse,
    LabelDecision,
    StrategyLabel,
    Taxonomy,
    canonical_json,
    content_hash,
)


class SolutionGenerator(Protocol):
    """Starting-policy generator. Implementations may use vLLM or an API."""

    @property
    def model_id(self) -> str: ...

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class BatchSolutionGenerator(SolutionGenerator, Protocol):
    """Optional high-throughput extension used for independent search slots."""

    def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResponse]: ...


class StrategyLabeler(Protocol):
    """Fixed external model used only to create strategy metadata."""

    @property
    def model_id(self) -> str: ...

    def describe(
        self, *, problem_id: str, labeler_input: str, seed: int
    ) -> str: ...

    def consolidate(
        self,
        *,
        domain: str,
        descriptions: Sequence[str],
        size: int,
        seed: int,
    ) -> Sequence[StrategyLabel]: ...

    def classify(
        self,
        *,
        problem_id: str,
        labeler_input: str,
        taxonomy: Taxonomy,
        seed: int,
        deterministic: bool,
    ) -> LabelDecision: ...

    def plan_feasible(
        self,
        *,
        problem_id: str,
        problem_context: str,
        taxonomy: Taxonomy,
        observed: Sequence[str],
        limit: int,
        seed: int,
    ) -> Sequence[str]: ...


class ChatCompletionProvider(Protocol):
    """Small common denominator that vLLM and Bedrock adapters can implement."""

    @property
    def model_id(self) -> str: ...

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        decoding: DecodingConfig,
        seed: int,
    ) -> GenerationResponse: ...


class ChatSolutionGenerator:
    def __init__(self, provider: ChatCompletionProvider):
        self._provider = provider

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    @property
    def max_prompt_tokens(self) -> int:
        return self._provider.max_prompt_tokens

    def prompt_token_count(self, messages) -> int:
        return self._provider.prompt_token_count(messages)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return self._provider.complete(
            messages=request.messages,
            decoding=request.decoding,
            seed=request.generation_seed,
        )

    def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResponse]:
        batch_method = getattr(self._provider, "complete_batch", None)
        if callable(batch_method):
            calls = [
                {
                    "messages": request.messages,
                    "decoding": request.decoding,
                    "seed": request.generation_seed,
                }
                for request in requests
            ]
            return tuple(batch_method(calls))
        return tuple(self.generate(request) for request in requests)


T = TypeVar("T")


class JsonRequestCache:
    """One immutable JSON object per deterministic request hash.

    Files are intentionally independent: an interrupted process can lose at
    most the in-flight request, and concurrent readers never observe a partial
    JSON document.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get_or_compute(
        self,
        namespace: str,
        request: Mapping[str, Any],
        compute: Callable[[], T],
        encode: Callable[[T], Any],
        decode: Callable[[Any], T],
    ) -> T:
        cached = self.get(namespace, request, decode)
        if cached is not None:
            return cached

        value = compute()
        return self.put(namespace, request, value, encode, decode)

    def get(
        self,
        namespace: str,
        request: Mapping[str, Any],
        decode: Callable[[Any], T],
    ) -> T | None:
        request_body, request_hash, path = self._location(namespace, request)
        if not path.exists():
            return None
        return decode(self._read_checked(path, request_hash, request_body))

    def put(
        self,
        namespace: str,
        request: Mapping[str, Any],
        value: T,
        encode: Callable[[T], Any],
        decode: Callable[[Any], T],
    ) -> T:
        request_body, request_hash, path = self._location(namespace, request)
        document = {
            "schema_version": "EliteReplay.request_cache.v1",
            "request_hash": request_hash,
            **request_body,
            "response": encode(value),
        }
        self._write_once(path, document)
        # Another writer may have won os.replace. Always return the persisted
        # value so a caller sees one authoritative result.
        return decode(self._read_checked(path, request_hash, request_body))

    def _location(
        self, namespace: str, request: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str, Path]:
        request_body = {"namespace": namespace, "request": request}
        request_hash = content_hash(request_body)
        path = self.root / namespace / f"{request_hash}.json"
        return request_body, request_hash, path

    @staticmethod
    def _read_checked(
        path: Path, request_hash: str, request_body: Mapping[str, Any]
    ) -> Any:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("request_hash") != request_hash:
            raise ValueError(f"cache key mismatch in {path}")
        actual_body = {
            "namespace": document.get("namespace"),
            "request": document.get("request"),
        }
        if canonical_json(actual_body) != canonical_json(request_body):
            raise ValueError(f"cache request payload mismatch in {path}")
        return document["response"]

    @staticmethod
    def _write_once(path: Path, document: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = canonical_json(document) + "\n"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                return
            os.replace(temporary, path)
            temporary = ""
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass


class CachedSolutionGenerator:
    def __init__(self, wrapped: SolutionGenerator, cache: JsonRequestCache):
        self._wrapped = wrapped
        self._cache = cache

    @property
    def model_id(self) -> str:
        return self._wrapped.model_id

    @property
    def max_prompt_tokens(self) -> int:
        return self._wrapped.max_prompt_tokens

    def prompt_token_count(self, messages) -> int:
        return self._wrapped.prompt_token_count(messages)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = {"model": self.model_id, "generation": request.to_dict()}
        return self._cache.get_or_compute(
            "generator",
            payload,
            lambda: self._wrapped.generate(request),
            lambda response: response.to_dict(),
            GenerationResponse.from_dict,
        )

    def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResponse]:
        results: list[GenerationResponse | None] = [None] * len(requests)
        pending_indexes: list[int] = []
        payloads: list[dict[str, Any]] = []
        for index, request in enumerate(requests):
            payload = {"model": self.model_id, "generation": request.to_dict()}
            cached = self._cache.get(
                "generator", payload, GenerationResponse.from_dict
            )
            if cached is None:
                pending_indexes.append(index)
                payloads.append(payload)
            else:
                results[index] = cached

        pending_requests = [requests[index] for index in pending_indexes]
        if not pending_requests:
            return tuple(result for result in results if result is not None)
        batch_method = getattr(self._wrapped, "generate_batch", None)
        if callable(batch_method):
            generated = tuple(batch_method(pending_requests))
        else:
            generated = tuple(
                self._wrapped.generate(request) for request in pending_requests
            )
        if len(generated) != len(pending_requests):
            raise ValueError("generator batch returned the wrong number of responses")

        for index, payload, response in zip(
            pending_indexes, payloads, generated, strict=True
        ):
            results[index] = self._cache.put(
                "generator",
                payload,
                response,
                lambda item: item.to_dict(),
                GenerationResponse.from_dict,
            )
        if any(result is None for result in results):  # pragma: no cover - invariant
            raise RuntimeError("generator batch left an unresolved response")
        return tuple(result for result in results if result is not None)


class CachedStrategyLabeler:
    def __init__(self, wrapped: StrategyLabeler, cache: JsonRequestCache):
        self._wrapped = wrapped
        self._cache = cache

    @property
    def model_id(self) -> str:
        return self._wrapped.model_id

    def describe(self, *, problem_id: str, labeler_input: str, seed: int) -> str:
        request = {
            "model": self.model_id,
            "problem_id": problem_id,
            "labeler_input": labeler_input,
            "seed": seed,
        }
        return self._cache.get_or_compute(
            "label_describe", request,
            lambda: self._wrapped.describe(
                problem_id=problem_id, labeler_input=labeler_input, seed=seed
            ),
            str,
            str,
        )

    def describe_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[str]:
        return self._batch_cached(
            "label_describe", requests, str, str, "describe_batch", "describe"
        )

    def consolidate(
        self,
        *,
        domain: str,
        descriptions: Sequence[str],
        size: int,
        seed: int,
    ) -> Sequence[StrategyLabel]:
        request = {
            "model": self.model_id,
            "domain": domain,
            "descriptions": list(descriptions),
            "size": size,
            "seed": seed,
        }
        return self._cache.get_or_compute(
            "label_consolidate",
            request,
            lambda: tuple(
                self._wrapped.consolidate(
                    domain=domain,
                    descriptions=descriptions,
                    size=size,
                    seed=seed,
                )
            ),
            lambda labels: [label.to_dict() for label in labels],
            lambda values: tuple(StrategyLabel.from_dict(v) for v in values),
        )

    def classify(
        self,
        *,
        problem_id: str,
        labeler_input: str,
        taxonomy: Taxonomy,
        seed: int,
        deterministic: bool,
    ) -> LabelDecision:
        request = {
            "model": self.model_id,
            "problem_id": problem_id,
            "labeler_input": labeler_input,
            "taxonomy_hash": taxonomy.hash,
            "taxonomy": taxonomy.to_dict(),
            "seed": seed,
            "deterministic": deterministic,
        }
        return self._cache.get_or_compute(
            "label_classify",
            request,
            lambda: self._wrapped.classify(
                problem_id=problem_id,
                labeler_input=labeler_input,
                taxonomy=taxonomy,
                seed=seed,
                deterministic=deterministic,
            ),
            lambda decision: decision.to_dict(),
            LabelDecision.from_dict,
        )

    def classify_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[LabelDecision]:
        return self._batch_cached(
            "label_classify",
            requests,
            lambda decision: decision.to_dict(),
            LabelDecision.from_dict,
            "classify_batch",
            "classify",
        )

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
        request = {
            "model": self.model_id,
            "problem_id": problem_id,
            "problem_context": problem_context,
            "taxonomy_hash": taxonomy.hash,
            "taxonomy": taxonomy.to_dict(),
            "observed": list(observed),
            "limit": limit,
            "seed": seed,
        }
        return self._cache.get_or_compute(
            "label_feasible",
            request,
            lambda: tuple(
                self._wrapped.plan_feasible(
                    problem_id=problem_id,
                    problem_context=problem_context,
                    taxonomy=taxonomy,
                    observed=observed,
                    limit=limit,
                    seed=seed,
                )
            ),
            list,
            lambda values: tuple(str(value) for value in values),
        )

    def plan_feasible_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[Sequence[str]]:
        return self._batch_cached(
            "label_feasible",
            requests,
            list,
            lambda values: tuple(str(value) for value in values),
            "plan_feasible_batch",
            "plan_feasible",
        )

    def _batch_cached(
        self,
        namespace: str,
        calls: Sequence[Mapping[str, Any]],
        encode: Callable[[Any], Any],
        decode: Callable[[Any], Any],
        batch_method_name: str,
        scalar_method_name: str,
    ) -> Sequence[Any]:
        """Cache a batch of keyword calls without defeating provider batching."""

        results: list[Any | None] = [None] * len(calls)
        pending_indexes: list[int] = []
        pending_calls: list[dict[str, Any]] = []
        payloads: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            materialized = dict(call)
            payload = self._label_payload(namespace, materialized)
            cached = self._cache.get(namespace, payload, decode)
            if cached is None:
                pending_indexes.append(index)
                pending_calls.append(materialized)
                payloads.append(payload)
            else:
                results[index] = cached

        if not pending_calls:
            return tuple(result for result in results if result is not None)
        batch_method = getattr(self._wrapped, batch_method_name, None)
        try:
            if callable(batch_method):
                generated = tuple(batch_method(pending_calls))
            else:
                scalar = getattr(self._wrapped, scalar_method_name)
                generated = tuple(scalar(**call) for call in pending_calls)
        except MetadataBatchExhaustedError as exc:
            # Persist every independently valid reply before surfacing the
            # failed item. A resumed batch then contacts Bedrock only for the
            # unresolved logical requests.
            for pending_index, result in exc.partial_results.items():
                if not 0 <= pending_index < len(pending_indexes):
                    raise RuntimeError(
                        "metadata retry returned an invalid partial-result index"
                    ) from exc
                output_index = pending_indexes[pending_index]
                results[output_index] = self._cache.put(
                    namespace,
                    payloads[pending_index],
                    result,
                    encode,
                    decode,
                )
            mapped_failed = tuple(
                pending_indexes[pending_index]
                for pending_index in exc.failed_indexes
            )
            mapped_results = {
                index: result
                for index, result in enumerate(results)
                if result is not None
            }
            raise MetadataBatchExhaustedError(
                str(exc),
                partial_results=mapped_results,
                failed_indexes=mapped_failed,
                batch_size=len(calls),
                structural_failure=exc.structural_failure,
            ) from exc
        if len(generated) != len(pending_calls):
            raise ValueError(
                f"{batch_method_name} returned the wrong number of responses"
            )
        for index, payload, result in zip(
            pending_indexes, payloads, generated, strict=True
        ):
            results[index] = self._cache.put(
                namespace, payload, result, encode, decode
            )
        if any(result is None for result in results):  # pragma: no cover - invariant
            raise RuntimeError(f"{batch_method_name} left an unresolved response")
        return tuple(result for result in results if result is not None)

    def _label_payload(
        self, namespace: str, call: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload = {"model": self.model_id, **dict(call)}
        taxonomy = payload.get("taxonomy")
        if isinstance(taxonomy, Taxonomy):
            payload["taxonomy_hash"] = taxonomy.hash
            payload["taxonomy"] = taxonomy.to_dict()
        return payload


def _decode_json_object(text: str) -> Mapping[str, Any]:
    """Decode the first JSON object from a metadata-only labeler response."""

    start = text.find("{")
    if start < 0:
        raise ValueError("labeler response did not contain a JSON object")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, Mapping):
        raise ValueError("labeler response must be a JSON object")
    return value


class MetadataResponseError(RuntimeError):
    """A metadata-only labeler reply remained invalid after bounded retries."""


class MetadataBatchExhaustedError(MetadataResponseError):
    """Batch failure carrying per-item state for cache commit/recovery.

    ``structural_failure`` is true only when every failed index exhausted its
    bounded JSON/schema retries. Provider and infrastructure failures remain
    fatal even when other items in the same batch already produced valid
    metadata.
    """

    def __init__(
        self,
        message: str,
        *,
        partial_results: Mapping[int, Any],
        failed_indexes: Sequence[int],
        batch_size: int,
        structural_failure: bool,
    ) -> None:
        super().__init__(message)
        self.partial_results = dict(partial_results)
        self.failed_indexes = tuple(int(index) for index in failed_indexes)
        self.batch_size = int(batch_size)
        self.structural_failure = bool(structural_failure)
        covered = set(self.partial_results) | set(self.failed_indexes)
        if self.batch_size < 0 or covered != set(range(self.batch_size)):
            raise ValueError(
                "metadata batch result indexes must partition the full batch"
            )
        if set(self.partial_results) & set(self.failed_indexes):
            raise ValueError("metadata batch result indexes overlap")


def _safe_metadata_error(error: Exception) -> str:
    """Describe structure errors without reproducing model/task response text."""

    if isinstance(error, json.JSONDecodeError):
        return f"invalid_json_line_{error.lineno}_column_{error.colno}"
    if isinstance(error, KeyError):
        return "missing_required_field"
    if isinstance(error, TypeError):
        return "invalid_field_type"
    return "invalid_schema"


class PromptedStrategyLabeler:
    """Reference labeler prompts over any chat-completion provider.

    This class never asks for a solution.  It sends only metadata requests and
    is suitable for a fixed Bedrock model once a ChatCompletionProvider is
    supplied by the launcher.
    """

    _SYSTEM = (
        "You are a strategy taxonomy annotator. Never solve the task or emit a "
        "replacement answer. Return only the requested JSON metadata. Describe "
        "computational/query structure, never wording or formatting."
    )
    _RETRY_SUFFIX = (
        "\n\nThe preceding attempt for this same request was rejected because "
        "its metadata was not valid. Retry now with exactly one JSON object, "
        "no Markdown fence, no prose before or after it, and the exact keys and "
        "value types requested above."
    )

    def __init__(
        self,
        provider: ChatCompletionProvider,
        *,
        max_tokens: int = 4096,
        metadata_attempts: int | None = None,
    ):
        self._provider = provider
        self._max_tokens = max_tokens
        attempts = (
            getattr(provider, "attempts", 1)
            if metadata_attempts is None
            else metadata_attempts
        )
        self._metadata_attempts = int(attempts)
        if self._metadata_attempts <= 0:
            raise ValueError("metadata_attempts must be positive")

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    def _complete(self, prompt: str, *, seed: int, deterministic: bool) -> str:
        response = self._provider.complete(
            messages=(
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": prompt},
            ),
            decoding=DecodingConfig(
                temperature=0.0 if deterministic else 0.2,
                top_p=1.0,
                max_tokens=self._max_tokens,
            ),
            seed=seed,
        )
        return response.raw_response

    def _complete_batch(
        self, prompts: Sequence[str], seeds: Sequence[int], *, deterministic: bool
    ) -> tuple[str, ...]:
        calls = [
            {
                "messages": (
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": prompt},
                ),
                "decoding": DecodingConfig(
                    temperature=0.0 if deterministic else 0.2,
                    top_p=1.0,
                    max_tokens=self._max_tokens,
                ),
                "seed": seed,
            }
            for prompt, seed in zip(prompts, seeds, strict=True)
        ]
        batch_method = getattr(self._provider, "complete_batch", None)
        if callable(batch_method):
            responses = tuple(batch_method(calls))
        else:
            responses = tuple(self._provider.complete(**call) for call in calls)
        if len(responses) != len(calls):
            raise ValueError("labeler chat provider returned wrong batch cardinality")
        return tuple(response.raw_response for response in responses)

    def _parse_with_retries(
        self,
        *,
        prompt: str,
        seed: int,
        deterministic: bool,
        parser: Callable[[str], T],
        operation: str,
        context: str,
    ) -> T:
        last_error = "unattempted"
        for attempt in range(self._metadata_attempts):
            text = self._complete(
                self._retry_prompt(prompt, attempt),
                seed=seed,
                # A deterministic malformed reply can otherwise be reproduced
                # verbatim. The first request remains exactly as configured;
                # only repair attempts use sampling plus a stricter prompt.
                deterministic=deterministic if attempt == 0 else False,
            )
            try:
                return parser(text)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                last_error = _safe_metadata_error(error)
        request_ref = content_hash({"operation": operation, "context": context})[:12]
        raise MetadataResponseError(
            f"{operation} metadata response invalid after "
            f"{self._metadata_attempts} attempts "
            f"(request={request_ref}, last_error={last_error})"
        )

    def _parse_batch_with_retries(
        self,
        *,
        prompts: Sequence[str],
        seeds: Sequence[int],
        deterministic: bool,
        parser: Callable[[str], T],
        operation: str,
        contexts: Sequence[str],
    ) -> tuple[T, ...]:
        if not (len(prompts) == len(seeds) == len(contexts)):
            raise ValueError("metadata batch inputs have different cardinalities")
        results: dict[int, T] = {}
        pending = list(range(len(prompts)))
        last_errors: dict[int, str] = {}
        for attempt in range(self._metadata_attempts):
            if not pending:
                break
            try:
                texts = self._complete_batch(
                    [
                        self._retry_prompt(prompts[index], attempt)
                        for index in pending
                    ],
                    [seeds[index] for index in pending],
                    deterministic=deterministic if attempt == 0 else False,
                )
            except Exception as error:
                if not results:
                    raise
                raise MetadataBatchExhaustedError(
                    f"{operation} metadata provider failed while retrying "
                    f"{len(pending)} request(s); valid peer replies were preserved",
                    partial_results=results,
                    failed_indexes=pending,
                    batch_size=len(prompts),
                    structural_failure=False,
                ) from error

            retry: list[int] = []
            for index, text in zip(pending, texts, strict=True):
                try:
                    results[index] = parser(text)
                except (
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as error:
                    last_errors[index] = _safe_metadata_error(error)
                    retry.append(index)
            pending = retry

        if pending:
            references = [
                content_hash(
                    {"operation": operation, "context": contexts[index]}
                )[:12]
                for index in pending
            ]
            error_kinds = sorted({last_errors[index] for index in pending})
            raise MetadataBatchExhaustedError(
                f"{operation} metadata response invalid after "
                f"{self._metadata_attempts} attempts for {len(pending)} "
                f"request(s) (requests={references}, errors={error_kinds})",
                partial_results=results,
                failed_indexes=pending,
                batch_size=len(prompts),
                structural_failure=True,
            )
        return tuple(results[index] for index in range(len(prompts)))

    @classmethod
    def _retry_prompt(cls, prompt: str, attempt: int) -> str:
        """Keep the initial request unchanged and harden only repair attempts."""

        if attempt == 0:
            return prompt
        return f"{prompt}{cls._RETRY_SUFFIX} Retry attempt: {attempt}."

    @staticmethod
    def _describe_prompt(problem_id: str, labeler_input: str) -> str:
        return (
            "Describe the strategy used by this verified solution in one concise "
            "sentence. Return JSON as {\"strategy_description\": \"...\"}.\n\n"
            f"Problem id: {problem_id}\n{labeler_input}"
        )

    def describe(self, *, problem_id: str, labeler_input: str, seed: int) -> str:
        prompt = self._describe_prompt(problem_id, labeler_input)

        def parse(text: str) -> str:
            value = _decode_json_object(text)
            description = value.get("strategy_description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("invalid strategy description")
            return description.strip()

        return self._parse_with_retries(
            prompt=prompt,
            seed=seed,
            deterministic=True,
            parser=parse,
            operation="describe",
            context=problem_id,
        )

    def describe_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[str]:
        prompts = [
            self._describe_prompt(str(call["problem_id"]), str(call["labeler_input"]))
            for call in requests
        ]

        def parse(text: str) -> str:
            value = _decode_json_object(text)
            description = value.get("strategy_description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("invalid strategy description")
            return description.strip()

        return self._parse_batch_with_retries(
            prompts=prompts,
            seeds=[int(call["seed"]) for call in requests],
            deterministic=True,
            parser=parse,
            operation="describe",
            contexts=[
                canonical_json(
                    {
                        "problem_id": str(call["problem_id"]),
                        "seed": int(call["seed"]),
                    }
                )
                for call in requests
            ],
        )

    def consolidate(
        self,
        *,
        domain: str,
        descriptions: Sequence[str],
        size: int,
        seed: int,
    ) -> Sequence[StrategyLabel]:
        prompt = (
            f"Merge the following verified {domain} strategy descriptions into "
            f"exactly {size} reusable domain-level labels. Return JSON object "
            "{\"labels\": [{\"id\": \"stable_snake_case\", \"name\": \"...\", "
            "\"definition\": \"...\", \"positive_cues\": [\"...\"], "
            "\"negative_cues\": [\"...\"]}]}. IDs must be unique.\n\n"
            + "\n".join(f"- {item}" for item in descriptions)
        )

        def parse(text: str) -> tuple[StrategyLabel, ...]:
            value = _decode_json_object(text)
            labels = value.get("labels")
            if not isinstance(labels, list) or len(labels) != size:
                raise ValueError("taxonomy response has invalid label count")
            parsed = tuple(StrategyLabel.from_dict(label) for label in labels)
            if len({label.id for label in parsed}) != len(parsed):
                raise ValueError("taxonomy response has duplicate ids")
            return parsed

        return self._parse_with_retries(
            prompt=prompt,
            seed=seed,
            deterministic=True,
            parser=parse,
            operation="consolidate",
            context=f"{domain}:{size}",
        )

    def classify(
        self,
        *,
        problem_id: str,
        labeler_input: str,
        taxonomy: Taxonomy,
        seed: int,
        deterministic: bool,
    ) -> LabelDecision:
        prompt = self._classify_prompt(problem_id, labeler_input, taxonomy)
        return self._parse_with_retries(
            prompt=prompt,
            seed=seed,
            deterministic=deterministic,
            parser=lambda text: self._parse_classification(text, taxonomy),
            operation="classify",
            context=problem_id,
        )

    @staticmethod
    def _parse_classification(text: str, taxonomy: Taxonomy) -> LabelDecision:
        value = _decode_json_object(text)
        required = {"primary_strategy", "confidence", "justification"}
        if not required.issubset(value):
            raise KeyError("classification field")
        if not isinstance(value["primary_strategy"], str) and not (taxonomy.domain == "optmath" and value["primary_strategy"] is None):
            raise TypeError("primary_strategy")
        if not isinstance(value["justification"], str) or not value[
            "justification"
        ].strip():
            raise ValueError("invalid justification")
        decision = LabelDecision.from_dict(value)
        if decision.primary_strategy is not None or taxonomy.domain != "optmath":
            taxonomy.require_label(decision.primary_strategy or "")
        if decision.secondary_strategy is not None:
            taxonomy.require_label(decision.secondary_strategy)
        return decision

    @staticmethod
    def _classify_prompt(
        problem_id: str, labeler_input: str, taxonomy: Taxonomy
    ) -> str:
        rules = ""
        if taxonomy.domain == "optmath":
            from .optmath_strategies import LABELING_RULES
            rules = LABELING_RULES + "\n\n"
        return (rules +
            "Classify this verified solution using the frozen taxonomy. Return "
            "JSON {\"primary_strategy\": \"id\", \"secondary_strategy\": null, "
            "\"confidence\": 0.0, \"justification\": \"...\"}. The primary "
            + ("label may be null if no formulation is evidenced.\n\nTaxonomy:\n" if taxonomy.domain == "optmath" else "label must be one listed id.\n\nTaxonomy:\n") +
            f"{canonical_json(taxonomy.to_dict())}\n\nProblem id: {problem_id}\n"
            f"{labeler_input}\n\nAllowed strategy IDs (copy exactly; never rename):\n"
            f"{canonical_json([label.id for label in taxonomy.labels])}"
        )

    def classify_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[LabelDecision]:
        if not requests:
            return ()
        taxonomy_hashes = {call["taxonomy"].hash for call in requests}
        if len(taxonomy_hashes) != 1:
            raise ValueError("one labeler batch cannot mix frozen taxonomies")
        prompts = [
            self._classify_prompt(
                str(call["problem_id"]),
                str(call["labeler_input"]),
                call["taxonomy"],
            )
            for call in requests
        ]
        deterministic_values = {bool(call["deterministic"]) for call in requests}
        if len(deterministic_values) > 1:
            raise ValueError("one labeler batch cannot mix decoding modes")
        deterministic = deterministic_values.pop()
        return self._parse_batch_with_retries(
            prompts=prompts,
            seeds=[int(call["seed"]) for call in requests],
            deterministic=deterministic,
            parser=lambda text: self._parse_classification(
                text, requests[0]["taxonomy"]
            ),
            operation="classify",
            contexts=[
                canonical_json(
                    {
                        "problem_id": str(call["problem_id"]),
                        "seed": int(call["seed"]),
                    }
                )
                for call in requests
            ],
        )

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
        prompt = self._feasible_prompt(
            problem_id, problem_context, taxonomy, observed, limit
        )
        return self._parse_with_retries(
            prompt=prompt,
            seed=seed,
            deterministic=True,
            parser=lambda text: self._parse_feasible(
                text, taxonomy=taxonomy, limit=limit
            ),
            operation="plan_feasible",
            context=problem_id,
        )

    @staticmethod
    def _parse_feasible(
        text: str, *, taxonomy: Taxonomy, limit: int
    ) -> tuple[str, ...]:
        value = _decode_json_object(text)
        labels = value.get("strategy_ids")
        if (
            not isinstance(labels, list)
            or (not labels and taxonomy.domain != "optmath")
            or len(labels) > limit
            or any(not isinstance(label, str) for label in labels)
        ):
            raise ValueError("invalid feasible strategy list")
        result = tuple(labels)
        if len(set(result)) != len(result):
            raise ValueError("duplicate feasible strategy ids")
        for label in result:
            taxonomy.require_label(label)
        return result

    @staticmethod
    def _feasible_prompt(
        problem_id: str,
        problem_context: str,
        taxonomy: Taxonomy,
        observed: Sequence[str],
        limit: int,
    ) -> str:
        rules = ""
        if taxonomy.domain == "optmath":
            from .optmath_strategies import LABELING_RULES
            rules = LABELING_RULES + "\nOnly choose formulations soundly applicable to this specific problem.\n"
        return (rules +
            f"Select between {0 if taxonomy.domain == 'optmath' else 1} and {limit} distinct frozen taxonomy IDs that plausibly represent "
            "distinct ways to solve this problem. Return labels only, never a "
            "solution, as JSON {\"strategy_ids\": [\"id\"]}.\n\nTaxonomy:\n"
            f"{canonical_json(taxonomy.to_dict())}\n\nProblem id: {problem_id}\n"
            f"Problem context:\n{problem_context}\nObserved labels: {list(observed)}\n\n"
            "Allowed strategy IDs (copy exactly; never rename or invent IDs):\n"
            f"{canonical_json([label.id for label in taxonomy.labels])}"
        )

    def plan_feasible_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Sequence[Sequence[str]]:
        if not requests:
            return ()
        taxonomy_hashes = {call["taxonomy"].hash for call in requests}
        limits = {int(call["limit"]) for call in requests}
        if len(taxonomy_hashes) != 1 or len(limits) != 1:
            raise ValueError(
                "one feasible-strategy batch cannot mix taxonomies or limits"
            )
        prompts = [
            self._feasible_prompt(
                str(call["problem_id"]),
                str(call["problem_context"]),
                call["taxonomy"],
                call["observed"],
                int(call["limit"]),
            )
            for call in requests
        ]
        # Pipeline batches one frozen taxonomy/limit at a time.
        taxonomy = requests[0]["taxonomy"]
        limit = int(requests[0]["limit"])
        return self._parse_batch_with_retries(
            prompts=prompts,
            seeds=[int(call["seed"]) for call in requests],
            deterministic=True,
            parser=lambda text: self._parse_feasible(
                text, taxonomy=taxonomy, limit=limit
            ),
            operation="plan_feasible",
            contexts=[
                canonical_json(
                    {
                        "problem_id": str(call["problem_id"]),
                        "seed": int(call["seed"]),
                    }
                )
                for call in requests
            ],
        )


class VLLMOpenAIChatProvider:
    """Thread-batched client for a local vLLM OpenAI-compatible server."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str = "unused",
        max_workers: int = 64,
        timeout_seconds: int = 600,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        if not api_base:
            raise ValueError("vLLM api_base is required")
        if max_workers <= 0:
            raise ValueError("vLLM max_workers must be positive")
        self.api_base = api_base.rstrip("/")
        self._model_id = model
        self.api_key = api_key
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.extra_body = dict(extra_body or {})

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        decoding: DecodingConfig,
        seed: int,
    ) -> GenerationResponse:
        body = {
            "model": self.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "max_tokens": decoding.max_tokens,
            "seed": seed,
            **self.extra_body,
        }
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                value = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"vLLM request failed ({exc.code}): {detail}") from exc
        choice = value["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
        usage = value.get("usage", {})
        return GenerationResponse(
            raw_response=str(text),
            finish_reason=str(choice.get("finish_reason") or "other"),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            model=str(value.get("model") or self.model_id),
            metadata={"request_id": value.get("id")},
        )

    def complete_batch(
        self, calls: Sequence[Mapping[str, Any]]
    ) -> Sequence[GenerationResponse]:
        if not calls:
            return ()
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(calls))
        ) as executor:
            futures = [executor.submit(self.complete, **dict(call)) for call in calls]
            # Preserve request order. Exceptions propagate before anything is
            # journaled; successful calls are already protected by request cache.
            return tuple(future.result() for future in futures)


class LiteLLMChatProvider(VLLMOpenAIChatProvider):
    """Hosted strategy annotations through the shared LiteLLM transport."""

    def __init__(self, *, model, api_base=None, max_workers=16, attempts=5,
                 timeout_seconds=600, usage_path=None):
        from evorl.litellm import LiteLLMClient
        self._client = LiteLLMClient(endpoint=api_base, attempts=attempts,
                                    timeout=timeout_seconds, usage_path=usage_path)
        self._model_id = model
        self.max_workers = int(max_workers)
        self.attempts = int(attempts)
        if self.max_workers < 1:
            raise ValueError("LiteLLM max_workers must be positive")

    def complete(self, *, messages, decoding, seed):
        value = self._client.complete(model=self.model_id, messages=messages,
            max_tokens=decoding.max_tokens, temperature=decoding.temperature, top_p=decoding.top_p)
        choice = value["choices"][0]
        usage = value.get("usage", {})
        return GenerationResponse(raw_response=choice["message"]["content"],
            finish_reason=str(choice.get("finish_reason") or "other"),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)), model=self.model_id,
            metadata={"provider": "litellm", "base_url": self._client.base_url,
                      "request_id": value.get("id"), "response_model": value.get("model")})


def make_litellm_labeler(*, config: Any, settings: Mapping[str, Any]) -> StrategyLabeler:
    model = str(settings.get("model") or config.labeler_model)
    provider = LiteLLMChatProvider(model=model, api_base=settings.get("api_base"),
        max_workers=int(settings.get("max_workers", 16)), attempts=int(settings.get("attempts", 5)),
        timeout_seconds=int(settings.get("timeout_seconds", 600)),
        usage_path=settings.get("usage_path") or Path(config.output_dir) / "labeler_usage.jsonl")
    return PromptedStrategyLabeler(provider, max_tokens=int(settings.get("max_tokens", 4096)))


class BedrockConverseChatProvider:
    """AWS Bedrock Converse provider for the fixed external taxonomy labeler.

    AWS credentials use boto3's normal resolution chain (environment, profile,
    or node role). ``AWS_REGION``/``AWS_DEFAULT_REGION`` are honored by the
    factory below.
    """

    def __init__(
        self,
        *,
        model: str,
        region: str = "us-east-1",
        max_workers: int = 32,
        attempts: int = 5,
        timeout_seconds: int = 600,
    ) -> None:
        if max_workers <= 0 or attempts <= 0:
            raise ValueError("Bedrock worker and attempt counts must be positive")
        try:
            import boto3
            from botocore.client import Config
        except ImportError as exc:  # pragma: no cover - production environment
            raise RuntimeError("boto3 is required for the Bedrock labeler") from exc
        self._model_id = model
        self.max_workers = max_workers
        self.attempts = attempts
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                retries={"mode": "standard", "total_max_attempts": 10},
                connect_timeout=30,
                read_timeout=timeout_seconds,
                max_pool_connections=max_workers,
            ),
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        decoding: DecodingConfig,
        seed: int,
    ) -> GenerationResponse:
        del seed  # Converse does not currently expose a portable seed field.
        systems = [
            message["content"] for message in messages if message["role"] == "system"
        ]
        converse_messages = [
            {
                "role": message["role"],
                "content": [{"text": message["content"]}],
            }
            for message in messages
            if message["role"] != "system"
        ]
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                kwargs: dict[str, Any] = {
                    "modelId": self.model_id,
                    "messages": converse_messages,
                    "inferenceConfig": {
                        "maxTokens": min(decoding.max_tokens, 4096),
                        "temperature": decoding.temperature,
                        "topP": decoding.top_p,
                    },
                }
                if systems:
                    kwargs["system"] = [{"text": "\n\n".join(systems)}]
                response = self._client.converse(**kwargs)
                text = "".join(
                    str(block.get("text", ""))
                    for block in response["output"]["message"]["content"]
                    if isinstance(block, Mapping)
                )
                usage = response.get("usage", {})
                stop = str(response.get("stopReason") or "other")
                finish_reason = "length" if stop in {"max_tokens", "length"} else stop
                return GenerationResponse(
                    raw_response=text,
                    finish_reason=finish_reason,
                    input_tokens=int(usage.get("inputTokens", 0)),
                    output_tokens=int(usage.get("outputTokens", 0)),
                    model=self.model_id,
                    metadata={"stop_reason": stop},
                )
            except Exception as exc:  # boto errors differ across versions
                last_error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(min(30.0, (2**attempt) + random.random()))
        raise RuntimeError(
            f"Bedrock Converse failed after {self.attempts} attempts: {last_error}"
        ) from last_error

    def complete_batch(
        self, calls: Sequence[Mapping[str, Any]]
    ) -> Sequence[GenerationResponse]:
        if not calls:
            return ()
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(calls))
        ) as executor:
            futures = [executor.submit(self.complete, **dict(call)) for call in calls]
            return tuple(future.result() for future in futures)


def make_vllm_generator(*, config: Any, settings: Mapping[str, Any]) -> SolutionGenerator:
    """CLI factory for the local Qwen3.5-4B vLLM server."""

    api_base = str(
        settings.get("api_base")
        or os.environ.get("ELITEREPLAY_VLLM_API_BASE", "")
    )
    model = str(
        settings.get("served_model")
        or os.environ.get("ELITEREPLAY_VLLM_MODEL")
        or config.base_model
    )
    provider = VLLMOpenAIChatProvider(
        api_base=api_base,
        model=model,
        api_key=str(settings.get("api_key", "unused")),
        max_workers=int(settings.get("max_workers", 64)),
        timeout_seconds=int(settings.get("timeout_seconds", 600)),
        extra_body=settings.get("extra_body", {}),
    )
    return ChatSolutionGenerator(provider)


def make_bedrock_labeler(*, config: Any, settings: Mapping[str, Any]) -> StrategyLabeler:
    """CLI factory for the existing Qwen3-Coder Bedrock labeler path."""

    model = str(
        settings.get("model")
        or os.environ.get("ELITEREPLAY_BEDROCK_MODEL")
        or config.labeler_model
    )
    region = str(
        settings.get("region")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    provider = BedrockConverseChatProvider(
        model=model,
        region=region,
        max_workers=int(settings.get("max_workers", 32)),
        attempts=int(settings.get("attempts", 5)),
        timeout_seconds=int(settings.get("timeout_seconds", 600)),
    )
    return PromptedStrategyLabeler(
        provider, max_tokens=int(settings.get("max_tokens", 4096))
    )
