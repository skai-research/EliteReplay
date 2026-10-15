from __future__ import annotations

import sys
import types
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from evorl.EliteReplay.providers import (
    BedrockConverseChatProvider,
    CachedStrategyLabeler,
    JsonRequestCache,
    MetadataBatchExhaustedError,
    PromptedStrategyLabeler,
)
from evorl.EliteReplay.schema import GenerationResponse, StrategyLabel, Taxonomy


class ScriptedMetadataProvider:
    model_id = "scripted-bedrock"

    def __init__(
        self, replies: Mapping[str, Sequence[str]], *, attempts: int
    ) -> None:
        self.attempts = attempts
        self.replies = {
            problem_id: deque(values) for problem_id, values in replies.items()
        }
        self.batch_calls: list[list[dict[str, Any]]] = []
        self.problem_call_counts: dict[str, int] = defaultdict(int)

    def complete_batch(
        self, calls: Sequence[Mapping[str, Any]]
    ) -> Sequence[GenerationResponse]:
        materialized = [dict(call) for call in calls]
        self.batch_calls.append(materialized)
        responses = []
        for call in materialized:
            prompt = str(call["messages"][-1]["content"])
            matches = [
                problem_id
                for problem_id in self.replies
                if f"Problem id: {problem_id}\n" in prompt
            ]
            if len(matches) != 1:
                raise AssertionError("test provider could not identify problem")
            problem_id = matches[0]
            self.problem_call_counts[problem_id] += 1
            if not self.replies[problem_id]:
                raise AssertionError(f"unexpected repeated request for {problem_id}")
            responses.append(
                GenerationResponse(
                    raw_response=self.replies[problem_id].popleft(),
                    finish_reason="stop",
                    input_tokens=1,
                    output_tokens=1,
                    model=self.model_id,
                )
            )
        return tuple(responses)


def _describe_calls(*problem_ids: str) -> list[dict[str, Any]]:
    return [
        {
            "problem_id": problem_id,
            "labeler_input": f"private task body for {problem_id}",
            "seed": index + 10,
        }
        for index, problem_id in enumerate(problem_ids)
    ]


def _temperatures(batch: Sequence[Mapping[str, Any]]) -> list[float]:
    return [
        float(call["decoding"].temperature)  # type: ignore[union-attr]
        for call in batch
    ]


def test_batch_retries_only_malformed_index_with_hardened_sampling() -> None:
    provider = ScriptedMetadataProvider(
        {
            "good": ['{"strategy_description":"hash join"}'],
            "repair": [
                "not JSON",
                '{"strategy_description": 42}',
                '{"strategy_description":"window aggregation"}',
            ],
        },
        attempts=3,
    )
    labeler = PromptedStrategyLabeler(provider)

    result = labeler.describe_batch(_describe_calls("good", "repair"))

    assert result == ("hash join", "window aggregation")
    assert [len(batch) for batch in provider.batch_calls] == [2, 1, 1]
    assert provider.problem_call_counts == {"good": 1, "repair": 3}
    assert _temperatures(provider.batch_calls[0]) == [0.0, 0.0]
    assert _temperatures(provider.batch_calls[1]) == [0.2]
    first_prompt = str(provider.batch_calls[0][1]["messages"][-1]["content"])
    retry_prompt = str(provider.batch_calls[1][0]["messages"][-1]["content"])
    assert "preceding attempt" not in first_prompt
    assert "preceding attempt" in retry_prompt


def test_exhaustion_caches_valid_peers_and_redacts_sensitive_text(
    tmp_path: Path,
) -> None:
    secret_problem = "SECRET_TASK_IDENTIFIER"
    provider = ScriptedMetadataProvider(
        {
            "good": ['{"strategy_description":"valid peer"}'],
            secret_problem: [
                "SECRET_RAW_REPLY aws_secret_access_key=do-not-log",
                '{"strategy_description": false}',
                '{"strategy_description":"recovered after resume"}',
            ],
        },
        attempts=2,
    )
    labeler = CachedStrategyLabeler(
        PromptedStrategyLabeler(provider), JsonRequestCache(tmp_path / "cache")
    )
    calls = _describe_calls("good", secret_problem)

    with pytest.raises(MetadataBatchExhaustedError) as captured:
        labeler.describe_batch(calls)

    message = str(captured.value)
    assert "after 2 attempts" in message
    assert secret_problem not in message
    assert "SECRET_RAW_REPLY" not in message
    assert "aws_secret_access_key" not in message
    assert captured.value.structural_failure is True
    assert captured.value.batch_size == 2
    assert captured.value.failed_indexes == (1,)
    assert captured.value.partial_results == {0: "valid peer"}
    assert provider.problem_call_counts == {"good": 1, secret_problem: 2}

    batch_count = len(provider.batch_calls)
    assert labeler.describe_batch(calls) == (
        "valid peer",
        "recovered after resume",
    )
    assert len(provider.batch_calls) == batch_count + 1
    assert len(provider.batch_calls[-1]) == 1
    assert provider.problem_call_counts == {"good": 1, secret_problem: 3}

    completed_batch_count = len(provider.batch_calls)
    assert labeler.describe_batch(calls) == (
        "valid peer",
        "recovered after resume",
    )
    assert len(provider.batch_calls) == completed_batch_count


def test_out_of_taxonomy_classification_is_an_item_local_failure() -> None:
    taxonomy = Taxonomy(
        domain="fake",
        version="v1",
        labeler_model="scripted-bedrock",
        labels=(StrategyLabel("known_strategy", "Known", "Known method."),),
    )
    valid = (
        '{"primary_strategy":"known_strategy","secondary_strategy":null,'
        '"confidence":0.9,"justification":"matches"}'
    )
    unknown = (
        '{"primary_strategy":"invented_palindrome_strategy",'
        '"secondary_strategy":null,"confidence":0.9,'
        '"justification":"no frozen label fits"}'
    )
    provider = ScriptedMetadataProvider(
        {"good": [valid], "palindrome": [unknown, unknown]}, attempts=2
    )
    labeler = PromptedStrategyLabeler(provider)
    requests = [
        {
            "problem_id": problem_id,
            "labeler_input": f"solution for {problem_id}",
            "taxonomy": taxonomy,
            "seed": index + 1,
            "deterministic": False,
        }
        for index, problem_id in enumerate(("good", "palindrome"))
    ]

    with pytest.raises(MetadataBatchExhaustedError) as captured:
        labeler.classify_batch(requests)

    assert captured.value.structural_failure is True
    assert captured.value.failed_indexes == (1,)
    assert captured.value.partial_results[0].primary_strategy == "known_strategy"
    assert provider.problem_call_counts == {"good": 1, "palindrome": 2}


def test_bedrock_connection_pool_matches_worker_count(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured["config_kwargs"] = kwargs

    fake_boto3 = types.ModuleType("boto3")

    def fake_client(service: str, **kwargs: Any) -> object:
        captured["service"] = service
        captured["client_kwargs"] = kwargs
        return object()

    fake_boto3.client = fake_client  # type: ignore[attr-defined]
    fake_botocore = types.ModuleType("botocore")
    fake_botocore.__path__ = []  # type: ignore[attr-defined]
    fake_botocore_client = types.ModuleType("botocore.client")
    fake_botocore_client.Config = FakeConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.client", fake_botocore_client)

    provider = BedrockConverseChatProvider(
        model="bedrock-model", max_workers=17, attempts=2
    )

    assert provider.max_workers == 17
    assert captured["service"] == "bedrock-runtime"
    assert captured["config_kwargs"]["max_pool_connections"] == 17
    assert isinstance(captured["client_kwargs"]["config"], FakeConfig)
