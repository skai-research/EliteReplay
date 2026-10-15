"""Pure helpers used by the project-local Trinity-RFT EliteReplay plugin.

This module deliberately does not import Trinity-RFT.  Archive validation,
two-stage sampling, and exact EOS rendering can therefore be tested in the
lightweight EliteReplay environment before a GPU training runtime is launched.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ON_POLICY_SOURCE = "on_policy_rollout"
FROZEN_SFT_SOURCE = "frozen_sft"
ON_POLICY_SOURCE_ID = 0
FROZEN_SFT_SOURCE_ID = 1


def _plain_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def normalize_messages(value: Any) -> list[dict[str, str]]:
    """Validate the single-turn messages wire format and return a plain copy."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("messages is not valid JSON") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("messages must be a sequence")
    messages: list[dict[str, str]] = []
    for raw_message in value:
        message = _plain_mapping(raw_message, field="message")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        messages.append({"role": str(role), "content": content})
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("SFT messages must end in one assistant response")
    if any(message["role"] == "assistant" for message in messages[:-1]):
        raise ValueError("EliteReplay SFT supports one assistant turn only")
    if not any(message["role"] == "user" for message in messages[:-1]):
        raise ValueError("SFT messages must contain a user prompt")
    if not messages[-1]["content"]:
        raise ValueError("assistant response must not be empty")
    return messages


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class ArchiveRow:
    """A validated archive row and the keys used by two-stage sampling."""

    row: Mapping[str, Any]
    problem_id: str
    niche: tuple[str, str | None]
    candidate_id: str


class UniformProblemNicheArchive:
    """Uniformly sample a problem, then a niche available for that problem.

    There is exactly one elite per ``(problem, niche)``. Sampling is with
    replacement, as required for persistent replay of a small frozen archive.
    The Python RNG state and draw cursor are JSON-serialisable so a Trinity
    checkpoint resumes the exact expert-example sequence.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        seed: int = 42,
        expected_archive_method: str | None = "qd",
        require_fully_correct: bool = True,
    ) -> None:
        grouped: dict[str, dict[tuple[str, str | None], ArchiveRow]] = defaultdict(dict)
        candidate_ids: set[str] = set()
        canonical_rows: list[dict[str, Any]] = []

        for index, original in enumerate(rows):
            row = dict(original)
            metadata = _plain_mapping(row.get("metadata", {}), field=f"row {index} metadata")
            messages = normalize_messages(row.get("messages"))
            problem_id = str(metadata.get("problem_id", row.get("problem_id", ""))).strip()
            primary = str(
                metadata.get("primary_strategy", row.get("primary_strategy", ""))
            ).strip()
            secondary_value = metadata.get(
                "secondary_strategy", row.get("secondary_strategy")
            )
            secondary = None if secondary_value in (None, "") else str(secondary_value).strip()
            candidate_id = str(
                metadata.get("candidate_id", row.get("candidate_id", ""))
            ).strip()
            if not problem_id:
                raise ValueError(f"archive row {index} has no problem_id")
            if not primary:
                raise ValueError(f"archive row {index} has no primary_strategy")
            if not candidate_id:
                raise ValueError(f"archive row {index} has no candidate_id")
            if candidate_id in candidate_ids:
                raise ValueError(f"duplicate archive candidate_id: {candidate_id}")

            archive_method = metadata.get("archive_method", row.get("archive_method"))
            if expected_archive_method is not None and archive_method != expected_archive_method:
                raise ValueError(
                    f"archive row {candidate_id} has method {archive_method!r}; "
                    f"expected {expected_archive_method!r}"
                )
            verifier_score = metadata.get("verifier_score", row.get("verifier_score"))
            fully_correct = metadata.get("fully_correct", row.get("fully_correct"))
            if require_fully_correct and not (
                fully_correct is True
                or float(verifier_score if verifier_score is not None else -1) == 1.0
            ):
                raise ValueError(f"archive row {candidate_id} is not fully correct")

            niche = (primary, secondary)
            if niche in grouped[problem_id]:
                raise ValueError(
                    f"multiple elites for problem={problem_id!r}, niche={niche!r}"
                )
            plain_row = dict(row)
            plain_row["messages"] = messages
            plain_row["metadata"] = metadata
            record = ArchiveRow(
                row=plain_row,
                problem_id=problem_id,
                niche=niche,
                candidate_id=candidate_id,
            )
            grouped[problem_id][niche] = record
            candidate_ids.add(candidate_id)
            canonical_rows.append(
                {
                    "problem_id": problem_id,
                    "niche": [primary, secondary],
                    "candidate_id": candidate_id,
                    "messages": messages,
                    "metadata": metadata,
                }
            )

        if not grouped:
            raise ValueError("frozen SFT archive is empty")
        self._records = {
            problem_id: tuple(
                niches[niche] for niche in sorted(niches, key=lambda item: (item[0], item[1] or ""))
            )
            for problem_id, niches in sorted(grouped.items())
        }
        self.problem_ids = tuple(self._records)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.cursor = 0
        self.fingerprint = hashlib.sha256(
            _canonical_json(sorted(canonical_rows, key=lambda row: row["candidate_id"])).encode(
                "utf-8"
            )
        ).hexdigest()

    def sample(self, count: int) -> list[ArchiveRow]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        result: list[ArchiveRow] = []
        for _ in range(count):
            problem_id = self._rng.choice(self.problem_ids)
            result.append(self._rng.choice(self._records[problem_id]))
            self.cursor += 1
        return result

    @property
    def problem_count(self) -> int:
        return len(self.problem_ids)

    @property
    def niche_count(self) -> int:
        return sum(len(records) for records in self._records.values())

    def state_dict(self) -> dict[str, Any]:
        version, internal, gaussian = self._rng.getstate()
        return {
            "version": self.STATE_VERSION,
            "seed": self.seed,
            "cursor": self.cursor,
            "archive_fingerprint": self.fingerprint,
            "rng_state": [version, list(internal), gaussian],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("unsupported expert sampler state version")
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("expert sampler seed changed across resume")
        if state.get("archive_fingerprint") != self.fingerprint:
            raise ValueError("frozen SFT archive changed across resume")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, list) or len(rng_state) != 3:
            raise ValueError("malformed expert RNG state")
        self._rng.setstate((int(rng_state[0]), tuple(rng_state[1]), rng_state[2]))
        cursor = int(state.get("cursor", 0))
        if cursor < 0:
            raise ValueError("expert sampler cursor cannot be negative")
        self.cursor = cursor

    @staticmethod
    def batch_counts(records: Sequence[ArchiveRow]) -> tuple[Counter[str], Counter[str]]:
        problems: Counter[str] = Counter(record.problem_id for record in records)
        niches: Counter[str] = Counter(
            f"{record.problem_id}:{record.niche[0]}:{record.niche[1] or ''}"
            for record in records
        )
        return problems, niches


def _token_ids(value: Any) -> list[int]:
    """Flatten common tokenizer return shapes without importing torch/numpy."""

    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ValueError("expected one tokenized sequence")
        value = value[0]
    if not isinstance(value, Sequence):
        raise ValueError("tokenizer did not return a token sequence")
    return [int(token_id) for token_id in value]


def render_sft_with_terminal_eos(
    tokenizer: Any,
    messages_value: Any,
    *,
    chat_template: str | None = None,
    enable_thinking: bool | None = False,
) -> tuple[list[int], int, list[dict[str, str]]]:
    """Render one assistant turn with one explicitly supervised actual EOS.

    Only the prompt is rendered with the chat template.  The canonical
    assistant artifact is tokenized verbatim and the tokenizer's real EOS ID
    is appended once.  This avoids depending on whether a model template ends
    assistant messages with an end-of-turn token rather than EOS.
    """

    messages = normalize_messages(messages_value)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer has no eos_token_id")
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if chat_template is not None:
        template_kwargs["chat_template"] = chat_template
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(messages[:-1], **template_kwargs)
    )
    response_ids = _token_ids(
        tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    )
    if int(eos_token_id) in prompt_ids:
        raise ValueError("chat prompt already contains the model EOS token")
    if int(eos_token_id) in response_ids:
        raise ValueError("canonical response contains an embedded EOS token")
    if not response_ids:
        raise ValueError("canonical response tokenizes to an empty sequence")
    tokens = prompt_ids + response_ids + [int(eos_token_id)]
    prompt_length = len(prompt_ids)
    supervised = tokens[prompt_length:]
    if supervised[-1] != int(eos_token_id) or supervised.count(int(eos_token_id)) != 1:
        raise AssertionError("SFT response must contain exactly one terminal EOS")
    return tokens, prompt_length, messages


def repeated_special_tag(text: str) -> bool:
    lowered = text.lower()
    tags = ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")
    return any(lowered.count(tag) > 0 for tag in tags)
