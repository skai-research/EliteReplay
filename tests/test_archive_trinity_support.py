from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import pytest
import yaml

from elitereplay.EliteReplay.trinity_support import (
    UniformProblemNicheArchive,
    render_sft_with_terminal_eos,
)


def _row(problem: str, niche: str, candidate: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Return only the artifact."},
            {"role": "user", "content": f"problem {problem}"},
            {"role": "assistant", "content": f"answer {candidate}"},
        ],
        "metadata": {
            "problem_id": problem,
            "primary_strategy": niche,
            "secondary_strategy": None,
            "candidate_id": candidate,
            "verifier_score": 1.0,
            "archive_method": "qd",
        },
    }


def test_problem_then_niche_sampling_is_not_flat_row_sampling() -> None:
    rows = [_row("one_niche", "n0", "a")]
    rows.extend(_row("nine_niches", f"n{i}", f"b{i}") for i in range(9))
    archive = UniformProblemNicheArchive(rows, seed=7)

    samples = archive.sample(40_000)
    problem_counts = Counter(record.problem_id for record in samples)
    assert problem_counts["one_niche"] / len(samples) == pytest.approx(0.5, abs=0.015)
    assert problem_counts["nine_niches"] / len(samples) == pytest.approx(0.5, abs=0.015)

    niche_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in samples:
        niche_counts[record.problem_id][record.niche[0]] += 1
    expected = problem_counts["nine_niches"] / 9
    for count in niche_counts["nine_niches"].values():
        assert count == pytest.approx(expected, rel=0.08)


def test_expert_sampler_resume_replays_exact_sequence_and_checks_archive() -> None:
    rows = [_row("p0", "n0", "a"), _row("p0", "n1", "b"), _row("p1", "n0", "c")]
    first = UniformProblemNicheArchive(rows, seed=42)
    first.sample(17)
    state = first.state_dict()
    expected = [record.candidate_id for record in first.sample(25)]

    resumed = UniformProblemNicheArchive(rows, seed=42)
    resumed.load_state_dict(state)
    actual = [record.candidate_id for record in resumed.sample(25)]
    assert actual == expected
    assert resumed.cursor == 42

    changed = UniformProblemNicheArchive(rows[:-1], seed=42)
    with pytest.raises(ValueError, match="archive changed"):
        changed.load_state_dict(state)


def test_archive_rejects_duplicate_niche_and_unverified_row() -> None:
    with pytest.raises(ValueError, match="multiple elites"):
        UniformProblemNicheArchive(
            [_row("p", "n", "a"), _row("p", "n", "b")]
        )
    bad = _row("p", "n", "a")
    bad["metadata"]["verifier_score"] = 0.5
    with pytest.raises(ValueError, match="not fully correct"):
        UniformProblemNicheArchive([bad])


class _FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        assert messages[-1]["role"] == "user"
        return [10, 11, 12]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [20, 21] if text != "embedded-eos" else [20, 99]


def test_sft_render_masks_prompt_and_supervises_one_actual_terminal_eos() -> None:
    tokens, prompt_length, _ = render_sft_with_terminal_eos(
        _FakeTokenizer(), _row("p", "n", "a")["messages"], enable_thinking=False
    )
    assert tokens[:prompt_length] == [10, 11, 12]
    assert tokens[prompt_length:] == [20, 21, 99]
    assert tokens[prompt_length:].count(_FakeTokenizer.eos_token_id) == 1
    assert tokens[-1] == _FakeTokenizer.eos_token_id


def test_sft_render_rejects_embedded_eos() -> None:
    messages = _row("p", "n", "a")["messages"]
    messages[-1]["content"] = "embedded-eos"
    with pytest.raises(ValueError, match="embedded EOS"):
        render_sft_with_terminal_eos(_FakeTokenizer(), messages)
