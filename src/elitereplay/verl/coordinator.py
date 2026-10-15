"""In-process coordination for two-phase evolutionary rollouts.

veRL's TransferQueue agent-loop worker starts all ``rollout.n`` sessions for a
prompt as asyncio tasks on one Ray actor.  This module uses that property to
let parent sessions publish candidates and offspring sessions wait for them.
It has no veRL dependency, which makes the concurrency protocol CPU-testable.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Hashable, Sequence


GroupKey = tuple[Hashable, str]


class EvolutionCoordinationError(RuntimeError):
    """Base error for an evolutionary rollout group."""


class ParentGenerationError(EvolutionCoordinationError):
    """Raised in offspring sessions when at least one parent failed."""


class EvolutionCoordinationTimeout(EvolutionCoordinationError):
    """Raised when offspring cannot observe every parent before the deadline."""


@dataclass(frozen=True)
class ParentCandidate:
    """The information an offspring session needs from a direct rollout."""

    parent_index: int
    response_ids: tuple[int, ...]
    response_text: str
    # DARE children receive only the exact Python arguments written by their
    # parents.  Text-only tasks leave this empty.
    code_calls: tuple[str, ...] = ()
    final_artifact: str = ""
    verifier_score: float = 0.0
    task_score: float = 0.0
    eligible: bool = True
    ineligible_reason: str = ""


@dataclass
class _GroupState:
    expected_parents: int
    expected_offspring: int
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    parents: dict[int, ParentCandidate] = field(default_factory=dict)
    failures: dict[int, str] = field(default_factory=dict)
    finished_children: set[int] = field(default_factory=set)
    expiry_handle: asyncio.TimerHandle | None = None


def stable_seed(*parts: object) -> int:
    """Return a process-independent 64-bit seed for the supplied values."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def choose_parent_indices(
    *,
    num_parents: int,
    child_index: int,
    crossover_probability: float,
    seed: int,
    group_key: GroupKey,
) -> tuple[str, tuple[int, ...]]:
    """Choose mutation/crossover parents deterministically.

    Primary parents cover the population round-robin (with a deterministic
    per-group rotation).  Crossover partners are uniform among the remaining
    parents.  This is deliberately reward-free: rewards are computed only
    after each agent-loop session returns in veRL v0.8.
    """

    if num_parents < 1:
        raise ValueError("num_parents must be at least 1")
    if child_index < 0:
        raise ValueError("child_index must be non-negative")
    if not 0.0 <= crossover_probability <= 1.0:
        raise ValueError("crossover_probability must be in [0, 1]")

    rng = random.Random(stable_seed(seed, group_key[0], group_key[1], child_index))
    rotation = rng.randrange(num_parents)
    primary = (rotation + child_index) % num_parents

    if num_parents == 1 or rng.random() >= crossover_probability:
        return "mutation", (primary,)

    partner = rng.choice([idx for idx in range(num_parents) if idx != primary])
    return "crossover", (primary, partner)


MUTATION_INSTRUCTION = (
    "Produce a new candidate solution to the original task. Use a meaningfully "
    "different solution strategy from the preceding candidate, rather than "
    "merely renaming variables, rephrasing, or changing surface formatting. "
    "Preserve every output-format and interface requirement from the original "
    "task. Return only the candidate solution in the form the task requests."
)

CROSSOVER_INSTRUCTION = (
    "A second candidate is shown below.\n\n"
    "--- candidate B ---\n"
    "{candidate_b}\n"
    "--- end candidate B ---\n\n"
    "Produce a new candidate solution to the original task. Combine useful "
    "ideas from the two candidates only when appropriate, but use a solution "
    "strategy that is meaningfully distinct from both. Do not make merely "
    "surface-level changes. Preserve every output-format and interface "
    "requirement from the original task. Return only the candidate solution "
    "in the form the task requests."
)


def build_child_messages(
    raw_prompt: Sequence[dict],
    *,
    operator: str,
    parent_texts: Sequence[str],
) -> list[dict]:
    """Build a domain-general mutation or crossover conversation."""

    if operator not in {"mutation", "crossover"}:
        raise ValueError(f"unsupported evolutionary operator: {operator}")
    expected = 1 if operator == "mutation" else 2
    if len(parent_texts) != expected:
        raise ValueError(f"{operator} requires {expected} parent text(s)")

    messages = copy.deepcopy(list(raw_prompt))
    messages.append({"role": "assistant", "content": parent_texts[0]})
    instruction = (
        MUTATION_INSTRUCTION
        if operator == "mutation"
        else CROSSOVER_INSTRUCTION.format(candidate_b=parent_texts[1])
    )
    messages.append({"role": "user", "content": instruction})
    return messages


def token_fingerprint(token_ids: Sequence[int]) -> str:
    """Hash a token sequence without retaining prompt text in metadata."""

    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class EvolutionCoordinator:
    """Coordinate parent and offspring tasks on one asyncio event loop.

    State is removed after every expected child finishes.  A delayed expiry is
    also installed so direct-only validation prompts and cancelled groups do
    not accumulate forever.
    """

    def __init__(self, *, minimum_expiry_seconds: float = 60.0) -> None:
        self._groups: dict[GroupKey, _GroupState] = {}
        self._minimum_expiry_seconds = minimum_expiry_seconds

    @property
    def active_groups(self) -> int:
        return len(self._groups)

    def clear(self) -> None:
        for state in self._groups.values():
            if state.expiry_handle is not None:
                state.expiry_handle.cancel()
        self._groups.clear()

    def _expire(self, key: GroupKey, state: _GroupState) -> None:
        if self._groups.get(key) is state:
            self._groups.pop(key, None)

    def _state(
        self,
        key: GroupKey,
        *,
        expected_parents: int,
        expected_offspring: int,
        timeout_seconds: float,
    ) -> _GroupState:
        if expected_parents < 1:
            raise ValueError("expected_parents must be at least 1")
        if expected_offspring < 1:
            raise ValueError("expected_offspring must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        state = self._groups.get(key)
        if state is not None:
            if (
                state.expected_parents != expected_parents
                or state.expected_offspring != expected_offspring
            ):
                raise EvolutionCoordinationError(
                    f"inconsistent group shape for {key!r}: "
                    f"existing=({state.expected_parents}, {state.expected_offspring}), "
                    f"requested=({expected_parents}, {expected_offspring})"
                )
            return state

        state = _GroupState(
            expected_parents=expected_parents,
            expected_offspring=expected_offspring,
        )
        expiry_seconds = max(self._minimum_expiry_seconds, 2.0 * timeout_seconds)
        state.expiry_handle = asyncio.get_running_loop().call_later(
            expiry_seconds, self._expire, key, state
        )
        self._groups[key] = state
        return state

    @staticmethod
    def _mark_ready_if_terminal(state: _GroupState) -> None:
        if len(state.parents) + len(state.failures) >= state.expected_parents:
            state.ready.set()

    def publish_parent(
        self,
        key: GroupKey,
        candidate: ParentCandidate,
        *,
        expected_parents: int,
        expected_offspring: int,
        timeout_seconds: float,
    ) -> None:
        state = self._state(
            key,
            expected_parents=expected_parents,
            expected_offspring=expected_offspring,
            timeout_seconds=timeout_seconds,
        )
        index = candidate.parent_index
        if not 0 <= index < expected_parents:
            raise EvolutionCoordinationError(f"parent index {index} is out of range")
        if index in state.failures:
            raise EvolutionCoordinationError(f"parent {index} was already marked failed")
        previous = state.parents.get(index)
        if previous is not None and previous != candidate:
            raise EvolutionCoordinationError(f"parent {index} was published twice")
        state.parents[index] = candidate
        self._mark_ready_if_terminal(state)

    def publish_failure(
        self,
        key: GroupKey,
        parent_index: int,
        error: str,
        *,
        expected_parents: int,
        expected_offspring: int,
        timeout_seconds: float,
    ) -> None:
        state = self._state(
            key,
            expected_parents=expected_parents,
            expected_offspring=expected_offspring,
            timeout_seconds=timeout_seconds,
        )
        if not 0 <= parent_index < expected_parents:
            raise EvolutionCoordinationError(f"parent index {parent_index} is out of range")
        if parent_index not in state.parents:
            state.failures[parent_index] = error
        self._mark_ready_if_terminal(state)

    async def wait_for_parents(
        self,
        key: GroupKey,
        *,
        expected_parents: int,
        expected_offspring: int,
        timeout_seconds: float,
        allow_failures: bool = False,
        minimum_available: int = 1,
    ) -> tuple[ParentCandidate, ...]:
        state = self._state(
            key,
            expected_parents=expected_parents,
            expected_offspring=expected_offspring,
            timeout_seconds=timeout_seconds,
        )
        try:
            await asyncio.wait_for(state.ready.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            state.failures.setdefault(
                -1,
                f"timed out after {timeout_seconds:g}s with "
                f"{len(state.parents)}/{expected_parents} parents",
            )
            state.ready.set()
            raise EvolutionCoordinationTimeout(state.failures[-1]) from exc

        if state.failures and not allow_failures:
            details = ", ".join(
                f"parent {idx}: {message}" if idx >= 0 else message
                for idx, message in sorted(state.failures.items())
            )
            raise ParentGenerationError(details)
        if len(state.parents) < minimum_available:
            details = ", ".join(
                f"parent {idx}: {message}" if idx >= 0 else message
                for idx, message in sorted(state.failures.items())
            )
            raise ParentGenerationError(
                f"only {len(state.parents)} available parents; require "
                f"{minimum_available}. {details}"
            )
        if not allow_failures and len(state.parents) != expected_parents:
            raise EvolutionCoordinationError(
                f"group {key!r} became ready with only "
                f"{len(state.parents)}/{expected_parents} parents"
            )
        return tuple(state.parents[idx] for idx in sorted(state.parents))

    def failed_parent_indices(self, key: GroupKey) -> tuple[int, ...]:
        """Return terminal parent failures while the group is still active.

        ``wait_for_parents(..., allow_failures=True)`` intentionally returns
        only usable candidates.  Evolution callers still need the failed
        indices for trajectory-integrity metadata, so expose that information
        without leaking error strings into training records.
        """

        state = self._groups.get(key)
        if state is None:
            return ()
        return tuple(sorted(index for index in state.failures if index >= 0))

    def child_finished(self, key: GroupKey, child_index: int) -> None:
        state = self._groups.get(key)
        if state is None or child_index in state.finished_children:
            return
        state.finished_children.add(child_index)
        if len(state.finished_children) >= state.expected_offspring:
            if state.expiry_handle is not None:
                state.expiry_handle.cancel()
            self._groups.pop(key, None)
