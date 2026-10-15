"""Append-only journals and deterministic per-descriptor MAP-Elites storage."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .schema import CandidateRecord, Taxonomy, canonical_json, content_hash


class AppendOnlyJournal:
    """Single-writer append-only JSONL journal with unique logical keys."""

    def __init__(self, path: str | Path, *, key_field: str):
        self.path = Path(path)
        self.key_field = key_field
        self._records: list[dict[str, Any]] = []
        self._by_key: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL record at {self.path}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"journal record at {self.path}:{line_number} is not an object"
                    )
                self._index(record, line_number=line_number)

    def _index(self, record: dict[str, Any], *, line_number: int | None) -> None:
        if self.key_field not in record:
            location = f" at line {line_number}" if line_number is not None else ""
            raise ValueError(f"journal record lacks {self.key_field!r}{location}")
        key = str(record[self.key_field])
        if key in self._by_key:
            location = f" at line {line_number}" if line_number is not None else ""
            raise ValueError(f"duplicate journal key {key!r}{location}")
        self._records.append(record)
        self._by_key[key] = record

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def get(self, key: str) -> dict[str, Any] | None:
        return self._by_key.get(key)

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self._by_key)

    def append(self, record: Mapping[str, Any]) -> bool:
        materialized = json.loads(canonical_json(record))
        key = str(materialized[self.key_field])
        existing = self._by_key.get(key)
        if existing is not None:
            if canonical_json(existing) != canonical_json(materialized):
                raise ValueError(f"conflicting record for journal key {key!r}")
            return False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (canonical_json(materialized) + "\n").encode("utf-8")
        descriptor = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640
        )
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):  # pragma: no cover - very rare OS failure
                raise OSError(f"short append to {self.path}: {written}/{len(encoded)}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._index(materialized, line_number=None)
        return True


def _eligible(candidate: CandidateRecord) -> bool:
    return bool(
        candidate.parse_accepted
        and candidate.roundtrip_accepted
        and candidate.executable
        and candidate.fully_correct
        and candidate.canonical_response is not None
        and candidate.primary_strategy is not None
        and candidate.niche is not None
        and math.isfinite(candidate.verifier_score)
    )


def _cleanliness(candidate: CandidateRecord) -> tuple[int, int]:
    # Strict parsing dominates response length. In practice every eligible
    # record has parse_accepted=True; retaining it here makes the invariant
    # explicit and supports replaying older journals.
    return (int(candidate.parse_accepted), -candidate.output_tokens)


class MapElitesArchive:
    """One fully verified elite per realized descriptor cell of each problem."""

    def __init__(self) -> None:
        self._cells: dict[str, dict[str, CandidateRecord]] = {}

    @property
    def problem_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._cells))

    def elites_for(self, problem_id: str) -> tuple[CandidateRecord, ...]:
        cells = self._cells.get(problem_id, {})
        return tuple(cells[key] for key in sorted(cells))

    def incumbent(
        self, problem_id: str, primary_strategy: str
    ) -> CandidateRecord | None:
        return self._cells.get(problem_id, {}).get(primary_strategy)

    def insert(self, candidate: CandidateRecord) -> CandidateRecord:
        """Apply deterministic replacement and return the journal-ready record."""

        baseline = dataclasses.replace(
            candidate, accepted_into_archive=False, replaced_candidate_id=None
        )
        if not _eligible(baseline):
            return baseline

        niche = baseline.niche
        assert niche is not None
        problem_cells = self._cells.setdefault(baseline.problem_id, {})
        incumbent = problem_cells.get(niche)
        should_replace = incumbent is None
        if incumbent is not None:
            should_replace = baseline.verifier_score > incumbent.verifier_score
            if baseline.verifier_score == incumbent.verifier_score:
                should_replace = _cleanliness(baseline) > _cleanliness(incumbent)

        if not should_replace:
            return baseline
        accepted = dataclasses.replace(
            baseline,
            accepted_into_archive=True,
            replaced_candidate_id=(
                None if incumbent is None else incumbent.candidate_id
            ),
        )
        problem_cells[niche] = accepted
        return accepted

    def replay(self, records: Iterable[CandidateRecord]) -> None:
        for stored in records:
            computed = self.insert(stored)
            if (
                computed.accepted_into_archive != stored.accepted_into_archive
                or computed.replaced_candidate_id != stored.replaced_candidate_id
            ):
                raise ValueError(
                    f"archive decision mismatch while replaying {stored.candidate_id}"
                )

    def coverage_counts(self) -> Counter[str]:
        # Global primary coverage counts questions, not the number of subtypes
        # one prolific question happens to occupy within a primary family.
        return Counter(
            primary
            for cells in self._cells.values()
            for primary in {candidate.primary_strategy for candidate in cells.values()}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            problem_id: {
                strategy: candidate.to_dict()
                for strategy, candidate in sorted(cells.items())
            }
            for problem_id, cells in sorted(self._cells.items())
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MapElitesArchive":
        archive = cls()
        for problem_id, cells in value.items():
            if not isinstance(cells, Mapping):
                raise ValueError(f"archive cells for {problem_id!r} are malformed")
            for strategy, raw_candidate in cells.items():
                candidate = CandidateRecord.from_dict(raw_candidate)
                if candidate.problem_id != problem_id or candidate.niche != strategy:
                    raise ValueError("archive cell key disagrees with candidate metadata")
                if not _eligible(candidate):
                    raise ValueError("frozen archive contains an ineligible candidate")
                archive._cells.setdefault(problem_id, {})[strategy] = candidate
        return archive


def _seeded_order(seed: int, namespace: str, values: Iterable[str]) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            content_hash({"seed": seed, "namespace": namespace, "value": value}),
            value,
        ),
    )


def choose_target_strategy(
    *,
    archive: MapElitesArchive,
    problem_id: str,
    feasible: Sequence[str],
    seed: int,
    slot_key: str,
    attempt_counts: Mapping[str, int] | None = None,
) -> str:
    if not feasible:
        raise ValueError(f"problem {problem_id!r} has no feasible strategies")
    occupied = {candidate.primary_strategy for candidate in archive.elites_for(problem_id)}
    global_counts = archive.coverage_counts()
    empty = [strategy for strategy in feasible if strategy not in occupied]
    if attempt_counts is not None:
        pool = empty or list(feasible)
        least_attempted = min(attempt_counts.get(strategy, 0) for strategy in pool)
        pool = [strategy for strategy in pool if attempt_counts.get(strategy, 0) == least_attempted]
        least_covered = min(global_counts[strategy] for strategy in pool)
        pool = [strategy for strategy in pool if global_counts[strategy] == least_covered]
        return _seeded_order(seed, f"target_attempts:{slot_key}", pool)[0]
    if empty:
        minimum = min(global_counts[strategy] for strategy in empty)
        least_represented = [
            strategy for strategy in empty if global_counts[strategy] == minimum
        ]
        return _seeded_order(seed, f"target:{slot_key}", least_represented)[0]

    filled = [candidate for candidate in archive.elites_for(problem_id)
              if candidate.primary_strategy in feasible]
    lowest_fitness = min(candidate.verifier_score for candidate in filled)
    lowest = [
        candidate.primary_strategy
        for candidate in filled
        if candidate.verifier_score == lowest_fitness and candidate.niche is not None
    ]
    return _seeded_order(seed, f"target_filled:{slot_key}", lowest)[0]


def choose_parent(
    *,
    archive: MapElitesArchive,
    problem_id: str,
    target_strategy: str,
    seed: int,
    slot_key: str,
) -> CandidateRecord | None:
    elites = list(archive.elites_for(problem_id))
    if not elites:
        return None
    different = [elite for elite in elites if elite.primary_strategy != target_strategy]
    candidates = different or elites
    ordered_ids = _seeded_order(
        seed, f"parent:{slot_key}", (item.candidate_id for item in candidates)
    )
    by_id = {item.candidate_id: item for item in candidates}
    return by_id[ordered_ids[0]]


def select_problem_balanced_elites(
    archive: MapElitesArchive,
    *,
    maximum: int,
    seed: int,
    maximum_per_problem: int | None = None,
    deduplication_key: Callable[[CandidateRecord], str] | None = None,
) -> tuple[CandidateRecord, ...]:
    """Round-robin problems, then niches, until *maximum* examples are selected."""

    if maximum <= 0:
        raise ValueError("maximum export size must be positive")
    problems = _seeded_order(seed, "export_problems", archive.problem_ids)
    queues: dict[str, list[CandidateRecord]] = {}
    for problem_id in problems:
        elites = archive.elites_for(problem_id)
        ordered_ids = _seeded_order(
            seed,
            f"export_niches:{problem_id}",
            (elite.candidate_id for elite in elites),
        )
        by_id = {elite.candidate_id: elite for elite in elites}
        queue = [by_id[candidate_id] for candidate_id in ordered_ids]
        if deduplication_key is not None:
            # Deduplicate before applying either cap, so duplicate labels do
            # not consume capacity and prevent another question being served.
            seen: set[str] = set()
            unique: list[CandidateRecord] = []
            for candidate in queue:
                key = deduplication_key(candidate)
                if key not in seen:
                    seen.add(key)
                    unique.append(candidate)
            queue = unique
        queues[problem_id] = (
            queue if maximum_per_problem is None else queue[:maximum_per_problem]
        )

    selected: list[CandidateRecord] = []
    offset = 0
    while len(selected) < maximum:
        added = False
        for problem_id in problems:
            queue = queues[problem_id]
            if offset < len(queue):
                selected.append(queue[offset])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


def frozen_archive_document(
    *,
    domain: str,
    taxonomy: Taxonomy,
    archive: MapElitesArchive,
    candidate_log_sha256: str,
    config_hash: str,
) -> dict[str, Any]:
    content = {
        "schema_version": "EliteReplay.archive.v1",
        "domain": domain,
        "method": "qd",
        "taxonomy_hash": taxonomy.hash,
        "candidate_log_sha256": candidate_log_sha256,
        "config_hash": config_hash,
        "problems": archive.to_dict(),
    }
    return {**content, "archive_hash": content_hash(content)}


def load_frozen_archive(
    path: str | Path, *, expected_taxonomy_hash: str | None = None
) -> tuple[MapElitesArchive, Mapping[str, Any]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded_hash = document.get("archive_hash")
    content = {key: value for key, value in document.items() if key != "archive_hash"}
    if recorded_hash != content_hash(content):
        raise ValueError("frozen archive content does not match archive_hash")
    if (
        expected_taxonomy_hash is not None
        and document.get("taxonomy_hash") != expected_taxonomy_hash
    ):
        raise ValueError("frozen archive was built with a different taxonomy")
    return MapElitesArchive.from_dict(document["problems"]), document
