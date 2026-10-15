"""BIRD-specific MAP-Elites primitives for the simple AMES training arm.

The genotype records structural SQL choices rather than answer text or schema
details.  This keeps strategy cards schema-blind while making the descriptor
deterministic from the SQL that was actually generated.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from elitereplay.verl.coordinator import GroupKey, stable_seed


SCHEMA_VERSION = 1
ARCHIVE_EPSILON = 0.0

GENOTYPE_VOCABULARY: dict[str, tuple[str, ...]] = {
    "query_form": ("flat_query", "nested_subquery", "common_table_expression"),
    "relation_strategy": (
        "single_relation",
        "explicit_join",
        "semijoin_or_correlated_subquery",
    ),
    "aggregation": (
        "no_aggregation",
        "scalar_aggregation",
        "grouped_aggregation",
        "conditional_aggregation",
        "window_function",
    ),
    "set_composition": ("no_set_operation", "union", "intersect", "except"),
    "duplicate_handling": ("preserve_duplicates", "explicit_distinct"),
}
GENOTYPE_FIELDS = tuple(GENOTYPE_VOCABULARY)
COARSE_FIELDS = ("query_form", "aggregation", "set_composition")

GENOTYPE_LABELS: dict[str, dict[str, str]] = {
    "query_form": {
        "flat_query": "a flat query without a CTE or nested SELECT",
        "nested_subquery": "a query decomposed with a nested subquery",
        "common_table_expression": "a query decomposed with a WITH common-table expression",
    },
    "relation_strategy": {
        "single_relation": "derive the result without an explicit JOIN",
        "explicit_join": "combine the needed relations with explicit JOIN clauses",
        "semijoin_or_correlated_subquery": "use EXISTS, IN with a subquery, or a correlated subquery to connect relations",
    },
    "aggregation": {
        "no_aggregation": "select or filter rows without aggregation",
        "scalar_aggregation": "use scalar aggregation without GROUP BY",
        "grouped_aggregation": "use GROUP BY aggregation",
        "conditional_aggregation": "use conditional aggregation with CASE",
        "window_function": "use a window function with OVER",
    },
    "set_composition": {
        "no_set_operation": "do not use a set operator",
        "union": "compose compatible result sets with UNION",
        "intersect": "compose compatible result sets with INTERSECT",
        "except": "compose compatible result sets with EXCEPT",
    },
    "duplicate_handling": {
        "preserve_duplicates": "preserve ordinary SQL bag semantics unless the task itself removes duplicates",
        "explicit_distinct": "use explicit DISTINCT for result deduplication",
    },
}
GENOTYPE_PROMPTS = {
    "query_form": "Query organization",
    "relation_strategy": "Relation strategy",
    "aggregation": "Aggregation strategy",
    "set_composition": "Set composition",
    "duplicate_handling": "Duplicate handling",
}


@dataclass(frozen=True)
class BirdStrategyGenotype:
    query_form: str
    relation_strategy: str
    aggregation: str
    set_composition: str
    duplicate_handling: str

    def __post_init__(self) -> None:
        for field_name in GENOTYPE_FIELDS:
            value = getattr(self, field_name)
            if value not in GENOTYPE_VOCABULARY[field_name]:
                raise ValueError(
                    f"invalid BIRD AMES {field_name} value {value!r}; expected "
                    f"one of {GENOTYPE_VOCABULARY[field_name]!r}"
                )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BirdStrategyGenotype":
        return cls(**{field: str(values[field]) for field in GENOTYPE_FIELDS})

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in GENOTYPE_FIELDS}

    def full_cell(self) -> str:
        return "|".join(getattr(self, field) for field in GENOTYPE_FIELDS)

    def coarse_cell(self) -> str:
        return "|".join(getattr(self, field) for field in COARSE_FIELDS)


def random_genotype(rng: random.Random) -> BirdStrategyGenotype:
    return BirdStrategyGenotype(
        **{
            field: rng.choice(GENOTYPE_VOCABULARY[field])
            for field in GENOTYPE_FIELDS
        }
    )


def point_mutation(
    parent: BirdStrategyGenotype, rng: random.Random
) -> tuple[BirdStrategyGenotype, str]:
    """Change exactly one uniformly selected card coordinate."""

    field_name = rng.choice(GENOTYPE_FIELDS)
    current = getattr(parent, field_name)
    replacement = rng.choice(
        [value for value in GENOTYPE_VOCABULARY[field_name] if value != current]
    )
    values = parent.as_dict()
    values[field_name] = replacement
    return BirdStrategyGenotype.from_mapping(values), field_name


def uniform_crossover(
    first: BirdStrategyGenotype,
    second: BirdStrategyGenotype,
    rng: random.Random,
) -> tuple[BirdStrategyGenotype, tuple[int, ...]]:
    """Independently inherit every coordinate from either source elite."""

    mask = tuple(int(rng.random() < 0.5) for _ in GENOTYPE_FIELDS)
    values = {
        field: getattr(second if from_second else first, field)
        for field, from_second in zip(GENOTYPE_FIELDS, mask, strict=True)
    }
    return BirdStrategyGenotype.from_mapping(values), mask


def _masked_sql(sql: str) -> str:
    """Remove comments and quoted content before structural keyword checks."""

    pattern = re.compile(
        r"--[^\n]*(?:\n|$)|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"",
        flags=re.DOTALL,
    )
    return re.sub(pattern, " ", sql).lower()


def realized_descriptor(sql: str) -> BirdStrategyGenotype:
    """Map generated SQLite text to one frozen, auditable strategy cell."""

    normalized = " ".join(_masked_sql(sql).split())
    if re.match(r"^with\b", normalized):
        query_form = "common_table_expression"
    elif re.search(r"\(\s*(?:with\b.*?)?select\b", normalized):
        query_form = "nested_subquery"
    else:
        query_form = "flat_query"

    if re.search(r"\bexists\s*\(", normalized) or re.search(
        r"\bin\s*\(\s*(?:with\b.*?)?select\b", normalized
    ):
        relation_strategy = "semijoin_or_correlated_subquery"
    elif re.search(r"\bjoin\b", normalized):
        relation_strategy = "explicit_join"
    else:
        relation_strategy = "single_relation"

    aggregate_call = bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", normalized))
    if re.search(r"\bover\s*\(", normalized):
        aggregation = "window_function"
    elif aggregate_call and re.search(r"\bcase\b", normalized):
        aggregation = "conditional_aggregation"
    elif re.search(r"\bgroup\s+by\b", normalized):
        aggregation = "grouped_aggregation"
    elif aggregate_call:
        aggregation = "scalar_aggregation"
    else:
        aggregation = "no_aggregation"

    set_matches = [
        (match.start(), name)
        for name in ("union", "intersect", "except")
        for match in re.finditer(rf"\b{name}\b", normalized)
    ]
    set_composition = min(set_matches)[1] if set_matches else "no_set_operation"
    duplicate_handling = (
        "explicit_distinct"
        if re.search(r"\bselect\s+distinct\b", normalized)
        else "preserve_duplicates"
    )
    return BirdStrategyGenotype(
        query_form=query_form,
        relation_strategy=relation_strategy,
        aggregation=aggregation,
        set_composition=set_composition,
        duplicate_handling=duplicate_handling,
    )


def render_card_text(genotype: BirdStrategyGenotype) -> str:
    return "\n".join(
        f"- {GENOTYPE_PROMPTS[field]}: "
        f"{GENOTYPE_LABELS[field][getattr(genotype, field)]}"
        for field in GENOTYPE_FIELDS
    )


@dataclass(frozen=True)
class BirdElite:
    genotype: BirdStrategyGenotype
    score: float
    exemplar_id: str
    population: str
    operator: str
    source_cells: tuple[str, ...]
    achieved_step: int
    last_improvement_step: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BirdElite":
        return cls(
            genotype=BirdStrategyGenotype.from_mapping(value["genotype"]),
            score=float(value["score"]),
            exemplar_id=str(value["exemplar_id"]),
            population=str(value.get("population", "")),
            operator=str(value.get("operator", "direct")),
            source_cells=tuple(str(item) for item in value.get("source_cells", [])),
            achieved_step=int(value.get("achieved_step", 0)),
            last_improvement_step=int(value.get("last_improvement_step", 0)),
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["genotype"] = self.genotype.as_dict()
        value["source_cells"] = list(self.source_cells)
        return value


@dataclass(frozen=True)
class BirdArchiveEvent:
    task_key: str
    coarse_event: str
    full_event: str
    coarse_cell: str
    full_cell: str
    candidate_score: float
    previous_coarse_score: float | None
    previous_full_score: float | None
    coarse_occupancy: int
    full_occupancy: int
    coarse_qd_score: float
    full_qd_score: float


@dataclass(frozen=True)
class BirdChildCard:
    operator: str
    genotype: BirdStrategyGenotype
    source_cells: tuple[str, ...] = ()
    source_scores: tuple[float, ...] = ()
    source_last_improvement_steps: tuple[int, ...] = ()
    mutated_field: str = ""
    crossover_mask: tuple[int, ...] = ()
    fallback_reason: str = ""


def build_child_cards(
    elites: Sequence[BirdElite], *, num_children: int, seed: int
) -> tuple[BirdChildCard, ...]:
    if num_children < 2 or num_children % 2:
        raise ValueError("BIRD AMES requires a positive even child population")
    rng = random.Random(seed)
    ordered = tuple(sorted(elites, key=lambda elite: elite.genotype.coarse_cell()))
    if not ordered:
        return tuple(
            BirdChildCard(operator="random", genotype=random_genotype(rng))
            for _ in range(num_children)
        )

    cards: list[BirdChildCard] = []
    half = num_children // 2
    for _ in range(half):
        source = rng.choice(ordered)
        child, field_name = point_mutation(source.genotype, rng)
        cards.append(
            BirdChildCard(
                operator="mutation",
                genotype=child,
                source_cells=(source.genotype.coarse_cell(),),
                source_scores=(source.score,),
                source_last_improvement_steps=(source.last_improvement_step,),
                mutated_field=field_name,
            )
        )
    for _ in range(half):
        if len(ordered) < 2:
            source = rng.choice(ordered)
            child, field_name = point_mutation(source.genotype, rng)
            cards.append(
                BirdChildCard(
                    operator="mutation",
                    genotype=child,
                    source_cells=(source.genotype.coarse_cell(),),
                    source_scores=(source.score,),
                    source_last_improvement_steps=(source.last_improvement_step,),
                    mutated_field=field_name,
                    fallback_reason="crossover_requires_two_distinct_elites",
                )
            )
            continue
        first, second = rng.sample(ordered, 2)
        child, mask = uniform_crossover(first.genotype, second.genotype, rng)
        cards.append(
            BirdChildCard(
                operator="crossover",
                genotype=child,
                source_cells=(
                    first.genotype.coarse_cell(),
                    second.genotype.coarse_cell(),
                ),
                source_scores=(first.score, second.score),
                source_last_improvement_steps=(
                    first.last_improvement_step,
                    second.last_improvement_step,
                ),
                crossover_mask=mask,
            )
        )
    return tuple(cards)


def stable_bird_task_key(extra_info: Mapping[str, Any]) -> str:
    db_id = str(extra_info.get("db_id", "")).strip()
    row_id = str(extra_info.get("filtered_row_id", "")).strip()
    if not db_id or not row_id:
        raise ValueError("BIRD AMES task identity requires db_id and filtered_row_id")
    return f"{db_id}::{row_id}"


def exemplar_id(*, task_key: str, step: int, session_id: int, sql: str) -> str:
    payload = f"{task_key}\x1f{step}\x1f{session_id}\x1f{sql}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BirdAMESArchiveStore:
    """Atomic per-question coarse and full MAP-Elites archives."""

    def __init__(
        self, root: str | os.PathLike[str], *, epsilon: float = ARCHIVE_EPSILON
    ) -> None:
        if epsilon < 0:
            raise ValueError("archive epsilon must be non-negative")
        self.root = Path(root).expanduser().resolve()
        self.epsilon = float(epsilon)
        self.tasks_root = self.root / "tasks"
        self.events_path = self.root / "events.jsonl"
        self.tasks_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(task_key: str) -> str:
        return hashlib.sha256(task_key.encode("utf-8")).hexdigest()

    def _paths(self, task_key: str) -> tuple[Path, Path]:
        stem = self._digest(task_key)
        return self.tasks_root / f"{stem}.json", self.tasks_root / f"{stem}.lock"

    @staticmethod
    def _empty(task_key: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "domain": "bird_sql",
            "task_key": task_key,
            "visits": 0,
            "last_visit_step": None,
            "coarse_cells": {},
            "full_cells": {},
        }

    def _read(self, task_key: str, path: Path) -> dict[str, Any]:
        if not path.exists():
            return self._empty(task_key)
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported BIRD AMES archive schema")
        if value.get("domain") != "bird_sql" or value.get("task_key") != task_key:
            raise ValueError("BIRD AMES archive task mismatch or corruption")
        return value

    @staticmethod
    def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
        descriptor, raw_tmp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(raw_tmp)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def task_state(self, task_key: str) -> dict[str, Any]:
        state_path, lock_path = self._paths(task_key)
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                return copy.deepcopy(self._read(task_key, state_path))
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def snapshot(self, task_key: str) -> tuple[BirdElite, ...]:
        state = self.task_state(task_key)
        return tuple(
            BirdElite.from_dict(value)
            for _, value in sorted(state["coarse_cells"].items())
        )

    def record_visit(self, task_key: str, *, step: int) -> dict[str, Any]:
        state_path, lock_path = self._paths(task_key)
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read(task_key, state_path)
                previous = state.get("last_visit_step")
                reused = int(state.get("visits", 0)) > 0 and previous != int(step)
                if previous != int(step):
                    state["visits"] = int(state.get("visits", 0)) + 1
                    state["last_visit_step"] = int(step)
                    self._write_atomic(state_path, state)
                return {"visits": int(state["visits"]), "archive_reused": reused}
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _insert_cell(
        cells: dict[str, Any], key: str, elite: BirdElite, epsilon: float
    ) -> tuple[str, float | None]:
        previous_raw = cells.get(key)
        if previous_raw is None:
            cells[key] = elite.as_dict()
            return "new_cell", None
        previous = BirdElite.from_dict(previous_raw)
        if elite.score > previous.score + epsilon:
            cells[key] = elite.as_dict()
            return "replacement", previous.score
        return "rejected", previous.score

    def insert(self, task_key: str, elite: BirdElite) -> BirdArchiveEvent:
        state_path, lock_path = self._paths(task_key)
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read(task_key, state_path)
                coarse_event, previous_coarse = self._insert_cell(
                    state["coarse_cells"],
                    elite.genotype.coarse_cell(),
                    elite,
                    self.epsilon,
                )
                full_event, previous_full = self._insert_cell(
                    state["full_cells"],
                    elite.genotype.full_cell(),
                    elite,
                    self.epsilon,
                )
                self._write_atomic(state_path, state)
                event = BirdArchiveEvent(
                    task_key=task_key,
                    coarse_event=coarse_event,
                    full_event=full_event,
                    coarse_cell=elite.genotype.coarse_cell(),
                    full_cell=elite.genotype.full_cell(),
                    candidate_score=float(elite.score),
                    previous_coarse_score=previous_coarse,
                    previous_full_score=previous_full,
                    coarse_occupancy=len(state["coarse_cells"]),
                    full_occupancy=len(state["full_cells"]),
                    coarse_qd_score=sum(
                        float(value["score"])
                        for value in state["coarse_cells"].values()
                    ),
                    full_qd_score=sum(
                        float(value["score"])
                        for value in state["full_cells"].values()
                    ),
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return event


class BirdGroupPlanCache:
    """Freeze all eight cards before same-group children alter the archive."""

    def __init__(self) -> None:
        self._plans: dict[GroupKey, tuple[BirdChildCard, ...]] = {}
        self._finished: dict[GroupKey, set[int]] = {}

    def get_or_create(
        self,
        key: GroupKey,
        factory: Callable[[], tuple[BirdChildCard, ...]],
    ) -> tuple[BirdChildCard, ...]:
        if key not in self._plans:
            self._plans[key] = factory()
        return self._plans[key]

    def child_finished(self, key: GroupKey, child_index: int, expected: int) -> None:
        finished = self._finished.setdefault(key, set())
        finished.add(int(child_index))
        if len(finished) >= expected:
            self._finished.pop(key, None)
            self._plans.pop(key, None)


def ames_seed(
    *, seed: int, step: int, task_key: str, group_fingerprint: str
) -> int:
    return stable_seed(seed, step, task_key, group_fingerprint, "bird-ames-child-plan")
