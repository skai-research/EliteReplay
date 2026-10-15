"""SQL adaptations of AST diversity metrics.

The paper evaluates 32 generations per prompt with pairwise lexical,
canonical-AST, neural, and quality-weighted diversity.  BIRD has one intended
answer per prompt, so execution correctness is the quality function and the
semantic object of interest is the *solution strategy*, not the result table.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Sequence
from typing import Any

import sqlglot
from sqlglot import exp


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sampled_pairs(
    size: int,
    draws: int = 32,
    *,
    seed_parts: Sequence[object] = (),
) -> list[tuple[int, int]]:
    """Sample unordered pairs with replacement, matching the sampled-pair protocol"""

    if size < 2 or draws < 1:
        return []
    population = [(left, right) for left in range(size) for right in range(left + 1, size)]
    rng = random.Random(stable_seed(*seed_parts))
    return [population[rng.randrange(len(population))] for _ in range(draws)]


def ngrams(tokens: Sequence[int | str], order: int) -> list[tuple[int | str, ...]]:
    if order < 1:
        raise ValueError("n-gram order must be positive")
    return [tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]


def expectation_adjusted_distinct(
    token_sequences: Iterable[Sequence[int | str]],
    *,
    order: int = 4,
    unigram_vocabulary_size: int,
) -> float:
    """Expectation-Adjusted Distinct-n from Liu et al. (2022).

    ``V`` in the paper is the number of possible n-grams, hence
    ``unigram_vocabulary_size ** order``.  ``expm1`` keeps the occupancy
    expectation stable when V is much larger than the observed token count.
    """

    grams = [gram for sequence in token_sequences for gram in ngrams(sequence, order)]
    count = len(grams)
    if not count:
        return 0.0
    vocabulary = float(unigram_vocabulary_size**order)
    expected_unique = -vocabulary * math.expm1(count * math.log1p(-1.0 / vocabulary))
    if expected_unique <= 0:
        return 0.0
    return min(1.0, len(set(grams)) / expected_unique)


def _ast_value(value: Any, depth: int) -> Any:
    if isinstance(value, exp.Expression):
        return _ast_fingerprint(value, depth)
    if isinstance(value, list):
        return tuple(_ast_value(item, depth) for item in value)
    if isinstance(value, tuple):
        return tuple(_ast_value(item, depth) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _ast_fingerprint(node: exp.Expression, depth: int) -> tuple[Any, ...]:
    if depth <= 1:
        return (node.key,)
    arguments = []
    for name, value in sorted(node.args.items()):
        if value is None or value == []:
            continue
        arguments.append((name, _ast_value(value, depth - 1)))
    return (node.key, tuple(arguments))


def canonical_ast_subtrees(sql: str, *, height: int = 4) -> list[str]:
    """Return identifier-normalized SQL AST subtrees truncated at ``height``.

    Callers pass the normalized SQL emitted by ``bird_modes.normalized_ast``.
    It canonicalizes aliases/literals/order while retaining schema identifiers,
    which carry real relational meaning in SQL and must not be erased.
    """

    if not sql.strip():
        return []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    return [
        json.dumps(_ast_fingerprint(node, height), separators=(",", ":"), sort_keys=False)
        for node in tree.walk()
    ]


def distinct_cast(
    first_subtrees: Sequence[str], second_subtrees: Sequence[str]
) -> float:
    """Pairwise Distinct-CAST: unique canonical subtrees / all subtrees."""

    total = len(first_subtrees) + len(second_subtrees)
    if not total:
        return 0.0
    return len(set(first_subtrees).union(second_subtrees)) / total


def pair_probability(valid_count: int, population: int = 32) -> float:
    if population < 2 or valid_count < 2:
        return 0.0
    if valid_count > population:
        raise ValueError((valid_count, population))
    return math.comb(valid_count, 2) / math.comb(population, 2)


def pair_key(task_id: str, first: str, second: str) -> str:
    ordered = sorted((first, second))
    payload = json.dumps([task_id, *ordered], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
