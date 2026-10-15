#!/usr/bin/env python3
"""Paper metrics from scored BIRD/Spider samples in a simple JSONL interchange."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evorl.analysis.sql_strategy import strategy_key, similarity_sql


@lru_cache(maxsize=131072)
def strategy(sql: str) -> str:
    return strategy_key(sql)


@lru_cache(maxsize=131072)
def pair_similarity(a: str, b: str) -> float:
    return similarity_sql(a, b)


def no_hit(n: int, count: int, k: int) -> float:
    if not 0 <= count <= n or not 1 <= k <= n:
        raise ValueError("Invalid count or k")
    if n - count < k:
        return 0.0
    # Multiplicative hypergeometric ratio avoids enormous binomial integers.
    value = 1.0
    for i in range(k):
        value *= (n - count - i) / (n - i)
    return value


def score_question(rows: list[dict], ks: list[int]) -> dict:
    n = len(rows)
    if n < 2:
        raise ValueError("Effective diversity requires at least two samples")
    good = [r["sql"] for r in rows if r["correct"]]
    counts = Counter(strategy(sql) for sql in good)
    result = {"n": n, "correct": len(good), "strategies": set(counts)}
    for k in ks:
        if k > n:
            raise ValueError(f"k={k} exceeds n={n}")
        result[f"u@{k}"] = sum(1.0 - no_hit(n, c, k) for c in counts.values())
        result[f"pass@{k}"] = 1.0 - no_hit(n, len(good), k)
    sql_counts = Counter(good)
    distinct_sql = list(sql_counts)
    dissimilarity = sum(
        sql_counts[a] * sql_counts[b] * (1.0 - pair_similarity(*sorted((a, b))))
        for i, a in enumerate(distinct_sql) for b in distinct_sql[i + 1:]
    )
    result["effective_diversity"] = 2.0 * dissimilarity / (n * (n - 1))
    return result


def evaluate(rows: list[dict], ks: list[int], base_model: str) -> dict:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        for field in ("model", "question_id", "sample_id", "correct", "sql"):
            if field not in row:
                raise ValueError(f"Missing field {field}")
        if type(row["correct"]) is not bool:
            raise ValueError("correct must be a Boolean")
        key = (str(row["model"]), str(row["question_id"]), str(row["sample_id"]))
        if key in seen:
            raise ValueError(f"Duplicate sample {key}")
        seen.add(key)
        groups[key[:2]].append(row)
    models = sorted({model for model, _ in groups})
    questions = sorted({question for _, question in groups})
    if base_model not in models:
        raise ValueError("Base model missing")
    if any((model, question) not in groups for model in models for question in questions):
        raise ValueError("Every model must have the same questions")
    sizes = {len(sample) for sample in groups.values()}
    if len(sizes) != 1:
        raise ValueError("All model/question groups must have equal sample count")
    scored = {(m, q): score_question(groups[m, q], ks) for m in models for q in questions}
    eligible = [q for q in questions if len(scored[base_model, q]["strategies"]) >= 2]
    output = {"questions": len(questions), "samples_per_question": sizes.pop(),
              "retention_eligible_questions": len(eligible), "strategy_descriptor": "canonical_ast_height4_multiset",
              "similarity": "Dice on height-4 canonical AST subtree multisets", "models": {}}
    for model in models:
        item = {name: sum(scored[model, q][name] for q in questions) / len(questions)
                for name in [*(f"u@{k}" for k in ks), *(f"pass@{k}" for k in ks), "effective_diversity"]}
        item["retention"] = (sum(len(scored[model, q]["strategies"] & scored[base_model, q]["strategies"])
                                  / len(scored[base_model, q]["strategies"]) for q in eligible) / len(eligible)
                             if eligible else None)
        output["models"][model] = item
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="JSONL: model, question_id, sample_id, correct, sql")
    parser.add_argument("--base-model", default="base")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.k or any(k < 1 for k in args.k) or len(set(args.k)) != len(args.k):
        parser.error("--k must list distinct positive integers")
    rows = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    result = evaluate(rows, args.k, args.base_model)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
