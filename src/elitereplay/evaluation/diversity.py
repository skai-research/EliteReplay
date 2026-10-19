#!/usr/bin/env python3
"""Diversity metrics from verified samples and domain-specific strategy descriptors."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys


from elitereplay.evaluation.descriptors.sql_strategy import strategy_key, similarity_sql
from elitereplay.evaluation.descriptors.sql_ast_modes import bound_alias_ast_key
from elitereplay.evaluation.descriptors.structural_descriptor_v2 import describe

DESCRIPTORS = ("coarsened_v2", "bound_alias_ast", "height4_dice", "precomputed")

@lru_cache(maxsize=65536)
def descriptor_key(sql, descriptor):
    if descriptor == "coarsened_v2":
        return describe(sql)["new"]
    if descriptor == "bound_alias_ast":
        return bound_alias_ast_key(sql)
    if descriptor == "height4_dice":
        return strategy_key(sql)
    raise ValueError(f"Unknown descriptor {descriptor}")



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


def sample_descriptor(row, descriptor):
    if descriptor == "precomputed":
        return row["strategy"]
    return descriptor_key(row["sql"], descriptor)


def score_question(rows: list[dict], ks: list[int], descriptor="height4_dice", effective_descriptor=None) -> dict:
    n = len(rows)
    if n < 2:
        raise ValueError("Effective diversity requires at least two samples")
    good = [r for r in rows if r["correct"]]
    counts = Counter(sample_descriptor(row, descriptor) for row in good)
    result = {"n": n, "correct": len(good), "strategies": set(counts)}
    for k in ks:
        if k > n:
            raise ValueError(f"k={k} exceeds n={n}")
        result[f"u@{k}"] = sum(1.0 - no_hit(n, c, k) for c in counts.values())
        result[f"pass@{k}"] = 1.0 - no_hit(n, len(good), k)
    effective_descriptor = effective_descriptor or descriptor
    if effective_descriptor == "height4_dice":
        sql_counts = Counter(row["sql"] for row in good)
        distinct_sql = list(sql_counts)
        dissimilarity = sum(sql_counts[a] * sql_counts[b] * (1.0 - pair_similarity(*sorted((a, b))))
                            for i, a in enumerate(distinct_sql) for b in distinct_sql[i + 1:])
        result["effective_diversity"] = 2.0 * dissimilarity / (n * (n - 1))
    else:
        ed_counts = Counter(sample_descriptor(row, effective_descriptor) for row in good)
        c = len(good)
        result["effective_diversity"] = (c * (c - 1) - sum(v * (v - 1) for v in ed_counts.values())) / (n * (n - 1))
    return result


def evaluate(rows, ks: list[int], base_model: str, descriptor="height4_dice", effective_descriptor=None, bootstrap=0, seed=20261001) -> dict:
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("k must contain distinct positive integers")
    if descriptor not in DESCRIPTORS or (effective_descriptor and effective_descriptor not in DESCRIPTORS):
        raise ValueError("Unknown descriptor")
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    needs_sql = any(d != "precomputed" for d in (descriptor, effective_descriptor or descriptor))
    needs_strategy = "precomputed" in (descriptor, effective_descriptor or descriptor)
    groups = defaultdict(list)
    datasets = set()
    seen = set()
    for row in rows:
        for field in ("model", "question_id", "sample_id", "correct"):
            if field not in row:
                raise ValueError(f"Missing field {field}")
        datasets.add(row.get("dataset", "unspecified"))
        if len(datasets) > 1:
            raise ValueError("Evaluate one dataset at a time")
        if needs_sql and not isinstance(row.get("sql"), str):
            raise ValueError("sql must be a string, including failed attempts")
        if type(row["correct"]) is not bool:
            raise ValueError("correct must be a Boolean")
        if needs_strategy and row["correct"]:
            label = row.get("strategy")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Correct samples require a nonempty strategy string in precomputed mode")
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
    scored = {(m, q): score_question(groups[m, q], ks, descriptor, effective_descriptor) for m in models for q in questions}
    eligible = [q for q in questions if len(scored[base_model, q]["strategies"]) >= 2]
    output = {"questions": len(questions), "samples_per_question": sizes.pop(),
              "retention_eligible_questions": len(eligible), "strategy_descriptor": descriptor,
              "effective_diversity_descriptor": effective_descriptor or descriptor,
              "effective_diversity_similarity": "Dice" if (effective_descriptor or descriptor) == "height4_dice" else "binary identity",
              "units": "probabilities except U@k (expected count)", "models": {}}
    for model in models:
        item = {name: sum(scored[model, q][name] for q in questions) / len(questions)
                for name in [*(f"u@{k}" for k in ks), *(f"pass@{k}" for k in ks), "effective_diversity"]}
        item["retention"] = (sum(len(scored[model, q]["strategies"] & scored[base_model, q]["strategies"])
                                  / len(scored[base_model, q]["strategies"]) for q in eligible) / len(eligible)
                             if eligible else None)
        output["models"][model] = item
    if bootstrap:
        import numpy as np
        rng = np.random.default_rng(seed)
        # Same resampled questions for every method; do not resample individual samples.
        for metric in [*(f"u@{k}" for k in ks), *(f"pass@{k}" for k in ks), "effective_diversity", "retention"]:
            qs = eligible if metric == "retention" else questions
            if not qs:
                continue
            values = np.array([[len(scored[m, q]["strategies"] & scored[base_model, q]["strategies"]) / len(scored[base_model, q]["strategies"])
                                if metric == "retention" else scored[m, q][metric] for m in models] for q in qs])
            estimates = np.empty((bootstrap, len(models)))
            for i in range(bootstrap):
                estimates[i] = values[rng.integers(len(qs), size=len(qs))].mean(axis=0)
            for j, m in enumerate(models):
                output["models"][m].setdefault("ci95", {})[metric] = np.quantile(estimates[:, j], [.025, .975]).tolist()
        output["bootstrap"] = {"replicates": bootstrap, "seed": seed, "unit": "question", "paired_across_models": True,
                               "limitations": "Conditional on saved runs; no training-seed or database-cluster uncertainty; marginal intervals."}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="JSONL: model, question_id, sample_id, correct, plus strategy (precomputed) or sql")
    parser.add_argument("--base-model", default="base")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--descriptor", choices=DESCRIPTORS, default="coarsened_v2", help="U@k and retention identity; precomputed uses supplied strategy labels for any domain")
    parser.add_argument("--effective-descriptor", choices=DESCRIPTORS, default=None, help="Effective-diversity identity/similarity (default: precomputed with precomputed labels, otherwise bound_alias_ast)")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if not args.k or any(k < 1 for k in args.k) or len(set(args.k)) != len(args.k):
        parser.error("--k must list distinct positive integers")
    rows = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    effective_descriptor = args.effective_descriptor or (
        "precomputed" if args.descriptor == "precomputed" else "bound_alias_ast"
    )
    result = evaluate(rows, args.k, args.base_model, args.descriptor, effective_descriptor, args.bootstrap, args.seed)
    import hashlib
    import importlib.metadata
    result["input_sha256"] = hashlib.sha256(args.samples.read_bytes()).hexdigest()
    if args.descriptor != "precomputed" or effective_descriptor != "precomputed":
        result["sqlglot_version"] = importlib.metadata.version("sqlglot")
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
