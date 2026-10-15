"""Read-only archive diagnostics; fingerprints are proxies, not semantic proofs.

Canonical identities used for export preserve identifiers and literal values.
The independent structure views deliberately abstract some of those details.
Nothing in this module executes a candidate or contacts a model.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .archive import MapElitesArchive, load_frozen_archive
from .manifests import sha256_file
from .schema import CandidateRecord, content_hash


def canonical_artifact_key(domain: str, text: str) -> str:
    """Conservative syntax identity; no execution, alias or literal erasure."""
    if domain == "bird":
        import sqlglot

        try:
            trees = sqlglot.parse(text, read="sqlite")
        except sqlglot.errors.ParseError as exc:
            raise ValueError(f"invalid SQL: {exc}") from exc
        if len(trees) != 1 or trees[0] is None:
            raise ValueError("expected one SQL statement")
        representation = trees[0].sql(dialect="sqlite", pretty=False, comments=False)
    elif domain in {"taco", "optmath"}:
        representation = ast.dump(ast.parse(text), include_attributes=False)
    else:
        raise ValueError(f"unsupported archive domain: {domain}")
    return content_hash({"domain": domain, "canonical_ast": representation})


def _python_shape(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        return ("Name", type(node.ctx).__name__)
    if isinstance(node, ast.Constant):
        return ("Constant", type(node.value).__name__)
    if isinstance(node, ast.alias):
        return (
            "alias",
            node.name,
        )  # retain the imported library, ignore its local alias
    fields = []
    for name, value in ast.iter_fields(node):
        if name == "type_comment":
            continue
        if name == "name" and isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler),
        ):
            continue
        if name == "arg" and isinstance(node, ast.arg):
            continue
        if isinstance(value, ast.AST):
            value = _python_shape(value)
        elif isinstance(value, list):
            value = [_python_shape(x) if isinstance(x, ast.AST) else x for x in value]
        fields.append((name, value))
    return (type(node).__name__, fields)


def fingerprints(domain: str, text: str) -> dict[str, str | None]:
    canonical = canonical_artifact_key(domain, text)
    if domain == "optmath":
        # Code shape does not establish a mathematical strategy. Trusted model
        # fingerprints are joined separately by audit_strategy_evidence.
        return {"canonical_ast": canonical, "structure": None}
    if domain == "bird":
        from evorl.analysis.bird_modes import extract_bird_modes

        parsed = extract_bird_modes(text)
        if not parsed.parse_ok or parsed.structural_mode_id is None:
            raise ValueError("SQL structure extraction failed")
        structure = parsed.structural_mode_id
    else:
        # Shape preserves control/operator structure, attribute names and
        # imports, but ignores local names and literal values. This can merge
        # different algorithms and split equivalent ones; audit it manually.
        structure = content_hash(_python_shape(ast.parse(text)))
    return {"canonical_ast": canonical, "structure": structure}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _view_counts(rows: list[dict[str, Any]], view: str) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get(view) is not None:
            result[row["problem_id"]][str(row[view])] += 1
    return dict(result)


def _mode_set(rows: list[dict[str, Any]], view: str) -> set[tuple[str, str]]:
    return {
        (p, mode) for p, counts in _view_counts(rows, view).items() for mode in counts
    }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize_rows(rows: list[dict[str, Any]], problem_ids: set[str]) -> dict[str, Any]:
    represented = {row["problem_id"] for row in rows}
    result: dict[str, Any] = {
        "examples": len(rows),
        "represented_problems": len(represented),
        "attempted_problems": len(problem_ids),
        "problem_coverage": len(represented) / len(problem_ids)
        if problem_ids
        else None,
        "examples_per_represented_problem": len(rows) / len(represented)
        if represented
        else None,
        "response_tokens": sum(row.get("output_tokens", 0) for row in rows),
        "views": {},
    }
    for view in ("primary_strategy", "canonical_ast", "structure"):
        counts_by_problem = _view_counts(rows, view)
        histogram = Counter(len(counts_by_problem.get(p, {})) for p in problem_ids)
        conditional = []
        effective_modes = []
        for counts in counts_by_problem.values():
            n = sum(counts.values())
            probabilities = [v / n for v in counts.values()]
            effective_modes.append(
                math.exp(-sum(p * math.log(p) for p in probabilities))
            )
            if n >= 2:
                conditional.append(
                    1 - sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))
                )
        multi = sum(len(counts_by_problem.get(p, {})) >= 2 for p in represented)
        result["views"][view] = {
            "distinct_problem_modes": sum(len(c) for c in counts_by_problem.values()),
            "rows_with_fingerprint": sum(
                sum(c.values()) for c in counts_by_problem.values()
            ),
            "modes_per_represented_problem": (
                sum(len(c) for c in counts_by_problem.values()) / len(represented)
                if represented
                else None
            ),
            "mode_count_histogram_all_problems": dict(sorted(histogram.items())),
            "multimode_problems": multi,
            "multimode_fraction_represented": multi / len(represented)
            if represented
            else None,
            "pair_difference_uniform_rows": _mean(conditional),
            "pair_difference_problem_support": len(conditional),
            "mean_exp_entropy_uniform_rows": _mean(effective_modes),
        }
    return result


def audit_archive(
    root: Path,
    *,
    export_path: Path | None = None,
    target_min: int = 1000,
    target_max: int = 2000,
    per_problem_cap: int = 4,
    replay_draws: int = 6240,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        not 0 <= target_min <= target_max
        or target_max < 1
        or per_problem_cap < 1
        or replay_draws < 0
    ):
        raise ValueError("invalid audit targets")
    candidate_path = root / "archive/candidates.jsonl"
    archive_path = root / "archive/archive.json"
    taxonomy_path = root / "taxonomy/taxonomy.json"
    raw = _read_jsonl(candidate_path)
    taxonomy = json.loads(taxonomy_path.read_text())
    known_labels = {item["id"] for item in taxonomy["labels"]}
    archive, document = load_frozen_archive(
        archive_path, expected_taxonomy_hash=taxonomy["taxonomy_hash"]
    )
    domain = document["domain"]
    problems = {r["problem_id"] for r in raw}
    errors: list[str] = []
    warnings: list[str] = []
    if sha256_file(candidate_path) != document["candidate_log_sha256"]:
        errors.append("candidate journal hash differs from frozen archive")
    taxonomy_content = {k: v for k, v in taxonomy.items() if k != "taxonomy_hash"}
    if content_hash(taxonomy_content) != taxonomy["taxonomy_hash"]:
        errors.append("taxonomy content hash mismatch")
    replayed = MapElitesArchive()
    try:
        replayed.replay(CandidateRecord.from_dict(r) for r in raw)
        if replayed.to_dict() != archive.to_dict():
            errors.append("journal replay does not reproduce the frozen archive")
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(f"journal replay failed: {exc}")
    ids = [r["candidate_id"] for r in raw]
    slots = [r["slot_key"] for r in raw]
    if len(set(ids)) != len(ids) or len(set(slots)) != len(slots):
        errors.append("duplicate candidate identity or generation slot")
    seen: dict[str, dict[str, Any]] = {}
    for row in raw:
        if row.get("taxonomy_hash") != taxonomy["taxonomy_hash"]:
            errors.append(f"candidate taxonomy mismatch: {row['candidate_id']}")
        parent_id = row.get("parent_candidate_id")
        if parent_id:
            parent = seen.get(parent_id)
            if (
                parent is None
                or not parent.get("accepted_into_archive")
                or parent["problem_id"] != row["problem_id"]
            ):
                errors.append(f"invalid parent lineage: {row['candidate_id']}")
        if (
            row.get("primary_strategy") is not None
            and row["primary_strategy"] not in known_labels
        ):
            errors.append(f"unknown realized label: {row['candidate_id']}")
        if row.get("secondary_strategy") is not None:
            errors.append(
                f"unexpected secondary strategy label: {row['candidate_id']}"
            )
        seen[row["candidate_id"]] = row

    cache: dict[str, dict[str, str | None]] = {}
    parse_failures: list[str] = []

    def enrich(row: dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        text = item.get("canonical_response") or ""
        if text not in cache:
            try:
                cache[text] = fingerprints(domain, text)
            except (ValueError, SyntaxError, RecursionError) as exc:
                cache[text] = {}
                parse_failures.append(f"{item['candidate_id']}: {exc}")
        item.update(cache[text])
        return item

    correct = [enrich(r) for r in raw if r.get("fully_correct")]
    seed_rows = [r for r in correct if r["method"] == "seed"]
    elites = [
        enrich(c.to_dict()) for p in archive.problem_ids for c in archive.elites_for(p)
    ]
    elite_ids = {r["candidate_id"] for r in elites}
    if export_path is None:
        export_path = root / "sft/train.jsonl"
    exported: list[dict[str, Any]] = []
    if export_path.is_file():
        for row in _read_jsonl(export_path):
            meta = row["metadata"]
            candidate = seen.get(meta.get("candidate_id"))
            if candidate is None:
                errors.append(
                    f"export references unknown candidate: {meta.get('candidate_id')}"
                )
                continue
            messages = row["messages"]
            if (
                not messages
                or messages[-1].get("role") != "assistant"
                or messages[-1].get("content") != candidate["canonical_response"]
            ):
                errors.append(
                    f"export target differs from verified candidate: {candidate['candidate_id']}"
                )
            if (
                meta.get("problem_id") != candidate["problem_id"]
                or meta.get("primary_strategy") != candidate["primary_strategy"]
                or (candidate.get("structural_subtype") is not None and (
                    meta.get("structural_subtype") != candidate["structural_subtype"]
                    or meta.get("mode_id") != candidate["niche"]))
            ):
                errors.append(f"export metadata mismatch: {candidate['candidate_id']}")
            if (
                meta.get("archive_hash") != document["archive_hash"]
                or meta.get("taxonomy_hash") != taxonomy["taxonomy_hash"]
            ):
                errors.append(
                    f"export provenance mismatch: {candidate['candidate_id']}"
                )
            if (
                not candidate.get("fully_correct")
                or not meta.get("fully_correct")
                or candidate["candidate_id"] not in elite_ids
            ):
                errors.append(
                    f"export contains a non-elite or incorrect candidate: {candidate['candidate_id']}"
                )
            exported.append(enrich(candidate))
    else:
        warnings.append(
            "training export missing; export diversity has not been measured"
        )
    if parse_failures:
        errors.append(
            f"structure extraction failed for {len(parse_failures)} distinct correct artifacts"
        )
    canonical_keys = [(r["problem_id"], r.get("canonical_ast")) for r in exported]
    duplicate_rows = len(canonical_keys) - len(set(canonical_keys))
    if duplicate_rows:
        warnings.append(
            f"export has {duplicate_rows} duplicate canonical-AST rows within questions"
        )
    if any(
        n > per_problem_cap for n in Counter(r["problem_id"] for r in exported).values()
    ):
        errors.append("export exceeds the per-problem cap")
    target_met = export_path.is_file() and target_min <= len(exported) <= target_max
    if not target_met:
        warnings.append(
            f"export size {len(exported)} is outside the {target_min}–{target_max} target; never pad"
        )

    stages = {
        "correct_seeds": seed_rows,
        "correct_candidate_pool": correct,
        "map_elites": elites,
        "training_export": exported,
    }
    stats = {name: summarize_rows(rows, problems) for name, rows in stages.items()}
    pool_modes = _mode_set(correct, "structure")
    seed_modes = _mode_set(seed_rows, "structure")
    retention = {}
    for name in ("map_elites", "training_export"):
        modes = _mode_set(stages[name], "structure")
        retention[name] = {
            "retained_pool_structures": len(modes & pool_modes),
            "pool_structures": len(pool_modes),
            "fraction": len(modes & pool_modes) / len(pool_modes)
            if pool_modes
            else None,
        }
    mutations = [r for r in raw if r["method"] == "qd_mutation"]
    target_counts = Counter(
        (r["problem_id"], r.get("requested_strategy")) for r in mutations
    )
    feasible_path = root / "archive/feasible_strategies.jsonl"
    feasible_pairs = (
        {
            (r["problem_id"], s)
            for r in _read_jsonl(feasible_path)
            for s in r["strategy_ids"]
        }
        if feasible_path.is_file()
        else set()
    )
    hits = sum(
        r.get("primary_strategy") == r.get("requested_strategy")
        for r in mutations
        if r.get("requested_strategy") is not None
    )
    # Same-label different structures quantify descriptor coarseness, not errors.
    within_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in correct:
        if row.get("primary_strategy") and row.get("structure"):
            within_label[(row["problem_id"], row["primary_strategy"])].add(
                row["structure"]
            )
    question_rows = []
    stage_counts = {
        name: _view_counts(rows, "structure") for name, rows in stages.items()
    }
    for problem in sorted(problems):
        entry: dict[str, Any] = {"problem_id": problem}
        for name, rows in stages.items():
            local = [r for r in rows if r["problem_id"] == problem]
            entry[name + "_examples"] = len(local)
            entry[name + "_structures"] = len(stage_counts[name].get(problem, {}))
        entry["structures_lost_before_training"] = len(
            set(stage_counts["correct_candidate_pool"].get(problem, {}))
            - set(stage_counts["training_export"].get(problem, {}))
        )
        question_rows.append(entry)
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exported:
        by_problem[row["problem_id"]].append(row)
    review = []
    for problem, rows in sorted(by_problem.items()):
        for a, b in combinations(rows, 2):
            same_structure = a.get("structure") == b.get("structure")
            same_label = a["primary_strategy"] == b["primary_strategy"]
            review.append(
                {
                    "problem_id": problem,
                    "candidate_a": a["candidate_id"],
                    "candidate_b": b["candidate_id"],
                    "same_structure": same_structure,
                    "same_label": same_label,
                    "sql_or_code_a": a["canonical_response"],
                    "sql_or_code_b": b["canonical_response"],
                    "meaningfully_different": "",
                    "both_credible": "",
                    "review_reason": "",
                }
            )
    review.sort(
        key=lambda r: content_hash(
            {"seed": 42, "a": r["candidate_a"], "b": r["candidate_b"]}
        )
    )
    # One pair per question first, so prolific questions do not dominate review.
    selected_review, reviewed_problems = [], set()
    for row in review:
        if row["problem_id"] not in reviewed_problems:
            selected_review.append(row)
            reviewed_problems.add(row["problem_id"])
        if len(selected_review) == 50:
            break
    exposures = (
        [replay_draws / (len(by_problem) * len(rows)) for rows in by_problem.values()]
        if by_problem
        else []
    )
    summary = {
        "schema_version": "EliteReplay.archive_audit.v1",
        "domain": domain,
        "interpretation": "Recorded verification only; no SQL/code re-execution and no semantic truth labels. Pair diversity uses uniform rows within each question, not model samples.",
        "fingerprints": {
            "canonical_ast": "syntax identity preserving identifiers and literals",
            "structure": "frozen BIRD structural signature v2"
            if domain == "bird"
            else "Unavailable from source code; use the separate trusted model evidence audit, which does not prove formulation equivalence" if domain == "optmath"
            else "Python AST shape v1; names/literal values abstracted; not algorithm identity",
        },
        "target": {
            "min_examples": target_min,
            "max_examples": target_max,
            "max_per_problem": per_problem_cap,
            "met": target_met,
        },
        "integrity_pass": not errors,
        "errors": errors,
        "warnings": warnings,
        "candidate_attempts": len(raw),
        "attempts_by_method": dict(Counter(r["method"] for r in raw)),
        "fully_correct_by_method": dict(Counter(r["method"] for r in correct)),
        "length_finished_correct_candidates": sum(
            r.get("finish_reason") == "length" for r in correct
        ),
        "stages": stats,
        "structural_retention": retention,
        "discovery": {
            "mutation_added_structures": len(pool_modes - seed_modes),
            "mutation_rescued_problems": len(
                {r["problem_id"] for r in correct}
                - {r["problem_id"] for r in seed_rows}
            ),
        },
        "targeting": {
            "mutations": len(mutations),
            "realized_target_matches": hits,
            "target_match_rate_all_mutations": hits / len(mutations)
            if mutations
            else None,
            "feasible_pairs_never_targeted": len(feasible_pairs - set(target_counts)),
            "max_attempts_same_problem_target": max(target_counts.values(), default=0),
        },
        "descriptor_coarseness": {
            "problem_labels_with_multiple_structures": sum(
                len(s) > 1 for s in within_label.values()
            ),
            "labeled_problem_cells_in_correct_pool": len(within_label),
        },
        "export_canonical_duplicate_rows": duplicate_rows,
        "replay_exposure": {
            "draws": replay_draws,
            "mean_draws_per_example": replay_draws / len(exported)
            if exported
            else None,
            "expected_min_under_problem_balanced_sampler": min(exposures, default=None),
            "expected_max_under_problem_balanced_sampler": max(exposures, default=None),
        },
        "parse_failures": parse_failures,
        "source_sha256": {
            str(p): sha256_file(p)
            for p in (candidate_path, archive_path, taxonomy_path, export_path)
            if p.is_file()
        },
    }
    return summary, question_rows, selected_review


def write_audit(
    output: Path,
    summary: dict[str, Any],
    questions: list[dict[str, Any]],
    review: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for filename, rows in (
        ("per_problem.csv", questions),
        ("manual_review.tsv", review),
    ):
        if not rows:
            (output / filename).write_text("")
            continue
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                delimiter="\t" if filename.endswith("tsv") else ",",
            )
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Archive diversity audit",
        "",
        summary["interpretation"],
        "",
        f"Integrity checks: **{'PASS' if summary['integrity_pass'] else 'FAIL'}**. Size target met: **{summary['target']['met']}**.",
        "",
        "| Stage | Examples | Questions | Structural modes | Questions with ≥2 structures |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stage in summary["stages"].items():
        view = stage["views"]["structure"]
        lines.append(
            f"| {name} | {stage['examples']} | {stage['represented_problems']} | {view['distinct_problem_modes']} | {view['multimode_problems']} |"
        )
    lines += [
        "",
        "Structural modes are task-specific automatic proxies. The size target is advisory; it is not a certificate of semantic diversity.",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {message}" for message in summary["errors"] + summary["warnings"])
    lines += [
        "",
        "See summary.json for pair diversity, entropy, retention, target adherence, and replay exposure; per_problem.csv for question-level losses. Fill manual_review.tsv using question/schema or problem context from the original dataset. Review labels are intentionally blank.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
