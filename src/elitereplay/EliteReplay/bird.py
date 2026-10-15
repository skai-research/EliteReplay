"""Strict BIRD adapter for the offline EliteReplay archive search."""

from __future__ import annotations

import json
from typing import Any, Mapping

from sqlglot import exp, parse
from sqlglot.errors import ParseError, TokenError
from sqlglot.tokens import TokenType, Tokenizer

from elitereplay.EliteReplay.domain import (
    DomainInfrastructureError,
    ParseResult,
    VerificationResult,
    item_value,
    normalized_messages,
    strict_envelope_error,
)
from elitereplay.reward import bird_reward


BIRD_SYSTEM_PROMPT = (
    "You are solving a BIRD text-to-SQL task against one SQLite database. "
    "Return exactly one complete read-only SQLite query that answers the "
    "question. Use the evidence and complete schema supplied by the user. "
    "Output only SQL, without Markdown fences, analysis, or explanation."
)


def parse_bird_sql_strict(
    raw_response: str,
    *,
    max_response_bytes: int = 32_768,
) -> ParseResult:
    """Accept exactly one complete, comment-free, read-only SQLite query.

    Unlike the legacy reward extractor, this parser never selects a suffix,
    removes a fence, or truncates at a semicolon.  The complete accepted text
    is the text handed to SQLite and later exported to SFT.
    """

    error = strict_envelope_error(
        raw_response, max_response_bytes=max_response_bytes
    )
    if error is not None:
        return ParseResult.reject(error)
    candidate = raw_response.strip()

    try:
        tokens = Tokenizer(dialect="sqlite").tokenize(candidate)
    except Exception:
        return ParseResult.reject("sql_tokenization_error")
    if not tokens or tokens[0].token_type not in {TokenType.SELECT, TokenType.WITH}:
        return ParseResult.reject("not_a_read_query")
    # Comments make it impossible to distinguish SQL annotation from hidden
    # leading/trailing prose.  They are unnecessary for an executable answer.
    if any(getattr(token, "comments", None) for token in tokens):
        return ParseResult.reject("sql_comments_not_allowed")
    semicolons = [index for index, token in enumerate(tokens) if token.token_type == TokenType.SEMICOLON]
    if len(semicolons) > 1 or (semicolons and semicolons[0] != len(tokens) - 1):
        return ParseResult.reject("multiple_statements_or_trailing_material")

    try:
        expressions = parse(candidate, read="sqlite")
    except (TokenError, ParseError, ValueError, TypeError, RecursionError, MemoryError):
        return ParseResult.reject("invalid_sql")
    parsed = [expression for expression in expressions if expression is not None]
    if len(parsed) != 1:
        return ParseResult.reject("multiple_statements_or_trailing_material")
    expression = parsed[0]
    if not isinstance(expression, exp.Query):
        return ParseResult.reject("not_a_read_query")

    # This is defensive in addition to checking the root expression.  It
    # prevents dialect extensions from embedding side-effecting statements in
    # an otherwise query-shaped AST.
    forbidden = tuple(
        expression_type
        for name in ("Insert", "Update", "Delete", "Create", "Drop", "Alter", "Merge", "Command")
        if (expression_type := getattr(exp, name, None)) is not None
    )
    if forbidden and any(expression.find(expression_type) for expression_type in forbidden):
        return ParseResult.reject("side_effecting_sql")
    return ParseResult.accept(candidate)


class BirdAdapter:
    """EliteReplay adapter backed by the repository's pinned BIRD EX worker."""

    verifier_version = (
        "bird-ex:"
        f"{bird_reward.PINNED_BIRD_EVALUATOR_COMMIT}:"
        f"{bird_reward.PINNED_BIRD_EX_SHA256}"
    )

    def __init__(
        self,
        *,
        max_response_bytes: int = 32_768,
        system_prompt: str = BIRD_SYSTEM_PROMPT,
        **_: Any,
    ) -> None:
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.max_response_bytes = int(max_response_bytes)
        self.system_prompt = str(system_prompt)

    def format_prompt(self, item: Any) -> list[dict[str, str]]:
        messages = normalized_messages(item_value(item, "prompt"))
        if not any(message["role"] == "system" for message in messages):
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def parse_strict(self, raw_response: str) -> ParseResult:
        return parse_bird_sql_strict(
            raw_response, max_response_bytes=self.max_response_bytes
        )

    def verify(self, item: Any, canonical_response: str) -> VerificationResult:
        parsed = self.parse_strict(canonical_response)
        if not parsed.accepted:
            return VerificationResult.rejected(f"strict_parse:{parsed.error}")
        if parsed.canonical_response != canonical_response:
            return VerificationResult.rejected("response_is_not_canonical")

        reward_model = _mapping(item_value(item, "reward_model"), "reward_model")
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError as exc:
                raise DomainInfrastructureError(
                    "BIRD ground truth is invalid JSON"
                ) from exc
        if not isinstance(ground_truth, Mapping):
            raise DomainInfrastructureError("BIRD item has no ground truth")
        extra_info = _mapping(item_value(item, "extra_info", {}), "extra_info")
        db_id = str(extra_info.get("db_id") or ground_truth.get("db_id") or "")
        gold_sql = str(ground_truth.get("gold_sql") or "")
        if not gold_sql:
            raise DomainInfrastructureError(f"BIRD item has no gold SQL for {db_id!r}")

        try:
            bird_reward.validate_official_evaluator()
            database_path = bird_reward._database_path(db_id)
            result = bird_reward._worker_result(
                predicted_sql=canonical_response,
                ground_truth_sql=gold_sql,
                db_path=database_path,
            )
        except bird_reward.BirdInfrastructureError as exc:
            raise DomainInfrastructureError(str(exc)) from exc
        correct = bool(result.get("execution_correct", False))
        executable = bool(result.get("executable", False))
        evaluator_error = str(result.get("error") or result.get("worker_error") or "")
        return VerificationResult(
            score=1.0 if correct else 0.0,
            fully_correct=correct,
            executable=executable,
            per_test_scores=(1.0 if correct else 0.0,),
            error=evaluator_error or None,
            metadata={
                "db_id": db_id,
                "predicted_rows": int(result.get("predicted_rows", 0)),
                "ground_truth_rows": int(result.get("ground_truth_rows", 0)),
                "verifier_version": self.verifier_version,
            },
        )

    def format_labeler_input(self, item: Any, canonical_response: str) -> str:
        canonical = self.sanitize_for_sft(canonical_response)
        prompt = json.dumps(
            self.format_prompt(item), ensure_ascii=False, sort_keys=True
        )
        return f"Problem messages:\n{prompt}\n\nCandidate SQL:\n{canonical}"

    def sanitize_for_sft(self, canonical_response: str) -> str:
        parsed = self.parse_strict(canonical_response)
        if not parsed.accepted:
            raise ValueError(f"BIRD response failed strict parsing: {parsed.error}")
        if parsed.canonical_response != canonical_response:
            raise ValueError("BIRD response is not in canonical form")
        return canonical_response


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "as_py"):
        plain = value.as_py()
        if isinstance(plain, Mapping):
            return plain
    raise DomainInfrastructureError(f"BIRD item has malformed {field_name}")
