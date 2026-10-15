"""Complete SQLite AST identities for U@K, without semantic rewrites.

Retains aliases, identifier quoting, literals, bindings, expression order and all
operators. Ignores comments, source locations and ASCII unquoted identifier case only.
This measures SQL forms, not semantic equivalence or latent reasoning strategies.
Pin SQLGlot as well as AST_SPEC_VERSION when comparing saved results.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from enum import Enum

import sqlglot
from sqlglot import exp

AST_SPEC_VERSION = "sqlite_full_ast_v1"
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _structure(value):
    if isinstance(value, exp.Expression):
        args = {key: _structure(item) for key, item in value.args.items() if item is not None}
        if isinstance(value, exp.Identifier) and not value.args.get("quoted"):
            args["this"] = value.this.translate(_ASCII_LOWER)
        return [value.key, args]
    if isinstance(value, Enum):
        return [type(value).__name__, value.value]
    if isinstance(value, (list, tuple)):
        return [_structure(item) for item in value]
    return value


@lru_cache(maxsize=32768)
def full_ast_key(sql: str) -> str:
    """Hash a complete parsed query; reject empty/multiple/non-query statements.

    Walk AST arguments directly: no SQL serialization/reparse and no optimizer.
    Aliases are retained to avoid changing binding without schema resolution.
    """
    trees = [tree for tree in sqlglot.parse(sql, read="sqlite") if tree is not None]
    if len(trees) != 1 or not isinstance(trees[0], exp.Query):
        raise ValueError("Expected exactly one SQL query")
    payload = [AST_SPEC_VERSION, sqlglot.__version__, _structure(trees[0])]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


ALIAS_AST_SPEC_VERSION = "sqlite_bound_alias_ast_v1"


def canonicalize_relation_aliases(tree):
    """Return a copy with deterministic relation aliases, or None if unresolved.

    SQLGlot scopes resolve qualified columns to the nearest visible source.
    Source positions distinguish self-joins; scope positions distinguish outer
    references from local ones. CTE names and output-column aliases are retained.
    No schema inference, predicate reordering, or semantic SQL rewrites.
    """
    from sqlglot.optimizer.scope import traverse_scope
    value = tree.copy()
    scopes = traverse_scope(value)
    maps, bindings, sources = {}, [], []
    for si, scope in enumerate(scopes):
        mapping = {}
        if scope.udtfs:
            return None  # Table-valued functions need additional binding rules.
        for ri, source in enumerate(scope.tables + scope.derived_tables):
            name = source.alias_or_name.translate(_ASCII_LOWER)
            if not name or name in mapping or source.args.get('pivots'):
                return None
            canonical = f"_uk_s{si}_r{ri}"
            mapping[name] = canonical
            sources.append((source, canonical))
        maps[id(scope)] = mapping
    for scope in scopes:
        for column in scope.find_all(exp.Column):
            if not column.table:
                continue  # Unqualified columns remain intact; no schema guesses.
            if column.db or column.catalog:
                return None
            owner = scope
            name = column.table.translate(_ASCII_LOWER)
            while owner is not None:
                target = maps[id(owner)].get(name)
                if target is not None:
                    bindings.append((column, target))
                    break
                owner = owner.parent if owner.can_be_correlated else None
            else:
                return None
    # Mutate only after resolving every reference against the original names.
    for column, target in bindings:
        column.set('table', exp.to_identifier(target))
    for source, target in sources:
        alias = source.args.get('alias')
        alias = alias.copy() if alias is not None else exp.TableAlias()
        alias.set('this', exp.to_identifier(target))
        source.set('alias', alias)
    return value


@lru_cache(maxsize=32768)
def bound_alias_ast_identity(sql: str) -> tuple[str, bool]:
    """Return (identity, conservative_fallback), preserving full AST on fallback."""
    trees = [t for t in sqlglot.parse(sql, read='sqlite') if t is not None]
    if len(trees) != 1 or not isinstance(trees[0], exp.Query):
        raise ValueError('Expected exactly one SQL query')
    normalized = canonicalize_relation_aliases(trees[0])
    fallback = normalized is None
    # Namespace fallback keys separately to prevent collisions with normalized SQL.
    payload = [ALIAS_AST_SPEC_VERSION, sqlglot.__version__, fallback,
               _structure(trees[0] if fallback else normalized)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(), fallback


def bound_alias_ast_key(sql: str) -> str:
    return bound_alias_ast_identity(sql)[0]
