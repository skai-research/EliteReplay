"""Frozen BIRD SQL mode extraction used by the three-arm diversity audit.

The primary signature describes logical SQL structure and deliberately ignores
surface realization.  The AMES signatures reuse the descriptor that was frozen
before training, while the normalized AST hash is a finer sensitivity view.
"""

from __future__ import annotations

import hashlib
import json
import functools
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import sqlglot
from sqlglot import exp

from evorl.reward.bird_reward import extract_sql
from evorl.verl.bird_ames import realized_descriptor


MODE_SPEC_VERSION = 2
SQLGLOT_VERSION = sqlglot.__version__


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def signature_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:20]


def _name(value: Any) -> str:
    return str(value or "").strip('`"[]').casefold()


def _nearest_select(node: exp.Expression | None) -> exp.Select | None:
    current = node.parent if node is not None else None
    while current is not None:
        if isinstance(current, exp.Select):
            return current
        current = current.parent
    return None


def _direct_nodes(
    select: exp.Select, kind: type[exp.Expression]
) -> list[exp.Expression]:
    return [node for node in select.find_all(kind) if _nearest_select(node) is select]


def _table_name(table: exp.Table) -> str:
    parts = [_name(table.catalog), _name(table.db), _name(table.name)]
    return ".".join(part for part in parts if part)


def _counter_items(values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(values).items()))


def _scope_maps(
    tree: exp.Expression,
) -> tuple[
    dict[int, dict[str, str]],
    dict[str, str],
    dict[int, str],
]:
    cte_targets = {
        _name(cte.alias_or_name): f"cte:{index}"
        for index, cte in enumerate(tree.find_all(exp.CTE))
        if cte.alias_or_name
    }
    subquery_targets = {
        id(subquery): f"derived:{index}"
        for index, subquery in enumerate(tree.find_all(exp.Subquery))
    }
    aliases: dict[int, dict[str, str]] = {}
    for table in tree.find_all(exp.Table):
        scope = _nearest_select(table)
        if scope is None:
            continue
        raw_name = _table_name(table)
        target = cte_targets.get(_name(table.name), raw_name)
        scope_aliases = aliases.setdefault(id(scope), {})
        alias = _name(table.alias_or_name)
        if alias:
            scope_aliases[alias] = target
        if not table.alias:
            scope_aliases[_name(table.name)] = target
    for subquery in tree.find_all(exp.Subquery):
        if not subquery.alias_or_name:
            continue
        scope = _nearest_select(subquery)
        if scope is not None:
            aliases.setdefault(id(scope), {})[_name(subquery.alias_or_name)] = (
                subquery_targets[id(subquery)]
            )
    return aliases, cte_targets, subquery_targets


def _outer_selects(select: exp.Select) -> list[exp.Select]:
    result: list[exp.Select] = []
    current = _nearest_select(select)
    while current is not None:
        result.append(current)
        current = _nearest_select(current)
    return result


def _resolve_column(
    column: exp.Column,
    aliases: dict[int, dict[str, str]],
) -> str:
    column_name = _name(column.name)
    qualifier = _name(column.table)
    if not qualifier:
        return f"?.{column_name}"
    scope = _nearest_select(column)
    while scope is not None:
        target = aliases.get(id(scope), {}).get(qualifier)
        if target:
            return f"{target}.{column_name}"
        scope = _nearest_select(scope)
    return f"unknown:{qualifier}.{column_name}"


def _select_role(select: exp.Select) -> str:
    current: exp.Expression | None = select.parent
    saw_subquery = False
    while current is not None and not isinstance(current, exp.Select):
        if isinstance(current, exp.CTE):
            return "cte"
        if isinstance(current, exp.Exists):
            return "exists"
        if isinstance(current, exp.In):
            return "in_subquery"
        if isinstance(current, exp.Subquery):
            saw_subquery = True
            if isinstance(current.parent, (exp.From, exp.Join)):
                return "derived_table"
        if isinstance(current, (exp.Union, exp.Intersect, exp.Except)):
            return "set_branch"
        current = current.parent
    if saw_subquery:
        return "scalar_or_predicate_subquery"
    return "root"


def _is_correlated(
    select: exp.Select,
    aliases: dict[int, dict[str, str]],
) -> bool:
    local = aliases.get(id(select), {})
    outer_aliases: set[str] = set()
    for outer in _outer_selects(select):
        outer_aliases.update(aliases.get(id(outer), {}))
    if not outer_aliases:
        return False
    for column in _direct_nodes(select, exp.Column):
        qualifier = _name(column.table)
        if qualifier and qualifier not in local and qualifier in outer_aliases:
            return True
    return False


def _block_signature(
    select: exp.Select,
    aliases: dict[int, dict[str, str]],
    cte_targets: dict[str, str],
) -> dict[str, Any]:
    tables = []
    for table in _direct_nodes(select, exp.Table):
        raw = _table_name(table)
        tables.append(cte_targets.get(_name(table.name), raw))
    aggregates = _direct_nodes(select, exp.AggFunc)
    windows = _direct_nodes(select, exp.Window)
    group = select.args.get("group")
    return {
        "role": _select_role(select),
        "relations": sorted(set(tables)),
        "join_count": len(_direct_nodes(select, exp.Join)),
        "aggregate_functions": sorted(
            {type(node).__name__.casefold() for node in aggregates}
        ),
        "group_key_count": len(group.expressions) if group is not None else 0,
        "conditional_aggregation": any(node.find(exp.Case) for node in aggregates),
        "aggregate_distinct": any(node.find(exp.Distinct) for node in aggregates),
        "window_count": len(windows),
        "distinct": select.args.get("distinct") is not None,
        "ordered": select.args.get("order") is not None,
        "limited": select.args.get("limit") is not None,
        "correlated": _is_correlated(select, aliases),
    }


def structural_signature(tree: exp.Expression) -> dict[str, Any]:
    """Return the frozen, task-relative logical structure signature."""

    selects = list(tree.find_all(exp.Select))
    aliases, cte_targets, _ = _scope_maps(tree)
    cte_names = set(cte_targets)

    base_relations = sorted(
        {
            _table_name(table)
            for table in tree.find_all(exp.Table)
            if _name(table.name) not in cte_names
        }
    )

    join_edges: set[tuple[str, str]] = set()
    for equal in tree.find_all(exp.EQ):
        scope = _nearest_select(equal)
        if scope is None:
            continue
        left = equal.this
        right = equal.expression
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            continue
        first = _resolve_column(left, aliases)
        second = _resolve_column(right, aliases)
        first_relation = first.rsplit(".", 1)[0]
        second_relation = second.rsplit(".", 1)[0]
        if first_relation != second_relation:
            join_edges.add(tuple(sorted((first, second))))

    directional_joins: list[tuple[str, str]] = []
    for join in tree.find_all(exp.Join):
        scope = _nearest_select(join)
        side = _name(join.side)
        kind = _name(join.kind)
        join_type = "_".join(value for value in (side, kind) if value) or "inner"
        right = join.this
        if isinstance(right, exp.Table):
            right_target = aliases.get(id(scope), {}).get(
                _name(right.alias_or_name), _table_name(right)
            )
        elif isinstance(right, exp.Subquery):
            right_target = aliases.get(id(scope), {}).get(
                _name(right.alias_or_name), "derived"
            )
        else:
            right_target = type(right).__name__.casefold()
        has_equality = bool(join.args.get("on") and join.args["on"].find(exp.EQ))
        # A comma join parsed as CROSS plus an equality in WHERE is normalized
        # by the global equality graph; a true cross join remains explicit.
        if join_type not in {"inner", "cross"} or not has_equality:
            if join_type != "cross" or not any(
                right_target in endpoint
                for edge in join_edges
                for endpoint in edge
            ):
                directional_joins.append((join_type, right_target))

    roles = [_select_role(select) for select in selects]
    nested_roles = {
        "derived_table",
        "exists",
        "in_subquery",
        "scalar_or_predicate_subquery",
    }
    if cte_targets:
        query_form = "common_table_expression"
    elif any(role in nested_roles for role in roles):
        query_form = "nested_subquery"
    else:
        query_form = "flat_query"

    aggregate_nodes = list(tree.find_all(exp.AggFunc))
    windows = list(tree.find_all(exp.Window))
    set_operations = []
    for node in tree.find_all((exp.Union, exp.Intersect, exp.Except)):
        suffix = "distinct" if node.args.get("distinct") is not False else "all"
        set_operations.append(f"{type(node).__name__.casefold()}_{suffix}")

    predicates: set[str] = set()
    if any(tree.find_all(exp.Between)):
        predicates.add("between")
    if any(tree.find_all(exp.Like)):
        predicates.add("like")
    ilike_type = getattr(exp, "ILike", None)
    if ilike_type is not None and any(tree.find_all(ilike_type)):
        predicates.add("ilike")
    if any(node.find(exp.Null) for node in tree.find_all(exp.Is)):
        predicates.add("null_test")
    if any(tree.find_all(exp.Having)):
        predicates.add("having")
    if any(tree.find_all(exp.Exists)):
        predicates.add("exists")
    for node in tree.find_all(exp.In):
        predicates.add("in_subquery" if node.args.get("query") else "in_list")

    rank_names = {"rownumber", "rank", "denserank", "ntile", "percentrank"}
    rank_like = any(_name(type(window.this).__name__) in rank_names for window in windows)

    return {
        "version": MODE_SPEC_VERSION,
        "base_relations": base_relations,
        "join_edges": [list(edge) for edge in sorted(join_edges)],
        "directional_joins": [list(value) for value in sorted(directional_joins)],
        "query_form": query_form,
        "query_blocks": len(selects),
        "subquery_roles": [list(value) for value in _counter_items(roles)],
        "correlated_subqueries": sum(
            _is_correlated(select, aliases)
            for select in selects
            if _select_role(select) in nested_roles
        ),
        "blocks": sorted(
            (_block_signature(select, aliases, cte_targets) for select in selects),
            key=canonical_json,
        ),
        "aggregate_functions": sorted(
            {type(node).__name__.casefold() for node in aggregate_nodes}
        ),
        "conditional_aggregation": any(
            node.find(exp.Case) for node in aggregate_nodes
        ),
        "aggregate_distinct": any(
            node.find(exp.Distinct) for node in aggregate_nodes
        ),
        "grouped": any(select.args.get("group") is not None for select in selects),
        "window": bool(windows),
        "set_operations": sorted(set_operations),
        "explicit_distinct": any(
            select.args.get("distinct") is not None for select in selects
        ),
        "ordered_blocks": sum(select.args.get("order") is not None for select in selects),
        "limited_blocks": sum(select.args.get("limit") is not None for select in selects),
        "rank_like_window": rank_like,
        "predicate_mechanisms": sorted(predicates),
    }


def _flatten_and(node: exp.Expression) -> list[exp.Expression]:
    if isinstance(node, exp.And):
        return _flatten_and(node.this) + _flatten_and(node.expression)
    return [node]


def normalized_ast(tree: exp.Expression) -> str:
    """Produce a surface-invariant SQL string for the fine sensitivity view."""

    value = tree.copy()
    cte_mapping: dict[str, str] = {}
    for index, cte in enumerate(value.find_all(exp.CTE)):
        old = _name(cte.alias_or_name)
        new = f"cte{index}"
        if old:
            cte_mapping[old] = new
            cte.set("alias", exp.TableAlias(this=exp.to_identifier(new)))
    for table in value.find_all(exp.Table):
        old = _name(table.name)
        if old in cte_mapping:
            table.set("this", exp.to_identifier(cte_mapping[old]))

    for alias in list(value.find_all(exp.Alias)):
        alias.replace(alias.this.copy())
    for table in value.find_all(exp.Table):
        table.set("alias", None)
    for subquery in value.find_all(exp.Subquery):
        subquery.set("alias", None)
    for column in value.find_all(exp.Column):
        column.set("table", None)
        column.set("db", None)
        column.set("catalog", None)
    for literal in list(value.find_all(exp.Literal)):
        replacement = (
            exp.Literal.string("__str__")
            if literal.is_string
            else exp.Literal.number("0")
        )
        literal.replace(replacement)
    for identifier in value.find_all(exp.Identifier):
        identifier.set("this", _name(identifier.this))
        identifier.set("quoted", False)

    for equal in list(value.find_all(exp.EQ)):
        left = equal.this
        right = equal.expression
        if right.sql(dialect="sqlite") < left.sql(dialect="sqlite"):
            equal.set("this", right.copy())
            equal.set("expression", left.copy())
    and_roots = [
        node for node in value.find_all(exp.And) if not isinstance(node.parent, exp.And)
    ]
    for root in and_roots:
        terms = sorted(
            (term.copy() for term in _flatten_and(root)),
            key=lambda term: term.sql(dialect="sqlite"),
        )
        combined = terms[0]
        for term in terms[1:]:
            combined = exp.and_(combined, term, copy=False)
        root.replace(combined)
    return value.sql(dialect="sqlite", pretty=False, normalize=True)


@dataclass(frozen=True)
class BirdModeExtraction:
    sql: str
    parse_ok: bool
    parse_error: str | None
    structural_signature: dict[str, Any] | None
    structural_mode_id: str
    ames_full_signature: dict[str, str]
    ames_full_mode_id: str
    ames_coarse_signature: dict[str, str]
    ames_coarse_mode_id: str
    normalized_ast_sql: str | None
    normalized_ast_mode_id: str


@functools.lru_cache(maxsize=65_536)
def extract_bird_modes(output: str) -> BirdModeExtraction:
    sql = extract_sql(output)
    card = realized_descriptor(sql)
    full = card.as_dict()
    coarse = {
        "query_form": card.query_form,
        "aggregation": card.aggregation,
        "set_composition": card.set_composition,
    }
    try:
        parsed = [item for item in sqlglot.parse(sql, read="sqlite") if item is not None]
        if len(parsed) != 1:
            raise ValueError(f"expected one SQL statement, parsed {len(parsed)}")
        tree = parsed[0]
        signature = structural_signature(tree)
        ast_sql = normalized_ast(tree)
        return BirdModeExtraction(
            sql=sql,
            parse_ok=True,
            parse_error=None,
            structural_signature=signature,
            structural_mode_id=signature_id(signature),
            ames_full_signature=full,
            ames_full_mode_id=signature_id(full),
            ames_coarse_signature=coarse,
            ames_coarse_mode_id=signature_id(coarse),
            normalized_ast_sql=ast_sql,
            normalized_ast_mode_id=signature_id(ast_sql),
        )
    except (ValueError, sqlglot.errors.SqlglotError, RecursionError) as exc:
        return BirdModeExtraction(
            sql=sql,
            parse_ok=False,
            parse_error=f"{type(exc).__name__}: {exc}",
            structural_signature=None,
            structural_mode_id="unparsed",
            ames_full_signature=full,
            ames_full_mode_id=signature_id(full),
            ames_coarse_signature=coarse,
            ames_coarse_mode_id=signature_id(coarse),
            normalized_ast_sql=None,
            normalized_ast_mode_id="unparsed",
        )
