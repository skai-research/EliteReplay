"""Coarse, versioned mechanisms for MAP-Elites cells, not semantic equivalence.

No identifiers, constants, whitespace, or complete AST hashes enter a subtype.
The external primary strategy supplies the algorithm family; these features
separate implementations within it. CTE wrappers and DISTINCT alone do not
create another SQL subtype. Python features are conservative static proxies.
"""
from __future__ import annotations

import ast

VERSION = "coarse_v1"


def structural_subtype(domain: str, text: str) -> str:
    if domain == "optmath":
        # Python syntax is not evidence of another mathematical formulation.
        ast.parse(text)
        return "optmath_formulation_v1:no_syntax_subdivision"
    if domain == "bird":
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(text, read="sqlite")
        joins = len(list(tree.find_all(exp.Join)))
        subqueries = set()
        for node in tree.find_all(exp.Select):
            parent = node.parent
            while parent is not None and not isinstance(parent, exp.Select):
                if isinstance(parent, exp.Exists):
                    subqueries.add("exists")
                elif isinstance(parent, exp.In):
                    subqueries.add("membership")
                elif isinstance(parent, exp.Subquery):
                    if isinstance(parent.parent, (exp.From, exp.Join)):
                        subqueries.add("derived")
                    elif not isinstance(parent.parent, (exp.In, exp.Exists)):
                        subqueries.add("scalar")
                parent = parent.parent
        aggregation = "grouped" if tree.find(exp.Group) else "global" if tree.find(exp.AggFunc) else "none"
        sets = sorted({type(x).__name__.lower() for x in tree.walk()
                       if isinstance(x, (exp.Union, exp.Intersect, exp.Except))})
        features = {"join": "0" if not joins else "1" if joins == 1 else "many",
                    "subquery": "+".join(sorted(subqueries)) or "none", "aggregation": aggregation,
                    "window": str(bool(tree.find(exp.Window))).lower(),
                    "set": "+".join(sets) or "none",
                    "topk": str(bool(tree.find(exp.Order) and tree.find(exp.Limit))).lower()}
    elif domain == "taco":
        tree = ast.parse(text)
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    aliases[name.asname or name.name] = name.name
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    aliases[name.asname or name.name] = f"{node.module}.{name.name}"
        def name(node):
            if isinstance(node, ast.Name):
                return aliases.get(node.id, node.id)
            if isinstance(node, ast.Attribute):
                return name(node.value) + "." + node.attr
            return ""
        recursive = False
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                recursive |= any(isinstance(n, ast.Call) and
                                 (isinstance(n.func, ast.Name) and n.func.id == fn.name or
                                  isinstance(n.func, ast.Attribute) and n.func.attr == fn.name)
                                 for n in ast.walk(fn))
        calls = {name(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        decorators = {name(d.func if isinstance(d, ast.Call) else d)
                      for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      for d in n.decorator_list}
        loops = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        def depth(node, current=0):
            here = current + int(isinstance(node, loops))
            return max([here] + [depth(child, here) for child in ast.iter_child_nodes(node)])
        mechanisms = set()
        for call in calls:
            if call.startswith("heapq."): mechanisms.add("heap")
            if call.startswith("bisect."): mechanisms.add("bisect")
            if call == "collections.deque": mechanisms.add("deque")
            if call == "sorted" or call.endswith(".sort"): mechanisms.add("sort")
        if decorators & {"functools.cache", "functools.lru_cache"}: mechanisms.add("memoized")
        if any(isinstance(n, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)) for n in ast.walk(tree)):
            mechanisms.add("bitwise")
        features = {"recursion": str(recursive).lower(), "loop_depth": str(min(depth(tree), 2)),
                    "mechanism": "+".join(sorted(mechanisms)) or "plain"}
    else:
        raise ValueError(f"Unsupported subtype domain: {domain}")
    return VERSION + ":" + ";".join(f"{key}={value}" for key, value in sorted(features.items()))


def cell_id(primary: str | None, subtype: str | None = None) -> str | None:
    return None if primary is None else primary if subtype is None else f"{primary}::{subtype}"
