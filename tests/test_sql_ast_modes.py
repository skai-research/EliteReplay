"""Full-AST U@K preserves distinctions and has a deliberately small normalization contract."""
import itertools
from collections import Counter
from pathlib import Path
import sys

import pytest

from elitereplay.evaluation.descriptors.sql_ast_modes import full_ast_key


@pytest.mark.parametrize('other', [
    'select T.ID from T where T.X = 1;',
    'SELECT t.id /* explanation */ FROM t WHERE t.x=1',
    'SELECT t.id\nFROM t\nWHERE t.x = 1',
])
def test_cosmetic_invariance(other):
    assert full_ast_key('SELECT t.id FROM t WHERE t.x = 1') == full_ast_key(other)


@pytest.mark.parametrize('a,b', [
    ('SELECT id FROM t WHERE x=1', 'SELECT id FROM t WHERE x=2'),
    ("SELECT id FROM t WHERE x='A'", "SELECT id FROM t WHERE x='a'"),
    ('SELECT a.id FROM a JOIN b ON a.id=b.id', 'SELECT b.id FROM a JOIN b ON a.id=b.id'),
    ('SELECT a.id FROM a JOIN b ON a.id=b.id', 'SELECT a.id FROM a JOIN c ON a.id=c.id'),
    ('SELECT SUM(x) OVER (PARTITION BY a) FROM t', 'SELECT SUM(x) OVER (PARTITION BY b) FROM t'),
    ('SELECT id FROM t WHERE id IN (SELECT id FROM u)', 'SELECT id FROM t WHERE id NOT IN (SELECT id FROM u)'),
    ('SELECT x FROM t ORDER BY x LIMIT 1', 'SELECT x FROM t ORDER BY x LIMIT 2'),
    ('SELECT x FROM t ORDER BY x', 'SELECT x FROM t ORDER BY x DESC'),
    ('SELECT x FROM t UNION SELECT x FROM u', 'SELECT x FROM t UNION ALL SELECT x FROM u'),
    ('SELECT SUM(a) FROM t', 'SELECT SUM(b) FROM t'),
    ('SELECT COUNT(*) FROM t GROUP BY a', 'SELECT COUNT(*) FROM t GROUP BY b'),
    ('SELECT id FROM t', 'SELECT DISTINCT id FROM t'),
    ('SELECT x.id FROM t x', 'SELECT y.id FROM t y'),
    ('SELECT "id" FROM t', 'SELECT id FROM t'),
    ('SELECT "A"', 'SELECT "a"'),  # SQLite can interpret these as string literals.
    ('SELECT CAST(x AS INTEGER) FROM t', 'SELECT CAST(x AS TEXT) FROM t'),
    ('SELECT a.id FROM t a WHERE EXISTS (SELECT 1 FROM u b WHERE b.id=a.id)',
     'SELECT a.id FROM t a WHERE EXISTS (SELECT 1 FROM u b WHERE b.id=b.id)'),
])
def test_retained_distinctions(a, b):
    assert full_ast_key(a) != full_ast_key(b)


@pytest.mark.parametrize('sql', ['', 'SELECT 1; SELECT 2', 'DELETE FROM t', 'SELECT ('])
def test_rejects_unusable_inputs(sql):
    with pytest.raises(Exception):
        full_ast_key(sql)


def test_quoted_identifier_is_not_reparsed_as_sql():
    a = full_ast_key('SELECT "Free Meal Count (K-12)" FROM frpm')
    assert a != full_ast_key('SELECT "Free Meal Count (K-12) extra" FROM frpm')
