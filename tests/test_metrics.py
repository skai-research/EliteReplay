import math
import unittest

from metrics import evaluate
from evorl.analysis.sql_strategy import similarity_sql, strategy_key

A = "SELECT a FROM t"
B = "SELECT a FROM t JOIN u ON t.id = u.id"


class MetricTests(unittest.TestCase):
    def test_paper_estimators_and_retention(self):
        rows = []
        for model, samples in {
            "base": [(A, True), (A, True), (B, True), (A, False)],
            "elite": [(A, True), (A, True), (A, True), (A, False)],
        }.items():
            for sample_id, (sql, correct) in enumerate(samples):
                rows.append(dict(model=model, question_id="q", sample_id=sample_id,
                                 correct=correct, sql=sql))
        result = evaluate(rows, [1, 2, 4], "base")
        base, elite = result["models"]["base"], result["models"]["elite"]
        self.assertNotEqual(strategy_key(A), strategy_key(B))
        self.assertAlmostEqual(base["u@1"], 0.75)
        self.assertAlmostEqual(base["u@2"], 4 / 3)
        self.assertAlmostEqual(base["u@4"], 2.0)
        self.assertAlmostEqual(base["effective_diversity"], (1 - similarity_sql(A, B)) / 3)
        self.assertAlmostEqual(elite["retention"], 0.5)
        self.assertAlmostEqual(elite["effective_diversity"], 0.0)

    def test_duplicate_sample_rejected(self):
        row = dict(model="base", question_id="q", sample_id=0, correct=True, sql=A)
        with self.assertRaisesRegex(ValueError, "Duplicate sample"):
            evaluate([row, row], [1], "base")

    def test_all_metrics_share_alias_invariant_ast_representation(self):
        first = "SELECT x.a AS score FROM t AS x ORDER BY score"
        renamed = " select q.a AS result FROM t AS q ORDER BY result "
        different = "SELECT q.b AS result FROM t AS q ORDER BY result"
        self.assertEqual(strategy_key(first), strategy_key(renamed))
        self.assertEqual(similarity_sql(first, renamed), 1.0)
        self.assertNotEqual(strategy_key(first), strategy_key(different))
        self.assertLess(similarity_sql(first, different), 1.0)
        rows = [
            dict(model=model, question_id="q", sample_id=i, correct=True, sql=sql)
            for model, samples in {"base": [first, different], "elite": [renamed, renamed]}.items()
            for i, sql in enumerate(samples)
        ]
        result = evaluate(rows, [2], "base")
        self.assertEqual(result["models"]["elite"]["u@2"], 1.0)
        self.assertEqual(result["models"]["elite"]["retention"], 0.5)
        self.assertEqual(result["models"]["elite"]["effective_diversity"], 0.0)


if __name__ == "__main__":
    unittest.main()
