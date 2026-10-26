import importlib.util
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

spec = importlib.util.spec_from_file_location('experiment', Path(__file__).parents[1] / 'src/experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class MetricTests(unittest.TestCase):
    def test_zero_query_has_no_arbitrary_hit(self):
        self.assertEqual(experiment.retrieval_metrics(np.zeros(20), {0}), [0.0, 0.0])

    def test_multi_positive_relevance_and_cutoff(self):
        scores = np.arange(20, 0, -1, dtype=float)
        self.assertEqual(experiment.retrieval_metrics(scores, {5, 19}), [1 / 6, 0.0])
        self.assertEqual(experiment.retrieval_metrics(scores, {10}), [0.0, 0.0])

    def test_tie_break_is_deterministic(self):
        self.assertEqual(experiment.retrieval_metrics(np.ones(20), {1}), [0.5, 1.0])

    def test_bm25_matches_hand_computation(self):
        matrix = sparse.csr_matrix([[2, 0], [0, 1]], dtype=float)
        actual = experiment.bm25_documents(matrix).toarray()
        idf = np.log(2)
        expected = idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / 1.5))
        self.assertAlmostEqual(actual[0, 0], expected)
        self.assertEqual(actual[0, 1], 0)

    def test_selection_regret_not_rank_distance(self):
        estimate = np.array([3, 2, 1, 0, -1, -2])
        truth = np.array([0.2, 0.5, 0.4, 0.3, 0.1, 0])
        self.assertAlmostEqual(experiment.compare(estimate, truth)['regret'], 0.3)

    def test_effective_sample_size(self):
        self.assertEqual(experiment.ess(np.ones(10)), 10)
        self.assertLess(experiment.ess(np.array([100, 1, 1])), 1.1)

    def test_pii_and_unicode(self):
        self.assertTrue(experiment.PII.search('联系13800138000'))
        self.assertTrue(experiment.PII.search('mail@example.org'))
        self.assertEqual(experiment.normalize('ＡＢＣ，保险？'), 'abc保险')


if __name__ == '__main__':
    unittest.main()
