import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))

# These tests require the isolated v2 environment (transformers >=4.37).
from generate_v2 import parse_question
from validate_v2 import order_metrics, rrf_scores, validate_logs


class ValidationTests(unittest.TestCase):
    def test_format_rejects_multiple_questions_and_cap(self):
        self.assertFalse(parse_question('退保需要哪些材料？')[1])
        self.assertTrue(parse_question('能退吗？如何退？')[1])
        self.assertIn('token_limit', parse_question('能办理退保吗？', capped=True)[1])
        self.assertTrue(parse_question('请联系13800138000？')[1])

    def test_rrf_zero_lexical_does_not_add_arbitrary_votes(self):
        dense = np.array([-0.3, -0.1, -0.2])
        result = rrf_scores(np.zeros(3), dense)
        np.testing.assert_allclose(result, [1 / 63, 1 / 61, 1 / 62])

    def test_rrf_uses_rank_not_score_scale(self):
        lexical = np.array([2.0, 1.0, 0.0])
        dense = np.array([0.1, 0.9, 0.3])
        np.testing.assert_allclose(rrf_scores(lexical, dense), rrf_scores(lexical * 100, dense * 20))

    def test_dense_order_accepts_negative_similarities(self):
        # Negative cosine scores are valid; the lexical positive-score gate cannot be reused.
        self.assertEqual(order_metrics(np.array([1, 0, 2]), {1}), [1.0, 1.0])

    def test_generation_denominator_preserved(self):
        records = [{'mode': m, 'doc_id': i, 'format_valid': not (m == 'zero_shot' and i == 2)}
                   for m in ['zero_shot', 'few_shot_3'] for i in [1, 2]]
        _, common = validate_logs(records, [1, 2])
        self.assertEqual(common, [1])
        with self.assertRaises(ValueError):
            validate_logs(records[:-1], [1, 2])
        with self.assertRaises(ValueError):
            validate_logs(records + records[:1], [1, 2])


if __name__ == '__main__':
    unittest.main()
