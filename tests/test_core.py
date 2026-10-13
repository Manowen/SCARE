import unittest
import numpy as np
from scare.prompts import prompt, render
from scare.trajectory import predict_delta, project_delta, endpoint_delta
from scare.metrics import summarize, repetition
from scare.select import select_candidate
from scare.grading import final_box, grade


class CoreTests(unittest.TestCase):
    def test_linear_trajectory(self):
        steps = np.array([10, 25, 50])
        direction = np.array([0.6, -0.8], dtype=np.float32)
        matrix = np.outer(2 * steps + 3, direction).astype(np.float32)
        prediction, stats = predict_delta(matrix, steps, 250)
        np.testing.assert_allclose(prediction, 503 * direction, rtol=1e-5)
        np.testing.assert_allclose(project_delta(matrix), matrix[-1], rtol=1e-5)
        np.testing.assert_allclose(endpoint_delta(matrix, steps, 250), matrix[-1] * 5, rtol=1e-5)
        self.assertAlmostEqual(stats['energy'], 1, places=5)
        self.assertAlmostEqual(stats['r2'], 1, places=5)

    def test_zero_trajectory(self):
        predicted, stats = predict_delta(np.zeros((3, 4), dtype=np.float32), [10, 25, 50], 250)
        np.testing.assert_array_equal(predicted, np.zeros(4))
        self.assertIsNone(stats['r2'])

    def test_hint_schedule(self):
        row = {'question': 'Compute a sum.', 'hint_english': 'First combine terms. Then simplify the sum.', 'hint_prefixes': ['First combine terms.', 'First combine terms. Then simplify the sum.']}
        self.assertNotIn('Partial solution', prompt(row))
        self.assertEqual(render(row, 'none', 1)[1], 0)
        self.assertEqual(render(row, 'fixed', 250)[1], 7)
        self.assertEqual(render(row, 'decay', 1)[1], 7)
        self.assertEqual(render(row, 'decay', 101)[1], 3)
        self.assertEqual(render(row, 'decay', 200)[1], 0)
        self.assertEqual(render(row, 'decay', 250)[1], 0)

    def test_grading(self):
        self.assertEqual(final_box(r'First \boxed{0}, finally \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertTrue(grade(r'\boxed{\frac{1}{2}}', '0.5')[0])
        self.assertFalse(grade('No final expression', '1')[0])

    def test_selection(self):
        self.assertEqual(select_candidate([{'target': 250, 'accuracy': .5}, {'target': 100, 'accuracy': .5}])['target'], 100)
        with self.assertRaises(ValueError):
            select_candidate([{'target': 100, 'accuracy': float('nan')}])

    def test_metrics(self):
        rows = [{'language': 'en', 'correct': level == 'top', 'truncated': False, 'new_tokens': 10, 'prediction': 'done', 'difficulty_bucket': level} for level in ['low', 'medium', 'high', 'top']]
        values = summarize(rows)['en']
        self.assertEqual(values['accuracy'], .25)
        self.assertAlmostEqual(values['difficulty_weighted_accuracy'], 8 / 15)
        self.assertNotIn('difficulty_weighted_accuracy', summarize(rows[:2])['en'])
        self.assertEqual(repetition('short output'), 0)
        self.assertGreater(repetition('word ' * 30), .9)


if __name__ == '__main__':
    unittest.main()
