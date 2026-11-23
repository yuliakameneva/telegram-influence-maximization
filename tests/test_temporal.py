"""Check time-forward windows and strict historical feature selection."""
import unittest
from collections import Counter
from telegram_stage10_forward_validation_corrected import get_dates, window_edges


class TimeForwardTests(unittest.TestCase):
    def test_september_first_snapshot(self):
        hist, future = get_dates('2026-09-08', back=7, ahead=3)
        self.assertEqual(hist, [f'2026-09-{day:02d}' for day in range(1, 8)])
        self.assertEqual(future, ['2026-09-08', '2026-09-09', '2026-09-10'])
        self.assertTrue(max(hist) < min(future))

    def test_time_windows_across_month_boundary(self):
        hist, future = get_dates('2026-10-01', back=7, ahead=3)
        self.assertEqual(hist[-1], '2026-09-30')
        self.assertEqual(future[0], '2026-10-01')

    def test_window_edges_does_not_use_future(self):
        posts = {
            '2026-09-01': [{'channel': 'A'}, {'channel': 'B'}],
            '2026-09-08': [{'channel': 'A'}, {'channel': 'C'}],
        }
        matches = {
            '2026-09-01': [(('A', 'B'), 0.9, None, None)],
            '2026-09-08': [(('A', 'C'), 0.9, None, None)],
        }
        counts, edges = window_edges(matches, posts, ['2026-09-01'])
        self.assertEqual(counts, Counter({'A': 1, 'B': 1}))
        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0]['channel_a'], edges[0]['channel_b']), ('A', 'B'))

    def test_empty_history_is_degenerate_case(self):
        self.assertEqual(window_edges({}, {}, [])[1], [])


if __name__ == '__main__':
    unittest.main()
