"""Typical, boundary, and degenerate cases for adaptive influence simulation."""
import random
import unittest

from telegram_stage8_simulate_adaptive import (
    adaptive_degree_pick, adaptive_evaluate, live_world, reach,
)


class AdaptiveDegreeTests(unittest.TestCase):
    def test_typical_weighted_degree(self):
        edges = [(0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.1)]
        self.assertEqual(adaptive_degree_pick(4, edges, 0), 1)

    def test_equal_scores_choose_smallest_index(self):
        self.assertEqual(adaptive_degree_pick(3, [], 0), 0)

    def test_active_node_is_excluded(self):
        edges = [(0, 1, 0.9), (1, 2, 0.8)]
        self.assertEqual(adaptive_degree_pick(3, edges, 1 << 1), 0)

    def test_all_active_returns_none(self):
        self.assertIsNone(adaptive_degree_pick(3, [], 0b111))

    def test_zero_node_graph_returns_none(self):
        self.assertIsNone(adaptive_degree_pick(0, [], 0))


class CascadeTests(unittest.TestCase):
    def test_deterministic_full_cascade_on_chain(self):
        world = live_world(3, [(0, 1, 1.0), (1, 2, 1.0)], random.Random(7))
        self.assertEqual(reach(0, world), 0b111)

    def test_no_edges_means_one_seed_only(self):
        world = live_world(4, [], random.Random(7))
        self.assertEqual(reach(2, world), 0b0100)

    def test_zero_probability_has_no_transmission(self):
        world = live_world(2, [(0, 1, 0.0)], random.Random(7))
        self.assertEqual(reach(0, world), 0b01)

    def test_blocked_seed_cannot_reactivate(self):
        self.assertEqual(reach(0, [0b10, 0b01], blocked=0b01), 0)

    def test_monotonicity_when_budget_exceeds_nodes(self):
        world = live_world(3, [(0, 1, 0.0), (1, 2, 0.0)], random.Random(7))
        values = adaptive_evaluate(world, 3, [(0, 1, 0.0), (1, 2, 0.0)],
                                   kmax=5, inner_worlds=2, rng=random.Random(7),
                                   kind='adaptive_degree')
        self.assertEqual(values, [1, 2, 3, 3, 3])
        self.assertTrue(all(x <= y for x, y in zip(values, values[1:])))


if __name__ == '__main__':
    unittest.main()
