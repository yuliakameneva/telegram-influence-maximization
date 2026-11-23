"""Verify graph export validation on synthetic graph data."""
import tempfile
import unittest
from pathlib import Path

from telegram_stage8_simulate_adaptive import read_graph
from scripts.generate_test_data import make_synthetic_graph


class GraphTests(unittest.TestCase):
    def test_read_valid_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            make_synthetic_graph(out, scenario='chain', n=4)
            nodes, edges = read_graph(out/'stage7_proxy_edges.csv', out/'stage7_channel_nodes.csv')
            self.assertEqual(nodes, 4)
            self.assertEqual(len(edges), 3)

    def test_disconnected_graph_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            make_synthetic_graph(out, scenario='disconnected', n=5)
            nodes, edges = read_graph(out/'stage7_proxy_edges.csv', out/'stage7_channel_nodes.csv')
            self.assertEqual(nodes, 5)
            self.assertEqual(len(edges), 2)

    def test_empty_edges_rejected_by_stage8_cli_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            make_synthetic_graph(out, scenario='empty', n=4)
            with self.assertRaises(SystemExit):
                read_graph(out/'stage7_proxy_edges.csv', out/'stage7_channel_nodes.csv')

    def test_n_less_than_three_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            make_synthetic_graph(out, scenario='empty', n=1)
            with self.assertRaises(SystemExit):
                read_graph(out/'stage7_proxy_edges.csv', out/'stage7_channel_nodes.csv')


if __name__ == '__main__':
    unittest.main()
