#!/usr/bin/env python3
"""Create small synthetic stage-7-compatible graph CSVs for algorithm checks.

No Telegram content or real diffusion measurements are used.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def make_synthetic_graph(output: Path, scenario: str = 'chain', n: int = 6) -> None:
    if n < 1:
        raise ValueError('Number of nodes must be at least one')
    if scenario not in ('chain', 'disconnected', 'empty'):
        raise ValueError('Unknown scenario')
    output.mkdir(parents=True, exist_ok=True)
    names = [f'SyntheticChannel{i+1:02d}' for i in range(n)]
    links = []
    if scenario == 'chain':
        links = [(i, i + 1) for i in range(n - 1)]
    elif scenario == 'disconnected':
        links = [(i, i + 1) for i in range(0, n-1, 2)]
    deg = [0] * n
    for a, b in links:
        deg[a] += 1
        deg[b] += 1

    with (output/'stage7_channel_nodes.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'channel', 'eligible_post_count', 'proxy_degree',
            'proxy_weighted_degree_matched_posts'])
        writer.writeheader()
        for i, name in enumerate(names):
            writer.writerow(dict(channel=name, eligible_post_count=20,
                                 proxy_degree=deg[i], proxy_weighted_degree_matched_posts=deg[i]*5))

    with (output/'stage7_proxy_edges.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'channel_a', 'channel_b', 'matched_post_pairs', 'shared_days',
            'mean_tfidf_similarity', 'normalized_overlap_rate',
            'first_matched_day_utc', 'last_matched_day_utc'])
        writer.writeheader()
        for a, b in links:
            writer.writerow(dict(channel_a=names[a], channel_b=names[b],
                                 matched_post_pairs=5, shared_days=2,
                                 mean_tfidf_similarity=0.88, normalized_overlap_rate=0.25,
                                 first_matched_day_utc='2026-09-01',
                                 last_matched_day_utc='2026-09-02'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('smoke_results/synthetic_graph'))
    parser.add_argument('--scenario', choices=('chain', 'disconnected', 'empty'), default='chain')
    parser.add_argument('--nodes', type=int, default=6)
    args = parser.parse_args()
    make_synthetic_graph(args.output, scenario=args.scenario, n=args.nodes)
    print('Synthetic graph CSV files written to', args.output.resolve())


if __name__ == '__main__':
    main()
