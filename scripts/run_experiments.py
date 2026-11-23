#!/usr/bin/env python3
"""Run reproducible stage-8 smoke test or final graph-based experiments.

This script does not invent experimental results and does not train a new NLP model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require(path: Path, description: str) -> None:
    if not path.is_file():
        raise SystemExit(f'Missing {description}: {path}. See README.md')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('smoke', 'real-graph', 'full-experiments'), default='smoke')
    p.add_argument('--repetitions', type=int, default=120,
                   help='Number of runs per scenario in study modes (120 used in report)')
    args = p.parse_args()
    if args.repetitions < 1:
        p.error('--repetitions must be positive')
    if args.mode == 'smoke':
        out = ROOT/'smoke_results'
        run('scripts/generate_test_data.py', '--output', str(out/'synthetic_graph'),
            '--scenario', 'chain', '--nodes', '6')
        run('telegram_stage8_simulate_adaptive.py',
            '--edges', str(out/'synthetic_graph/stage7_proxy_edges.csv'),
            '--nodes', str(out/'synthetic_graph/stage7_channel_nodes.csv'),
            '--output', str(out/'stage8'), '--repetitions', '3',
            '--planning-worlds', '4', '--adaptive-worlds', '2', '--budget', '2')
        return

    nodes = ROOT/'results_september_stage7/stage7_channel_nodes.csv'
    edges = ROOT/'results_september_stage7/stage7_proxy_edges.csv'
    require(nodes, 'stage 7 channel nodes')
    require(edges, 'stage 7 proxy edges')
    run('telegram_stage8_simulate_adaptive.py',
        '--repetitions', str(args.repetitions), '--planning-worlds', '180',
        '--adaptive-worlds', '24', '--budget', '4')
    if args.mode == 'real-graph':
        return

    signals = ROOT/'results_september_stage5/stage5_channel_days_PILOT.csv'
    prepared = ROOT/'results_september_stage2/posts_prepared.jsonl'
    require(signals, 'pilot time/text signal aggregates for stage 9')
    require(prepared, 'prepared September posts for stage 10')
    run('telegram_stage9_temporal_text_ablation.py',
        '--repetitions', str(args.repetitions), '--budget', '4',
        '--planning-worlds', '200')
    run('telegram_stage10_forward_validation_corrected.py',
        '--repetitions', str(args.repetitions), '--planning-worlds', '180',
        '--budget', '4', '--seed', '2026')


if __name__ == '__main__':
    main()
