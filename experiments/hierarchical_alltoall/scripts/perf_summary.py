#!/usr/bin/env python3
"""Summarize one AllToAll sweep from the rank logs on all participating nodes."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
import sys


def fields(line):
    return dict(token.split('=', 1) for token in line.split()[1:])


def rank_result(path):
    config, samples, passed = None, {}, False
    with path.open() as log:
        for line in log:
            if line.startswith('PERF_CONFIG '):
                if config is not None:
                    raise ValueError(f'{path}: duplicate configuration')
                config = fields(line)
            elif line.startswith('PERF_SAMPLE '):
                sample = fields(line)
                if config is None or sample['rank'] != config['rank']:
                    raise ValueError(f'{path}: sample rank mismatch')
                step = int(sample['sample'])
                kernel, complete = float(sample['kernel_us']), float(sample['complete_us'])
                if (step in samples or not math.isfinite(kernel) or not math.isfinite(complete)
                        or not 0 < kernel <= complete):
                    raise ValueError(f'{path}: invalid sample {step}')
                samples[step] = (kernel, complete)
            elif config is not None and line.startswith(f"rank={config['rank']} PASS data+scale "):
                passed = True
    if config is None or not passed or set(samples) != set(range(int(config['iterations']))):
        raise ValueError(f'{path}: missing configuration, timed samples, or PASS')
    rank = int(config.pop('rank'))
    build = json.loads((path.parent / 'build_config.json').read_text())
    for key in ('rdma_backend', 'pi_on_chip', 'group_size', 'comm_block_m', 'catlass_commit'):
        config[key] = build[key]
    return rank, config, samples


def summarize(paths):
    ranks, common = {}, None
    for path in paths:
        rank, config, samples = rank_result(path)
        if rank in ranks or (common is not None and config != common):
            raise ValueError(f'{path}: duplicate rank or inconsistent run parameters')
        ranks[rank] = samples
        common = config
    world = int(common['world'])
    if set(ranks) != set(range(world)):
        raise ValueError(f"data_peer_bytes={common['data_peer_bytes']}: missing ranks for world={world}")
    steps = range(int(common['iterations']))
    kernels = [max(samples[i][0] for samples in ranks.values()) for i in steps]
    completed = [max(samples[i][1] for samples in ranks.values()) for i in steps]
    remote = (world - 1) * (int(common['data_peer_bytes']) + int(common['scale_peer_bytes']))
    return dict(
        **common,
        remote_bytes_per_rank=remote,
        max_rank_kernel_mean_us=f'{mean(kernels):.3f}',
        max_rank_complete_mean_us=f'{mean(completed):.3f}',
        max_rank_complete_p50_us=f'{median(completed):.3f}',
        max_rank_complete_p95_us=f'{sorted(completed)[math.ceil(len(completed) * 0.95) - 1]:.3f}',
        effective_GBps_per_rank=f'{remote / mean(completed) / 1000:.3f}',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path, help='sweep log root from each node, or a shared root')
    args = parser.parse_args()
    cases = {}
    for root in args.roots:
        for path in root.glob('data-*-node*/rank*.log'):
            case = int(path.parent.name.split('-')[1])
            cases.setdefault(case, []).append(path)
    expected = {4096, 65536, 1048576, 2097152, 4194304}
    if set(cases) != expected:
        parser.error(f'missing data sizes: {sorted(expected - cases.keys())}; '
                     f'unexpected data sizes: {sorted(cases.keys() - expected)}')
    try:
        results = [summarize(cases[case]) for case in sorted(cases)]
        for case, result in zip(sorted(cases), results):
            if case != int(result['data_peer_bytes']):
                raise ValueError(f'case name does not match data_peer_bytes: {case}')
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'[ERROR] {error}\n')
    writer = csv.DictWriter(sys.stdout, fieldnames=results[0].keys(), lineterminator='\n')
    writer.writeheader()
    writer.writerows(results)


if __name__ == '__main__':
    main()
