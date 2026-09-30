# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Deterministic random routing; NumPy uint64 preserves C++ wraparound."""

import numpy as np


def private_random_routes(world, tokens, topk, experts, seed=1234):
    if not (world > 0 and tokens > 0 and 0 < topk <= experts and 0 <= seed < 2**64):
        raise ValueError("Invalid routing shape or seed")

    def mix(x):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> 31)

    rank = np.repeat(np.arange(world, dtype=np.uint64), tokens)
    token = np.tile(np.arange(tokens, dtype=np.uint64), world)
    state = mix(np.uint64(seed) ^ (rank << 32) ^ token)
    ids = np.full((world * tokens, topk), -1, dtype=np.int64)
    reject_below = np.uint64((1 << 64) % experts)
    for k in range(topk):
        pending = np.arange(world * tokens)
        while pending.size:
            state[pending] += np.uint64(0x9E3779B97F4A7C15)
            value = mix(state[pending])
            expert = (value % experts).astype(np.int64)
            valid = (value >= reject_below) & ~np.any(ids[pending, :k] == expert[:, None], axis=1)
            ids[pending[valid], k] = expert[valid]
            pending = pending[~valid]
    return ids.reshape(world, tokens, topk)
