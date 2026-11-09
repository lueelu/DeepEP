"""Random destination order with exact global per-rank quotas.

Matches the supplied algorithm at hot_multiplier_x2=2 and EP>=TopK.
For EP<TopK, allow ceil(TopK/EP) assignments per destination per token;
experts inside that destination remain distinct. No NPU/distributed imports.
"""

from functools import lru_cache
import random
import threading


ROUTING_SEED = 20261014
_cache_lock = threading.Lock()


@lru_cache(maxsize=1)
def _global_routes(bs, ep, experts, topk, seed):
    import numpy as np

    local_experts = experts // ep
    tokens = ep * bs
    # total/ep = BS*TopK exactly; this mode never creates a hot rank.
    remaining = [bs * topk] * ep
    occurrences = [0] * ep
    cap = (topk + ep - 1) // ep
    rng = random.Random(seed + bs * 1000 + 2)
    routes = np.empty((tokens, topk), dtype=np.int32)
    for token in range(tokens):
        left = tokens - token
        # Anything that cannot fit in subsequent tokens must go in this one.
        counts = [max(0, n - (left - 1) * cap) for n in remaining]
        chosen = [rank for rank, count in enumerate(counts) for _ in range(count)]
        if len(chosen) > topk or any(n > cap for n in counts):
            raise RuntimeError("Infeasible random-balanced rank quotas")
        while len(chosen) < topk:
            candidates = [r for r in range(ep) if counts[r] < cap and remaining[r] > counts[r]]
            if not candidates:
                raise RuntimeError("Random-balanced routing exhausted its candidates")
            rank = rng.choices(candidates, weights=[remaining[r] - counts[r] for r in candidates])[0]
            chosen.append(rank)
            counts[rank] += 1
        rng.shuffle(chosen)
        for slot, rank in enumerate(chosen):
            # Advance on EACH occurrence, including repeated destinations in EP4.
            routes[token, slot] = rank * local_experts + occurrences[rank] % local_experts
            remaining[rank] -= 1
            occurrences[rank] += 1
    if any(remaining):
        raise RuntimeError("Random-balanced routing left unassigned quotas")
    routes.setflags(write=False)
    return routes


def random_balanced_indices(bs, ep, experts, topk, rank, *, seed=ROUTING_SEED):
    """Return this source rank's deterministic slice of the global token stream.

    The routing seed intentionally matches the supplied script, independently
    of the model/input seed. Cache one global table per process, not per source
    rank; CPU Golden threads share it. Callers receive a writable private copy.
    """
    if any(type(n) is not int for n in (bs, ep, experts, topk, rank, seed)):
        raise ValueError("Routing dimensions and seed must be integers")
    if bs <= 0 or ep <= 0 or experts <= 0 or experts % ep or not 0 < topk <= experts or not 0 <= rank < ep:
        raise ValueError("Invalid random-balanced BS/EP/experts/TopK/rank")
    with _cache_lock:
        routes = _global_routes(bs, ep, experts, topk, seed)
        return routes[rank * bs : (rank + 1) * bs].copy()
