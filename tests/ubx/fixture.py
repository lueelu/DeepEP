# Copyright (c) 2026, Lu Lu
"""CPU construction of the RDMA server-dedup Combine wire metadata.

No device Dispatch or Notify is executed. Expert rows retain the Dispatch
order (local expert, source rank, occurrence), including full-mesh first hops.
"""

from array import array
from collections import Counter
import hashlib
import struct
import sys


def encode(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def decode(value):
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def value(source, token, expert, h):
    return ((source * 17 + token * 3 + expert * 7 + h % 31) % 127 - 63) / 64


def make_fixture(routes, world, rank, bs, hidden, topk, experts, ranks_per_host, capacity=256):
    if sys.byteorder != "little":
        raise ValueError("The UBX fixture requires little-endian storage")
    if not 0 <= rank < world or experts % world or ranks_per_host not in (4, 16):
        raise ValueError("Invalid fixture topology")
    if len(routes) != world * bs * topk or not 1 <= topk <= min(experts, 32) or not bs <= capacity <= 65535:
        raise ValueError("Invalid fixture shape")
    group = 2 if world == 2 else 4 if world <= 16 else 16
    if group > ranks_per_host:
        raise ValueError("Dedup groups must fit within one host")
    local_experts = experts // world
    dedup = world > 16 or capacity > 32
    deliveries = []
    masks = array("I", [0]) * bs
    weights = array("f", [1 / topk]) * (bs * topk)
    reference_groups = []
    for source in range(world):
        occurrences = Counter()
        for token in range(bs):
            start = (source * bs + token) * topk
            ids = list(routes[start : start + topk])
            if len(set(ids)) != topk or any(not 0 <= e < experts for e in ids):
                raise ValueError("Routes must contain distinct valid experts per token")
            q = [occurrences[e] for e in ids]
            for e in ids:
                occurrences[e] += 1
            grouped = {}
            for k, expert in enumerate(ids):
                target_group = expert // local_experts // group
                key = target_group if dedup and target_group != source // group else ("direct", k)
                grouped.setdefault(key, []).append(k)
            roles = [1] * topk
            members = {}
            token_groups = []
            for ks in grouped.values():
                first = ks[0]
                if len(ks) > 1 and group == ranks_per_host == 16:
                    mesh = {k: ((ids[k] // local_experts) & 15) // 4 for k in ks}
                    counts = Counter(mesh.values())
                    first = min(ks, key=lambda k: (-counts[mesh[k]], k))
                ordered = [first] + [k for k in ks if k != first]
                token_groups.append((first, [ids[k] for k in ordered]))
                if len(ks) > 1:
                    for k in ks:
                        roles[k] = 3
                    roles[first] = 2
                    members[first] = [(ids[k] | (q[k] << 16), 1 / topk) for k in ordered[1:]]
                if source == rank:
                    masks[token] |= 1 << first
                    if len(ks) > 1:
                        weights[token * topk + first] = 1
            if source == rank:
                reference_groups.append(sorted(token_groups))
            for k, expert in enumerate(ids):
                if expert // local_experts == rank:
                    slot = (((source + rank) % world) * local_experts + expert % local_experts) * bs + q[k]
                    destination = (slot, -1, -1) if roles[k] == 3 else (source, token, k)
                    deliveries.append(
                        (expert % local_experts, source, q[k], token, expert, roles[k], destination, members.get(k, []))
                    )
    deliveries.sort(key=lambda row: row[:3])
    x, destination, types, relay = array("H"), array("i"), array("i"), array("I")
    counts = [0] * experts
    for local, source, _, token, expert, role, dest, members in deliveries:
        counts[local * world + source] += 1
        x.extend(encode(value(source, token, expert, h)) for h in range(hidden))
        destination.extend(dest)
        types.append(role)
        record = [0] * (topk * 2)
        if role == 2:
            record[0], record[1] = len(members), struct.unpack("<I", struct.pack("<f", 1 / topk))[0]
            for i, (packed, weight) in enumerate(members, 1):
                record[2 * i] = packed
                record[2 * i + 1] = struct.unpack("<I", struct.pack("<f", weight))[0]
        relay.extend(record)
    for i in range(1, experts):
        counts[i] += counts[i - 1]
    expected = array("H")
    for token, groups in enumerate(reference_groups):
        for h in range(hidden):
            result = 0.0
            for _, ids in groups:
                partial = sum(decode(encode(value(rank, token, expert, h))) / topk for expert in ids)
                result += decode(encode(partial)) if len(ids) > 1 else partial
            expected.append(encode(result))
    return {
        "input": x.tobytes(),
        "destination": destination.tobytes(),
        "counts": array("i", counts).tobytes(),
        "weights": weights.tobytes(),
        "types": types.tobytes(),
        "relay": relay.tobytes(),
        "mask": masks.tobytes(),
        "expected": expected,
        "route_sha256": hashlib.sha256(array("i", routes).tobytes()).hexdigest(),
    }
