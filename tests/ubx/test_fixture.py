# Copyright (c) 2026, Lu Lu
"""Reconstruct Combine contributions from CPU wire metadata, independent of an NPU."""

from array import array
import random
import struct

import pytest

from fixture import decode, encode, make_fixture, value


def unpack(kind, data):
    result = array(kind)
    result.frombytes(data)
    return result


@pytest.mark.parametrize("world,width", [(8, 4), (32, 16), (64, 16)])
@pytest.mark.parametrize("bs,capacity", [(1, 256), (4, 256), (32, 32)])
def test_wire_routes_reconstruct_complete_output(world, width, bs, capacity):
    experts, topk, hidden = 384, 8, 3
    rng = random.Random(42)
    routes = array("i", (e for _ in range(world * bs) for e in rng.sample(range(experts), topk)))
    fixtures = [make_fixture(routes, world, rank, bs, hidden, topk, experts, width, capacity) for rank in range(world)]
    stage, returns = {}, {}
    for rank, f in enumerate(fixtures):
        dest, types = unpack("i", f["destination"]), unpack("i", f["types"])
        x = unpack("H", f["input"])
        counts = unpack("i", f["counts"])
        assert counts[-1] == len(types) and list(counts) == sorted(counts)
        for row, role in enumerate(types):
            data = [decode(v) for v in x[row * hidden : (row + 1) * hidden]]
            if role == 3:
                slot = dest[row * 3]
                assert (rank, slot) not in stage
                stage[rank, slot] = data
    for rank, f in enumerate(fixtures):
        dest, types, relay = (unpack("i" if k != "relay" else "I", f[k]) for k in ("destination", "types", "relay"))
        x = unpack("H", f["input"])
        for row, role in enumerate(types):
            if role == 3:
                continue
            source, token, k = dest[row * 3 : row * 3 + 3]
            data = [decode(v) for v in x[row * hidden : (row + 1) * hidden]]
            if role == 2:
                record = relay[row * topk * 2 : (row + 1) * topk * 2]

                def weight(i):
                    return struct.unpack("<f", struct.pack("<I", record[2 * i + 1]))[0]

                data = [v * weight(0) for v in data]
                for member in range(1, record[0] + 1):
                    expert, q = record[2 * member] & 65535, record[2 * member] >> 16
                    target, local = divmod(expert, experts // world)
                    slot = (((source + target) % world) * (experts // world) + local) * bs + q
                    assert target // width == rank // width
                    data = [a + b * weight(member) for a, b in zip(data, stage[target, slot])]
                data = [decode(encode(v)) for v in data]
            assert (source, token, k) not in returns
            returns[source, token, k] = data
    for rank, f in enumerate(fixtures):
        masks, weights = unpack("I", f["mask"]), unpack("f", f["weights"])
        result = []
        for token, mask in enumerate(masks):
            data = [0.0] * hidden
            for k in range(topk):
                assert bool(mask & (1 << k)) == ((rank, token, k) in returns)
                if mask & (1 << k):
                    data = [a + b * weights[token * topk + k] for a, b in zip(data, returns[rank, token, k])]
            result.extend(encode(v) for v in data)
        assert result == list(f["expected"])
        # Compare to the unfused expert sum as well: hierarchy may round BF16 partials.
        for token in range(bs):
            for h in range(hidden):
                ids = routes[(rank * bs + token) * topk : (rank * bs + token + 1) * topk]
                expected = sum(value(rank, token, e, h) / topk for e in ids)
                assert abs(decode(result[token * hidden + h]) - expected) < 0.02
