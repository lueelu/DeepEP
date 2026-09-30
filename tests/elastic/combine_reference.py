"""CPU-only Combine metadata reference for the elastic tests.

The full deterministic routing is regenerated locally, NEVER communicated. This
is a test planner, not a production distributed Notify implementation.
"""

from array import array
from dataclasses import dataclass

INT32_MAX = (1 << 31) - 1
REDUCE_ROW_MASK = (1 << 32) - 1
GATEWAY_CONTROL_UB_BYTES = 44 * 1024


def unpack_reduce_offset(packed):
    """Backward ABI: low uint32 row, high uint32 one-based chunk tail ID."""
    if not 0 <= packed <= (1 << 63) - 1:
        raise ValueError("packed reduce offset must be a nonnegative int64")
    return packed & REDUCE_ROW_MASK, packed >> 32


def reduce_row(packed):
    return unpack_reduce_offset(packed)[0]


def chunk_end_id(packed):
    return unpack_reduce_offset(packed)[1]


@dataclass(frozen=True)
class CombineCase:
    world: int
    tokens: int
    topk: int
    experts: int
    route: str = "balanced"
    seed: int = 1
    chunk_tokens: int = 256
    group_size: int = 64
    first_chunk_splits: int = 4

    @property
    def group_width(self):
        return min(self.world, self.group_size)

    @property
    def chunks_per_group(self):
        return self.logical_chunks + self.effective_first_chunk_splits - 1

    @property
    def logical_chunks(self):
        return (self.tokens + self.chunk_tokens - 1) // self.chunk_tokens

    @property
    def effective_first_chunk_splits(self):
        return min(self.first_chunk_splits, self.tokens, self.chunk_tokens)

    @property
    def first_chunk_tokens(self):
        """First subchunk length; zero means no split (retained for logging)."""
        first = min(self.tokens, self.chunk_tokens)
        parts = self.effective_first_chunk_splits
        return (first + parts - 1) // parts if parts > 1 else 0

    def token_chunk(self, token):
        first = min(self.tokens, self.chunk_tokens)
        parts = self.effective_first_chunk_splits
        return token * parts // first if token < first else token // self.chunk_tokens + parts - 1

    def chunk_token_end(self, chunk):
        parts = self.effective_first_chunk_splits
        if chunk < parts:
            return ((chunk + 1) * min(self.tokens, self.chunk_tokens) + parts - 1) // parts
        return min((chunk + 2 - parts) * self.chunk_tokens, self.tokens)

    @property
    def total_chunks(self):
        return self.world // self.group_width * self.chunks_per_group

    @property
    def gateway_control_bytes(self):
        ranges = 8 * self.world * self.chunks_per_group
        masks = 4 * self.total_chunks
        return ((ranges + 31) // 32 + (masks + 31) // 32) * 32

    def validate(self):
        if self.world not in (2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError("unsupported world")
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size not in (1, 2, 4, 8, 16, 32, 64, 128)
        ):
            raise ValueError("group_size must be a power of two in [1,128]")
        if not 0 < self.tokens or not 1 <= self.topk <= 16:
            raise ValueError("require tokens>0 and 1<=topk<=16")
        if not isinstance(self.chunk_tokens, int) or not 0 < self.chunk_tokens <= INT32_MAX:
            raise ValueError("chunk_tokens must be a positive int32")
        if (
            isinstance(self.first_chunk_splits, bool)
            or not isinstance(self.first_chunk_splits, int)
            or not 1 <= self.first_chunk_splits <= INT32_MAX
        ):
            raise ValueError("first_chunk_splits must be a positive int32")
        if self.experts <= 0 or self.experts % self.world:
            raise ValueError("experts must be divisible by world")
        if self.world * self.tokens * self.topk >= 2**31:
            raise ValueError("route IDs exceed int32")
        if self.total_chunks > INT32_MAX:
            raise ValueError("chunk completion IDs exceed int32")
        if self.gateway_control_bytes > GATEWAY_CONTROL_UB_BYTES:
            raise ValueError(
                f"chunk ranges/masks need {self.gateway_control_bytes} UB bytes, "
                f"exceeding {GATEWAY_CONTROL_UB_BYTES}; increase chunk_tokens "
                "(--num-tokens-per-chunk)"
            )
        if self.route not in ("balanced", "random", "duplicates", "invalid", "skew", "empty"):
            raise ValueError("unknown route")


def group_peer(rank, world, group, index, incoming=False, group_size=16):
    """Map a gateway's (producer group, position) to an actual source rank.

    Regroups first_hit_schedule::Peer's flattened order, including EP128 Staged
    topology. group_size counts local and cross ranks, not just cross peers.
    incoming preserves the legacy incoming-direction mapping for callers that
    need it; the Combine producer planner uses the default outgoing mapping.
    """
    # Regroup the existing topology order; do not change Dispatch's schedule.
    group, index = divmod(group * min(world, group_size) + index, min(world, 16))
    if world == 128:
        toggle, subround = divmod(group, 4)
        row, side = rank % 64 // 32, rank % 8 // 4
        peer_side, peer_row = side ^ (subround % 2), (row + subround // 2) % 2
        frame = rank // 64 ^ (peer_side if incoming else side) ^ toggle
        return frame * 64 + (peer_row * 4 + index // 4) * 8 + peer_side * 4 + index % 4
    width, half = min(world, 16), world // 2
    hw, groups = width // 2, world // width
    own = rank % half // hw
    target = (own + (-group if incoming else group)) % groups
    return index // hw * half + target * hw + index % hw


def group_steps(rank, world, group_size=16):
    width = min(world, group_size)
    steps = [0] * world
    for g in range(world // width):
        for i in range(width):
            steps[group_peer(rank, world, g, i, group_size=group_size)] = g
    return steps


def route_experts(source, token, case):
    local = case.experts // case.world
    lanes, servers = min(case.world, 8), (case.world + 7) // 8
    result = []
    state = (case.seed ^ (source * 0x9E3779B9) ^ (token * 0x85EBCA6B)) & 0xFFFFFFFF
    for k in range(case.topk):
        # Rotate INSIDE each server as well as across servers. A plain cyclic
        # global-rank sequence balances experts but biases first-hit to lane0/7.
        server = (k + token // lanes + source) % servers
        lane_offset = (k // servers) * 3
        if case.route == "balanced":
            # Keep first-hit unchanged, then rotate a permutation of the other
            # lanes. Each block of `lanes` tokens visits all gateway lanes;
            # successive blocks/sources rotate the nonzero contributor offsets.
            # A fixed +3 stride balances ranks but pins EP32/K8 gather to one
            # peer per sender. This schedule spreads those rows across 7 peers.
            lane_position = (k // servers) % lanes
            peer_rotation = (token // lanes + source + server) % (lanes - 1)
            lane_offset = 0 if lane_position == 0 else 1 + (lane_position - 1 + peer_rotation) % (lanes - 1)
        lane = (token + source * 3 + server * 5 + lane_offset) % lanes
        peer = server * 8 + lane
        expert = peer * local + (token // case.world + source + k // case.world) % local
        if case.route == "random":
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            expert = (state >> 8) % case.experts
        elif case.route == "skew":
            expert = (token + k) % local
        elif case.route == "empty" or (case.route == "invalid" and (source + token + k) % 5 == 0):
            expert = -1
        if case.route == "duplicates" and k == 1:
            expert = result[0]
        result.append(expert)
    return result


@dataclass
class CombinePlan:
    case: CombineCase
    rank: int
    forward: array  # int32 [source, capacity, (token_end, token, primary, peer, row, gather)]
    forward_count: array
    forward_capacity: int
    backward: array
    backward_count: array
    backward_capacity: int
    server_mask: array
    route_ids: array  # test-only: expert output row -> original route ID
    rank_rows: tuple
    gather_rows: tuple
    primary_counts: tuple
    chunk_ranges: array  # int32 [source_rank, raw-token chunk, (begin,end)]
    chunk_masks: array  # int32 [gateway group, chunk], remote contributors only

    @property
    def lanes(self):
        return min(self.case.world, 8)

    @property
    def chunks_per_group(self):
        return self.case.chunks_per_group

    @property
    def total_chunks(self):
        return self.case.total_chunks

    def forward_chunk_range(self, source, chunk):
        base = (source * self.chunks_per_group + chunk) * 2
        return tuple(self.chunk_ranges[base : base + 2])

    def forward_rows(self, source):
        base = source * self.forward_capacity * 6
        for i in range(self.forward_count[source]):
            yield tuple(self.forward[base + i * 6 : base + (i + 1) * 6])

    def backward_rows(self, lane):
        base = lane * self.backward_capacity * 3
        for i in range(self.backward_count[lane]):
            yield tuple(self.backward[base + i * 3 : base + (i + 1) * 3])


def build_plan(case, rank):
    """Build only this rank's tables; two streaming passes, no global hidden tensor.

    Output rows are stable expert/source/token/topk ordered. Gateway is the FIRST
    valid route per server; balanced fixtures rotate contributor/gateway pairs,
    not gateway semantics. Backward payloads are ordered by raw-token chunk,
    then the destination gateway's topology group, then source rank. Only each
    nonempty chunk's final payload carries its one-based completion ID; the
    gateway's contributor mask removes any need for empty control tasks.
    """
    case.validate()
    if not 0 <= rank < case.world:
        raise ValueError("invalid rank")
    w, t, k, e = case.world, case.tokens, case.topk, case.experts
    local = e // w
    sizes = [0] * e
    for source in range(w):
        for token in range(t):
            for expert in route_experts(source, token, case):
                if expert >= 0:
                    sizes[expert] += 1
    starts, rank_rows = [0] * e, []
    for peer in range(w):
        pos = 0
        for expert in range(peer * local, (peer + 1) * local):
            starts[expert] = pos
            pos += sizes[expert]
        rank_rows.append(pos)
    cursors, gathers, primary_counts = [0] * e, [0] * w, [0] * w
    ids = array("q", [-1]) * rank_rows[rank]
    forwards = [array("i") for _ in range(w)]
    backwards = [[] for _ in range(min(w, 8))]
    masks = array("q", [0]) * t
    for source in range(w):
        for token in range(t):
            hits = {}
            for slot, expert in enumerate(route_experts(source, token, case)):
                if expert < 0:
                    continue
                peer = expert // local
                row = starts[expert] + cursors[expert]
                cursors[expert] += 1
                if peer == rank:
                    ids[row] = (source * t + token) * k + slot
                hits.setdefault(peer // 8, []).append((peer, row))
            for server, records in hits.items():
                gateway, primary = records[0]
                primary_counts[gateway] += 1
                if source == rank:
                    masks[token] |= 1 << server
                if gateway == rank:
                    # One placeholder or len(records)-1 secondary rows; no boundary scan needed.
                    token_task_end = len(forwards[source]) // 6 + max(1, len(records) - 1)
                    if len(records) == 1:
                        forwards[source].extend((token_task_end, token, primary, -1, -1, -1))
                for peer, row in records[1:]:
                    reduce_row = -1
                    if peer != gateway:
                        reduce_row = gathers[gateway]
                        gathers[gateway] += 1
                        if peer == rank:
                            backwards[gateway % 8].append((source, row, reduce_row, token))
                    if gateway == rank:
                        forwards[source].extend((token_task_end, token, primary, peer, row, reduce_row))
    fc = array("i", (len(block) // 6 for block in forwards))
    fcap = max(1, max(fc))
    f = array("i", [-1]) * (w * fcap * 6)
    for source, block in enumerate(forwards):
        f[source * fcap * 6 : source * fcap * 6 + len(block)] = block
    bc = array("i", (len(block) for block in backwards))
    bcap = max(1, max(bc))
    b = array("q", [-1]) * (len(backwards) * bcap * 3)
    for lane, block in enumerate(backwards):
        steps = group_steps(rank // 8 * 8 + lane, w, case.group_size)
        block.sort(key=lambda row: (case.token_chunk(row[3]), steps[row[0]], row[0], row[2]))
        for i, row in enumerate(block):
            source, expert_row, target, token = row
            chunk_id = steps[source] * case.chunks_per_group + case.token_chunk(token) + 1
            next_chunk_id = (
                (steps[block[i + 1][0]] * case.chunks_per_group + case.token_chunk(block[i + 1][3]) + 1)
                if i + 1 < len(block)
                else -1
            )
            packed = target | ((chunk_id if next_chunk_id != chunk_id else 0) << 32)
            b[(lane * bcap + i) * 3 : (lane * bcap + i + 1) * 3] = array("q", (source, expert_row, packed))
    chunk_ranges = array("i", [0]) * (w * case.chunks_per_group * 2)
    chunk_masks = array("i", [0]) * case.total_chunks
    steps = group_steps(rank, w, case.group_size)
    for source, block in enumerate(forwards):
        cursor = 0
        for chunk in range(case.chunks_per_group):
            begin = cursor
            token_end = case.chunk_token_end(chunk)
            while cursor < fc[source] and block[cursor * 6 + 1] < token_end:
                peer = block[cursor * 6 + 3]
                if peer >= 0 and peer != rank:
                    chunk_masks[steps[source] * case.chunks_per_group + chunk] |= 1 << (peer % 8)
                cursor += 1
            base = (source * case.chunks_per_group + chunk) * 2
            chunk_ranges[base : base + 2] = array("i", (begin, cursor))
    plan = CombinePlan(
        case,
        rank,
        f,
        fc,
        fcap,
        b,
        bc,
        bcap,
        masks,
        ids,
        tuple(rank_rows),
        tuple(gathers),
        tuple(primary_counts),
        chunk_ranges,
        chunk_masks,
    )
    validate_plan(plan)
    return plan


def validate_plan(p):
    """Local safety/coverage checks before any collective device launch."""
    c, rank = p.case, p.rank
    c.validate()
    if not 0 <= rank < c.world or len(p.rank_rows) != c.world or len(p.gather_rows) != c.world:
        raise ValueError("bad rank/capacity vectors")
    if any(not 0 <= count <= INT32_MAX for count in (*p.rank_rows, *p.gather_rows)):
        raise ValueError("payload row counts exceed int32")
    if p.forward_capacity < 1 or len(p.forward) != c.world * p.forward_capacity * 6 or len(p.forward_count) != c.world:
        raise ValueError("bad forward shape")
    if (
        p.backward_capacity < 1
        or len(p.backward) != p.lanes * p.backward_capacity * 3
        or len(p.backward_count) != p.lanes
    ):
        raise ValueError("bad backward shape")
    if len(p.route_ids) != p.rank_rows[rank] or len(p.server_mask) != c.tokens:
        raise ValueError("bad payload/mask shape")
    if p.forward.typecode != "i" or p.forward.itemsize != 4 or p.backward.typecode != "q" or p.backward.itemsize != 8:
        raise ValueError("forward must use int32 and backward must use int64")
    if (
        p.chunk_ranges.typecode != "i"
        or p.chunk_ranges.itemsize != 4
        or len(p.chunk_ranges) != c.world * c.chunks_per_group * 2
    ):
        raise ValueError("bad chunk ranges shape/type")
    if p.chunk_masks.typecode != "i" or p.chunk_masks.itemsize != 4 or len(p.chunk_masks) != c.total_chunks:
        raise ValueError("bad chunk masks shape/type")
    seen_local, seen_gather, seen_backward = set(), set(), set()
    expected_masks = [0] * c.total_chunks
    own_steps = group_steps(rank, c.world, c.group_size)

    def own_local(row, source, token=None):
        if not 0 <= row < p.rank_rows[rank] or row in seen_local:
            raise ValueError("local expert row missing/duplicated/out of bounds")
        identity = p.route_ids[row]
        if not 0 <= identity < c.world * c.tokens * c.topk or identity // (c.tokens * c.topk) != source:
            raise ValueError("expert output source mismatch")
        if token is not None and identity // c.topk % c.tokens != token:
            raise ValueError("expert output token mismatch")
        seen_local.add(row)

    for source in range(c.world):
        if not 0 <= p.forward_count[source] <= p.forward_capacity:
            raise ValueError("forward count exceeds capacity")
        previous, primary, placeholder = -1, -1, False
        previous_end = 0
        for index, (token_end, token, pr, peer, row, reduce_row) in enumerate(p.forward_rows(source)):
            if not 0 <= token < c.tokens or token < previous:
                raise ValueError("forward token order invalid")
            if not index < token_end <= p.forward_count[source]:
                raise ValueError("forward token end out of bounds")
            if token != previous:
                if previous_end != index:
                    raise ValueError("forward token end does not match boundary")
            elif token_end != previous_end:
                raise ValueError("forward token end differs within token")
            previous_end = token_end
            if token != previous:
                own_local(pr, source, token)
                primary, placeholder = pr, False
            elif pr != primary or placeholder:
                raise ValueError("bad primary/placeholder run")
            if peer == -1:
                if token == previous or (row, reduce_row) != (-1, -1):
                    raise ValueError("bad primary-only placeholder")
                placeholder = True
            else:
                if not 0 <= peer < c.world or peer // 8 != rank // 8 or not 0 <= row < p.rank_rows[peer]:
                    raise ValueError("bad forward contributor")
                if peer == rank:
                    if reduce_row != -1:
                        raise ValueError("self contributor must not use gather")
                    own_local(row, source, token)
                else:
                    if not 0 <= reduce_row < p.gather_rows[rank] or reduce_row in seen_gather:
                        raise ValueError("duplicate/out-of-bounds gather row")
                    seen_gather.add(reduce_row)
                    chunk = own_steps[source] * c.chunks_per_group + c.token_chunk(token)
                    expected_masks[chunk] |= 1 << (peer % 8)
            previous = token
        if previous_end != p.forward_count[source]:
            raise ValueError("forward final token end mismatch")
        cursor = 0
        base = source * p.forward_capacity * 6
        for chunk in range(c.chunks_per_group):
            begin = cursor
            token_end = c.chunk_token_end(chunk)
            while cursor < p.forward_count[source] and p.forward[base + cursor * 6 + 1] < token_end:
                cursor += 1
            if p.forward_chunk_range(source, chunk) != (begin, cursor):
                raise ValueError("chunk range does not cover its complete token runs")
    for lane in range(p.lanes):
        if not 0 <= p.backward_count[lane] <= p.backward_capacity:
            raise ValueError("backward count exceeds capacity")
        gateway = rank // 8 * 8 + lane
        steps, previous_key, previous_marker = group_steps(gateway, c.world, c.group_size), None, 0
        for source, row, packed in p.backward_rows(lane):
            target, marker = unpack_reduce_offset(packed)
            if gateway == rank or not 0 <= source < c.world or not 0 <= target < p.gather_rows[gateway]:
                raise ValueError("invalid backward task")
            own_local(row, source)
            token = p.route_ids[row] // c.topk % c.tokens
            actual_chunk_id = steps[source] * c.chunks_per_group + c.token_chunk(token) + 1
            key = c.token_chunk(token), steps[source], source, target
            if previous_key is not None:
                if key < previous_key:
                    raise ValueError("backward tasks not in gateway chunk/group/source order")
                previous_id = previous_key[1] * c.chunks_per_group + previous_key[0] + 1
                expected_marker = previous_id if previous_id != actual_chunk_id else 0
                if previous_marker != expected_marker:
                    raise ValueError("backward chunk marker must be on its final payload row")
            if marker not in (0, actual_chunk_id):
                raise ValueError("backward chunk marker does not match payload token/group")
            if (gateway, target) in seen_backward:
                raise ValueError("duplicate backward reduce row")
            seen_backward.add((gateway, target))
            previous_key, previous_marker = key, marker
        if previous_key is not None and previous_marker != previous_key[1] * c.chunks_per_group + previous_key[0] + 1:
            raise ValueError("backward final payload row lacks chunk marker")
    if len(seen_local) != p.rank_rows[rank] or len(seen_gather) != p.gather_rows[rank]:
        raise ValueError("incomplete local/gather coverage")
    if list(p.chunk_masks) != expected_masks:
        raise ValueError("chunk contributor mask mismatch")
    for token, actual in enumerate(p.server_mask):
        expected = 0
        for expert in route_experts(rank, token, c):
            if expert >= 0:
                expected |= 1 << (expert // (c.experts // c.world) // 8)
        if actual != expected:
            raise ValueError("source server mask mismatch")
