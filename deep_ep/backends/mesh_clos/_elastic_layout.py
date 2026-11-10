# Copyright (c) 2026, Lu Lu

"""Host workspace layouts for Elastic Notify, dispatch and combine."""

WORLD_SIZES = (2, 4, 8, 16, 32, 64, 128, 256, 512)
MAX_BYTES = (1 << 63) - 1
PAGE_BYTES = 2 << 20


def integer(name, value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def validate_hidden(hidden, fp8=False):
    integer("hidden", hidden, 1, 7168)
    alignment = 128 if fp8 else 16
    if hidden % alignment:
        raise ValueError(f"hidden must be a multiple of {alignment}")


def workspace_layout(world, tokens, topk, experts=1024, chunk_tokens=256):
    """Mirror MakeNotifyTiling; launch also compares this with the native result."""
    if type(world) is not int or world not in WORLD_SIZES:
        raise ValueError("Notify supports EP2/4/8/16/32/64/128/256/512")
    integer("tokens", tokens, 1, 16384)
    integer("topk", topk, 1, 16)
    integer("experts", experts, world, 1024)
    if experts % world:
        raise ValueError("num_experts must be divisible by EP")
    if world * tokens * topk >= 2**31:
        raise ValueError("Notify int32 row indexing requires EP*T*K < 2**31")

    def align(n):
        return (n + 511) // 512 * 512

    if type(chunk_tokens) is not int or chunk_tokens not in (256, 512, 1024):
        raise ValueError("Notify chunk_tokens must be 256, 512 or 1024")
    fcap = tokens * topk
    bcap = world * fcap
    chunks = (tokens + chunk_tokens - 1) // chunk_tokens + min(4, tokens, chunk_tokens) - 1
    lanes = min(world, 8)
    cstride, fstride, bstride = align(experts * 4), align(64 + fcap * 24), align(64 + bcap * 24)
    regions = (
        ("count_send", cstride),
        ("count_recv", world * cstride),
        ("forward_send", world * fstride),
        ("forward_recv", world * fstride),
        ("backward_send", lanes * bstride),
        ("backward_recv", lanes * bstride),
        ("gather_send", align(world * 4)),
        ("gather_recv", world * 512),
        ("backward_task_offsets", align(lanes * world * chunks * 4)),
        ("forward_tile_offsets", align(world * ((tokens + 31) // 32) * 4)),
        ("gather_tile_offsets", align(world * ((fcap + 31) // 32) * 4)),
        ("backward_group_offsets", align(lanes * (world // min(world, 64)) * chunks * 2 * 4)),
        ("source_chunk_masks", align(world * chunks * 4)),
        ("expert_totals", align(experts * 4)),
        ("address_stride", 512),
        ("local_prefix_segments", align(((tokens + 255) // 256) * ((experts + world + 15) // 16 * 16) * 4)),
    )
    layout, cursor = {}, 0
    for name, size in regions:
        layout[name] = cursor
        cursor += size
    layout["workspace_bytes"] = cursor
    layout["num_bytes"] = max(PAGE_BYTES, (cursor + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES)
    if layout["num_bytes"] > MAX_BYTES:
        raise ValueError("Notify workspace exceeds the int64 byte-size limit")
    return layout


def dispatch_layout(world, tokens, topk, experts, rows, fp8=False, weights=False, hidden=7168):
    if world not in WORLD_SIZES:
        raise ValueError("Payload dispatch supports EP2/4/8/16/32/64/128/256/512")
    notify = workspace_layout(world, tokens, topk, experts)
    integer("receive capacity", rows, 1, world * tokens * topk)
    stride = 1 << (rows - 1).bit_length()
    if world * stride >= 2**63:
        raise ValueError("Dispatch power-of-two address encoding exceeds int64")
    if type(fp8) is not bool:
        raise TypeError("fp8 must be bool")
    if type(weights) is not bool:
        raise TypeError("weights must be bool")
    validate_hidden(hidden, fp8)
    routes = tokens * topk
    weight_capacity = (routes + 59) // 60 * 60 if weights else 0
    weight_inbox_stride = 512 + weight_capacity // 60 * 512 if weight_capacity else 0
    row_bytes = hidden * (1 if fp8 else 2)
    scale_shards = 32 // world if world < 32 else 1
    shard_routes = (tokens * topk + scale_shards - 1) // scale_shards
    inbox_stride = scale_shards * (512 + ((shard_routes + 1) // 2) * 512) if fp8 else 0
    regions = (
        ("sync", world * 49152 + 65536),
        ("book", world * 16),
        ("input", tokens * row_bytes),
        ("output", rows * row_bytes),
        ("weights", rows * 4),
        ("scales", rows * (hidden // 128) * 4 if fp8 else 0),
        ("inbox", world * inbox_stride),
        ("peer_book", world * 512),
        ("done", world * 3 * 512),  # Clos QP0/QP1 and FM completion slots.
        ("weight_inbox", world * weight_inbox_stride),
    )
    layout, cursor = {}, notify["workspace_bytes"]
    for name, size in regions:
        layout[name] = cursor
        cursor += (size + 511) // 512 * 512
    layout.update(workspace_bytes=cursor, address_stride=stride, inbox_stride=inbox_stride)
    layout.update(weight_capacity=weight_capacity, weight_inbox_stride=weight_inbox_stride)
    layout.update(
        aggregate_protocol=1,
        aggregate_packet_bytes=512,
        scale_batch_rows=(64 if world <= 64 else 56 if world <= 128 else 24 if world <= 256 else 8) if fp8 else 0,
    )
    layout["num_bytes"] = max(PAGE_BYTES, (cursor + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES)
    if layout["num_bytes"] > MAX_BYTES:
        raise ValueError("Dispatch workspace exceeds the int64 byte-size limit")
    return layout


def _combine_resident_bytes(world, tokens, chunk_tokens=256):
    chunks = (tokens + chunk_tokens - 1) // chunk_tokens + min(4, tokens, chunk_tokens) - 1
    groups = world // min(world, 64)
    return ((world * chunks * 8 + 31) // 32 + (groups * chunks * 4 + 31) // 32) * 32


def combine_layout(world, tokens, topk, rows, gather_rows, chunk_tokens=256, hidden=7168):
    if world not in WORLD_SIZES:
        raise ValueError("Combine supports EP2/4/8/16/32/64/128/256/512")
    integer("tokens", tokens, 1, 16384)
    integer("topk", topk, 1, 16)
    integer("rows", rows, 1, world * tokens * topk)
    integer("gather_rows", gather_rows, 1, world * tokens * topk)
    integer("chunk_tokens", chunk_tokens, 1, 16384)
    validate_hidden(hidden)
    if world * tokens * topk >= 2**31:
        raise ValueError("Combine metadata indexing exceeds int32")
    chunks = (tokens + chunk_tokens - 1) // chunk_tokens + min(4, tokens, chunk_tokens) - 1
    groups = world // min(world, 64)
    slots = 3 * world + 96 + groups * chunks * 8 + 64 + world * chunks * 2 + 64
    cursor, result = slots * 512, {"control_slot_num": slots}
    for name, count in (
        ("gather_input", gather_rows),
        ("server_partial", rows),
        ("returned_partial", tokens * max((world + 7) // 8, topk)),
        ("output", tokens),
    ):
        result[name] = cursor
        cursor += (count * hidden * 2 + 511) // 512 * 512
    for name, size in (
        ("weight_recv", tokens * topk * 4),
        ("weight_local_done", 64 * 512),
        ("weight_done", world * 512),
    ):
        result[name] = cursor
        cursor += (size + 511) // 512 * 512
    result["workspace_bytes"] = cursor
    if cursor > MAX_BYTES:
        raise ValueError("Combine workspace exceeds the int64 byte-size limit")
    return result


def elastic_size_hint(world, tokens, topk, experts, rows, gather_rows, fp8=False, hidden=7168):
    dispatch = dispatch_layout(world, tokens, topk, experts, rows, fp8, weights=True, hidden=hidden)["workspace_bytes"]
    combine = combine_layout(world, tokens, topk, rows, gather_rows, hidden=hidden)["workspace_bytes"]
    # A fixed boundary prevents fresh/cached calls with different shapes or
    # BF16/FP8 dispatch from ever overlapping outstanding combine output copies.
    half = (max(dispatch, combine) + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES
    total = 2 * half
    if total > MAX_BYTES:
        raise ValueError("Elastic workspace exceeds the int64 byte-size limit")
    return total
