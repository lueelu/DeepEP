# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Offline timer decoding; no Torch/NPU imports."""


def parse_rank_list(value, limit):
    result = []
    for part in value.split(","):
        bounds = part.split("-")
        if len(bounds) == 1:
            result.append(int(part))
        elif len(bounds) == 2:
            start, end = map(int, bounds)
            if start > end:
                raise ValueError("Rank range must be ascending")
            result.extend(range(start, end + 1))
        else:
            raise ValueError("Use ranks such as 0-7 or 0,3,7")
    if not result or min(result) < 0 or max(result) >= limit or len(set(result)) != len(result):
        raise ValueError(f"Ranks must be unique integers in [0,{limit - 1}]")
    return result
