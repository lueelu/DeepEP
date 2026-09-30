# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Bounded CPU diagnostics for the independent peer-table payload oracle."""

import numpy as np


class PayloadCheck:
    def __init__(self, rank, world, sources, tokens, slots, experts, sample_limit=4):
        self.rank, self.world = rank, world
        self.sources = np.asarray(sources, dtype=np.int64)
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.slots = np.asarray(slots, dtype=np.int64)
        self.experts = np.asarray(experts, dtype=np.int64)
        self.sample_limit = sample_limit
        self.fields = {}
        # The golden output is expert-major; transport packs token/top-k order
        # separately for each source->destination. Recover that order without
        # reading device Notify metadata.
        self.packed_indices = np.empty(len(self.sources), dtype=np.int64)
        self.source_counts = np.empty(len(self.sources), dtype=np.int64)
        self.scale_batch = 64 if world <= 64 else 56 if world <= 128 else 24
        for source in np.unique(self.sources):
            rows = np.flatnonzero(self.sources == source)
            order = np.lexsort((self.slots[rows], self.tokens[rows]))
            self.packed_indices[rows[order]] = np.arange(len(rows))
            self.source_counts[rows] = len(rows)

    def add(self, name, first, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.ndim == 1:
            actual, expected = actual[:, None], expected[:, None]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(
                f"{name} shape/dtype mismatch: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}"
            )
        # Retain exact FP32 bit patterns, including NaNs and signed zeros.
        floating = actual.dtype == np.float32
        a = actual.view(np.uint32) if floating else actual
        e = expected.view(np.uint32) if floating else expected
        bad = a != e
        state = self.fields.setdefault(
            name,
            dict(
                checked_rows=0,
                bad_values=0,
                rows=np.zeros(len(self.sources), dtype=bool),
                columns=np.zeros(actual.shape[1], dtype=np.int64),
                samples=[],
            ),
        )
        state["checked_rows"] += len(actual)
        state["bad_values"] += int(bad.sum())
        state["rows"][first : first + len(actual)] = bad.any(axis=1)
        state["columns"] += bad.sum(axis=0)
        for local in np.flatnonzero(bad.any(axis=1))[: max(0, self.sample_limit - len(state["samples"]))]:
            row = first + int(local)
            column = int(np.flatnonzero(bad[local])[0])
            source = int(self.sources[row])
            relative = (self.rank - source) % self.world
            sample = dict(
                row=row,
                column=column,
                source_rank=source,
                token=int(self.tokens[row]),
                topk_slot=int(self.slots[row]),
                expert=int(self.experts[row]),
                cross_64_rank_boundary=source // 64 != self.rank // 64,
                sender_aiv=32 + relative % 32,
                sender_peer_ordinal=relative // 32,
                receiver_aiv=32 + source % 32,
                bad_values_in_row=int(bad[local].sum()),
                actual_bits=hex(int(a[local, column])),
                expected_bits=hex(int(e[local, column])),
                actual_value=str(actual[local, column]),
                expected_value=str(expected[local, column]),
            )
            # A whole scale row reveals shifts/repeated patterns; hidden stays bounded.
            begin, end = (
                (0, actual.shape[1]) if name == "scale" else (max(0, column - 4), min(actual.shape[1], column + 12))
            )
            sample.update(
                sample_column_begin=begin,
                actual_row_bits=[hex(int(v)) for v in a[local, begin:end]],
                expected_row_bits=[hex(int(v)) for v in e[local, begin:end]],
            )
            if name == "scale":
                index, count = int(self.packed_indices[row]), int(self.source_counts[row])
                batch = index // self.scale_batch
                sample.update(
                    scale_locality="cross_pod" if source // 64 != self.rank // 64 else "intra_pod",
                    packed_index=index,
                    source_rows=count,
                    batch_rows=self.scale_batch,
                    batch_index=batch,
                    row_in_batch=index % self.scale_batch,
                    tail_batch=batch == (count - 1) // self.scale_batch,
                    wire_block=index // 2,
                    wire_flag_offset=512 + (index // 2) * 512 + 504,
                    wire_slot_offset=512 + (index // 2) * 512 + 448 + (index % 2) * 4,
                    ub_face=batch % 2,
                    completion_event=(2 * (relative // 32) + batch % 2) % 4,
                    bad_columns=[int(v) for v in np.flatnonzero(bad[local])],
                    zero_columns=[int(v) for v in np.flatnonzero(a[local] == 0)],
                )
            state["samples"].append(sample)

    def summary(self):
        result = {}
        for name, state in self.fields.items():
            bad_rows = state["rows"]
            per_source = []
            for source in np.unique(self.sources):
                selected = self.sources == source
                item = dict(
                    source_rank=int(source), rows=int(selected.sum()), bad_rows=int((selected & bad_rows).sum())
                )
                if name == "scale":
                    indices = self.packed_indices[selected & bad_rows]
                    tail_start = (int(selected.sum()) - 1) // self.scale_batch * self.scale_batch
                    item.update(
                        first_bad_packed_index=int(indices.min()) if len(indices) else None,
                        last_bad_packed_index=int(indices.max()) if len(indices) else None,
                        tail_bad_rows=int((indices >= tail_start).sum()),
                        earlier_bad_rows=int((indices < tail_start).sum()),
                    )
                per_source.append(item)
            cross = self.sources // 64 != self.rank // 64
            result[name] = dict(
                checked_rows=state["checked_rows"],
                bad_rows=int(bad_rows.sum()),
                bad_values=state["bad_values"],
                cross_64_bad_rows=int((bad_rows & cross).sum()),
                same_64_bad_rows=int((bad_rows & ~cross).sum()),
                bad_values_by_column={str(int(i)): int(state["columns"][i]) for i in np.flatnonzero(state["columns"])},
                by_source=per_source,
                samples=state["samples"],
            )
        return result
