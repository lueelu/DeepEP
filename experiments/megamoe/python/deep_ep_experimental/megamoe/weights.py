"""Prepared expert weights. Updates run outside capture under caller-owned stream ordering."""

from functools import wraps
from threading import RLock

from . import api
from .api import _SharedFp8Pair, _check_fp4_weight_format, _interleave_gate_up_128, _transform_nd_pairs


_MISSING = object()


def _format_class(tensor):
    import torch_npu

    fmt = torch_npu.get_npu_format(tensor)
    return 2 if fmt in (0, 2) or (fmt == 30 and tensor.is_contiguous()) else fmt


class WeightHandle:
    def __init__(self, owner):
        self.owner = owner


def _nz_storage(weight):
    """Recover the tensor whose shape describes this full packed NZ allocation."""
    owner = getattr(weight, "_megamoe_nz_owner", None)
    if owner is None:
        owner = getattr(weight, "_base", None)
    if owner is None:
        owner = weight
    storage = owner.untyped_storage()
    nbytes = weight.numel() * weight.element_size()
    if (
        owner.dtype != weight.dtype
        or owner.device != weight.device
        or owner.ndim not in (2, 3)
        or tuple(owner.shape[-2:]) != tuple(weight.shape[-2:])
        or owner.numel() != weight.numel()
        or not owner.is_contiguous()
        or not weight.is_contiguous()
        or owner.storage_offset()
        or weight.storage_offset()
        or owner.data_ptr() != weight.data_ptr()
        or storage.data_ptr() != weight.data_ptr()
        or weight.untyped_storage().data_ptr() != storage.data_ptr()
        or storage.nbytes() != nbytes
        or weight.untyped_storage().nbytes() != nbytes
        or _format_class(owner) != _format_class(weight)
    ):
        raise ValueError("packed NZ weight requires independent zero-offset storage covering the complete allocation")
    _check_fp4_weight_format(owner, "physical expert weight")
    return owner


def _bound_weight(weight):
    # Only list normalization introduces an NZ view of a different physical shape.
    return _nz_storage(weight) if hasattr(weight, "_megamoe_nz_owner") else weight


def _fp4_cast(weight, fmt):
    import torch
    import torch_npu

    return torch_npu.npu_format_cast(
        weight, fmt, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
    )


def _copy_prepared(dst, src):
    import torch

    if dst.dtype == torch.uint8:
        dst, src = _bound_weight(dst), _bound_weight(src)
        if dst.shape != src.shape:
            # Shape changes on ND bytes are unambiguous; CANN sees actual NZ allocation shapes at both ends.
            src = _fp4_cast(_fp4_cast(src, 2).reshape(dst.shape), 29)
    dst.copy_(src)


def _reorder(pair, inverse):
    import torch

    weight, scale = pair
    physical = _bound_weight(weight)
    nd = _fp4_cast(physical, 2).reshape(weight.shape)

    def reorder(t):
        if not inverse:
            return _interleave_gate_up_128(t)
        e, n, *tail = t.shape
        return t.reshape((e, n // 256, 2, 128, *tail)).transpose(1, 2).contiguous().reshape(t.shape)

    nz = _fp4_cast(reorder(nd).reshape(physical.shape), 29)
    if nz.ndim == 2:
        nz = _single((nz, scale), "prepared")[0]
    reordered = reorder(scale.view(torch.uint8))
    if reordered.data_ptr() == scale.data_ptr():
        reordered = reordered.clone()
    return nz, reordered.view(torch.float8_e8m0fnu)


def _validate(l1, l2, planar):
    import torch

    w1, w2 = l1[0], l2[0]
    fp4 = w1.dtype == torch.uint8
    if w1.ndim != 3 or w2.ndim != 3 or w2.dtype != w1.dtype or (not fp4 and w1.dtype != torch.float8_e4m3fn):
        raise ValueError("invalid prepared projection shape or dtype")
    e, n, packed = w1.shape
    h = packed * (2 if fp4 else 1)
    if min(e, n, h) <= 0 or n % 256 or h % 64 or tuple(w2.shape) != (e, h, n // (4 if fp4 else 2)):
        raise ValueError("incompatible prepared projection dimensions")
    for w, scale in (l1, l2, planar):
        if fp4:
            _check_fp4_weight_format(w, "prepared")
        elif _format_class(w) != 2:
            raise ValueError("FP8 weights require ND storage")
        if (
            w.device.type != "npu"
            or w.device != w1.device
            or scale.device != w.device
            or not w.is_contiguous()
            or not scale.is_contiguous()
            or _format_class(scale) != 2
            or scale.dtype != torch.float8_e8m0fnu
            or tuple(scale.shape) != (*w.shape[:2], (w.shape[-1] * (2 if fp4 else 1) + 63) // 64, 2)
        ):
            raise ValueError("invalid prepared weight or E8M0 scale layout")
    if planar[0].shape != w1.shape or planar[0].dtype != w1.dtype:
        raise ValueError("planar and fused GMM1 must match")


def _prepare(l1, l2, shared, layout, planar_l1, *, single_expert=False):
    import torch

    def pair(value):
        w, s = value
        if layout != "nd" and s.dtype == torch.int32:
            if w.ndim != 3:
                raise ValueError("prepared weights must have three dimensions")
            k = w.shape[-1] * (2 if w.dtype == torch.uint8 else 1)
            if k % 128 or tuple(s.shape) != (*w.shape[:2], k // 128) or not s.is_contiguous() or _format_class(s) != 2:
                raise ValueError("invalid packed int32 scale shape or storage")
            unpacked = s.view(torch.uint8).view(torch.float8_e8m0fnu).reshape((*w.shape[:2], k // 64, 2))
            if unpacked.data_ptr() != s.data_ptr() or _format_class(unpacked) != 2:
                raise ValueError("packed scale reinterpretation must preserve linear storage")
            return w, unpacked
        if s.dtype == torch.uint8 and _format_class(s) != 2:
            raise ValueError("packed E8M0 scale requires linear storage")
        return w, s.view(torch.float8_e8m0fnu) if s.dtype == torch.uint8 else s

    l1, l2 = pair(l1), pair(l2)
    dual = not shared and l1[0].dtype in (torch.uint8, torch.int8)
    if layout == "nd":
        if planar_l1 is not None:
            raise ValueError("planar_l1 belongs to prepared input")
        l1, l2 = _transform_nd_pairs(l1, l2, shared=shared, single_expert=single_expert)
        planar = getattr(l1, "_planar", l1)
    elif layout == "planar_nz":
        if l1[0].dtype != torch.uint8 or planar_l1 is not None:
            raise ValueError("planar_nz requires packed NZ weights")
        planar = l1
        _validate(planar, l2, planar)
        if dual:
            l1 = _reorder(planar, False)
    elif layout == "prepared":
        if dual != (planar_l1 is not None):
            raise ValueError("prepared routed FP4 requires planar_l1")
        planar = pair(planar_l1) if dual else l1
    else:
        raise ValueError("input_layout must be nd, planar_nz or prepared")
    _validate(l1, l2, planar)
    l1 = _SharedFp8Pair(l1) if shared and l1[0].dtype == torch.float8_e4m3fn else tuple(l1)
    return l1, tuple(l2), tuple(planar)


def _expert_group(pair):
    """Lists contain expert pairs. A tuple groups parallel weights and scales."""
    sequences = (list, tuple)
    if isinstance(pair, list):
        if not pair or any(not isinstance(item, sequences) or len(item) != 2 for item in pair):
            raise ValueError("supply a nonempty list of expert weight/scale pairs")
        return pair
    if isinstance(pair, tuple) and len(pair) == 2:
        weights, scales = pair
        if isinstance(weights, sequences) or isinstance(scales, sequences):
            if (
                not isinstance(weights, sequences)
                or not isinstance(scales, sequences)
                or not weights
                or len(weights) != len(scales)
            ):
                raise ValueError("supply nonempty parallel weight and scale sequences of equal length")
            return list(zip(weights, scales))
    return None


def _single(pair, layout):
    import torch

    if any(not isinstance(t, torch.Tensor) for t in pair):
        raise TypeError("expert entries require weight and scale tensors")
    w, s = pair
    if layout == "nd" and any(_format_class(t) != 2 for t in pair):
        raise ValueError("raw weight and scale entries require linear ND storage")

    def expand(t, rank):
        if t.ndim == rank + 1 and t.shape[0] == 1:
            return t
        if t.ndim != rank:
            raise ValueError("list entries require a single expert with an optional leading dimension of one")
        return t.unsqueeze(0)

    expanded = expand(w, 2), expand(s, 2 if s.dtype == torch.int32 else 3)
    for tensor, view in zip(pair, expanded):
        if (
            view.data_ptr() != tensor.data_ptr()
            or view.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
            or view.dtype != tensor.dtype
            or _format_class(tensor) != _format_class(view)
            or (layout != "nd" and not tensor.is_contiguous())
        ):
            raise ValueError("adding the expert dimension must preserve storage and format")
    if layout != "nd" and w.dtype == torch.uint8:
        owner = _nz_storage(w)
        if expanded[0] is not owner:
            expanded[0]._megamoe_nz_owner = owner
    return expanded


def _records(l1, l2, shared, layout, planar):
    groups = _expert_group(l1), _expert_group(l2)
    if groups == (None, None):
        if planar is not None and _expert_group(planar) is not None:
            raise ValueError("planar_l1 must match the continuous projections")
        return False, [_prepare(l1, l2, shared, layout, planar)]
    if any(g is None or not len(g) for g in groups):
        raise ValueError("supply complete nonempty lists for both projections")
    planes = [None] * len(groups[0]) if planar is None else _expert_group(planar)
    if planes is None:
        raise ValueError("planar_l1 must match the routed list")

    if not (len(planes) == len(groups[0]) == len(groups[1])):
        raise ValueError("list projections and planar_l1 must have the same expert count")

    def normalized(group):
        entries = [_single(pair, layout) for pair in group]

        def signature(entry):
            # Stride of a size-one expert dimension has no effect on address calculation.
            return tuple((t.shape, t.dtype, t.device, t.stride()[1:], _format_class(t)) for t in entry)

        expected = signature(entries[0])
        if any(signature(entry) != expected for entry in entries[1:]):
            raise ValueError("all list experts must share one shape, dtype, stride, device and format")
        return entries

    records = [
        _prepare(x, y, shared, layout, p, single_expert=True)
        for x, y, p in zip(
            normalized(groups[0]),
            normalized(groups[1]),
            normalized(planes) if planar is not None else planes,
            strict=True,
        )
    ]
    signature = _signature(records[0])
    if any(_signature(record) != signature for record in records):
        raise ValueError("all list experts must share one shape, dtype, stride and device")
    return True, records


def _outside_capture(device):
    npu = api.npu_api()
    with npu.device(device):
        if npu.is_current_stream_capturing():
            raise RuntimeError("expert weight preparation and updates require execution outside graph capture")


def _mutation(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            _outside_capture(self._device)
            return method(self, *args, **kwargs)

    return wrapped


def _slots(record, dual):
    for pair in record[: 3 if dual else 2]:
        yield from pair


def _extent(tensor):
    import torch

    storage = tensor.untyped_storage()
    start = tensor.data_ptr()
    if tensor.dtype == torch.uint8:
        if tensor.storage_offset():
            raise ValueError("each NZ weight needs independent zero-offset storage")
        return start, start + storage.nbytes()
    return start, start + tensor.numel() * tensor.element_size()


def _validate_slots(records, dual):
    intervals = sorted(_extent(t) for r in records for t in _slots(r, dual))
    if any(end > start for (_, end), (start, _) in zip(intervals, intervals[1:])):
        raise ValueError("expert weight and scale storage intervals overlap")


def _signature(record):
    return tuple(
        (t.shape, t.dtype, t.device, t.stride()[1:] if t.shape[0] == 1 else t.stride()) for pair in record for t in pair
    )


class PreparedWeights:
    def __init__(self, l1, l2, shared, layout, planar):
        import torch

        group = _expert_group(l1)
        weight = (group[0] if group is not None else api._pair(l1, "l1_weights"))[0]
        if not isinstance(weight, torch.Tensor) or weight.device.type != "npu":
            raise ValueError("expert weights require an NPU tensor")
        self._device = weight.device
        _outside_capture(self._device)
        self._lock = RLock()
        self.shared = shared
        self.list_mode, self._records = _records(l1, l2, shared, layout, planar)

        self._dual = not shared and self._records[0][0][0].dtype == torch.uint8
        _validate_slots(self._records, self._dual)
        self._remember_storage()
        self.count = len(self._records) if self.list_mode else self._records[0][0][0].shape[0]
        self.l1, self.l2 = WeightHandle(self), WeightHandle(self)
        self._tables = {}
        self._retired = []
        self._retirement_batches = []
        self._exported = {}
        self._valid = True
        self._pending = set()
        self._staged_aliases = {}
        self._dirty_tables = False
        if self.list_mode:
            self._publish_tables()
        self.commit_prepared_update()

    def _addresses(self):
        return tuple(
            (_bound_weight(t).untyped_storage().data_ptr(), t.data_ptr())
            for r in self._records
            for t in _slots(r, self._dual)
        )

    def _remember_storage(self):
        self._addresses_at_bind = self._addresses()
        self.storage_bases = {base for base, _ in self._addresses_at_bind}

    def _check_bindings(self):
        if any(
            t.data_ptr() != ptr or t.untyped_storage().data_ptr() != base
            for views in self._exported.values()
            for pair in views.values()
            for t, ptr, base in pair
        ):
            raise RuntimeError("exported expert view storage was rebound")
        try:
            addresses = self._addresses()
        except ValueError as exc:
            raise RuntimeError("prepared weight storage was rebound") from exc
        if addresses != self._addresses_at_bind:
            raise RuntimeError("prepared weight storage was rebound")

    def _publish_tables(self):
        import torch

        for fused, index in ((True, 0), (False, 2)):
            rows = [[r[p][s].data_ptr() for r in self._records] for p, s in ((index, 0), (1, 0), (index, 1), (1, 1))]
            host = torch.tensor(rows, dtype=torch.int64, device="cpu", pin_memory=True)
            self._retired.append(host)
            if fused not in self._tables:
                self._tables[fused] = torch.empty_like(host, device=self._records[0][0][0].device)
            self._tables[fused].copy_(host, non_blocking=True)

    def select(self, fused):
        if not self._valid:
            raise RuntimeError("weight update incomplete")
        first, second, planar = self._records[0]
        return (first if fused or self.shared else planar), second, self._tables.get(fused)

    def expert_views(self, expert=None, *, layout="planar"):
        import torch

        if layout not in ("planar", "fused"):
            raise ValueError("layout must be planar or fused")
        if expert is None:
            first = 2 if layout == "planar" else 0
            if not self.list_mode:
                return self._records[0][first], self._records[0][1]
            views = [self.expert_views(i) for i in range(self.count)]

            def projection(index, key):
                weights = [
                    _nz_storage(record[index][0]) if record[index][0].dtype == torch.uint8 else view[key][0]
                    for record, view in zip(self._records, views)
                ]
                return weights, [view[key][1] for view in views]

            return projection(first, "planar_l1" if first == 2 else "l1"), projection(1, "l2")
        if not 0 <= expert < self.count:
            raise IndexError("expert out of range")
        if expert in self._exported:
            return {name: tuple(entry[0] for entry in pair) for name, pair in self._exported[expert].items()}
        record = self._records[expert if self.list_mode else 0]
        i = 0 if self.list_mode else expert

        views = {}
        for name, pair in zip(("l1", "l2", "planar_l1"), record):
            weight, scale = pair
            single_nz = self.list_mode and weight.dtype == torch.uint8
            physical = _bound_weight(weight) if single_nz else weight
            w = physical if single_nz and physical.ndim == 2 else weight[i]
            slices = w, scale[i]
            if any(v.untyped_storage().data_ptr() != t.untyped_storage().data_ptr() for v, t in zip(slices, pair)):
                raise RuntimeError("backend did not return a zero-copy expert view")
            if w.dtype == torch.uint8:
                _check_fp4_weight_format(w, "exported expert view")
                if single_nz and w is not physical:
                    w._megamoe_nz_owner = physical
            views[name] = slices
        self._exported[expert] = {
            name: tuple((t, t.data_ptr(), t.untyped_storage().data_ptr()) for t in pair) for name, pair in views.items()
        }
        return views

    def _versions(self):
        return tuple(t._version if not t.is_inference() else None for r in self._records for p in r for t in p)

    def check_versions(self):
        self._check_bindings()
        if not self._valid or self._versions() != self._snapshot:
            raise RuntimeError("expert weights changed without a complete update")
        return True

    @_mutation
    def commit_prepared_update(self):
        """Publish a complete externally ordered update to all weight representations."""
        self._check_bindings()
        if self._dirty_tables:
            self._publish_tables()
            self._dirty_tables = False
        if self._retired:
            event = api.npu_api().Event()
            event.record(api.npu_api().current_stream(self._records[0][0][0].device))
            self._retirement_batches.append((event, self._retired))
            self._retired = []
        self._snapshot = self._versions()
        self._pending.clear()
        self._staged_aliases.clear()
        self._valid = True

    @_mutation
    def release_retired(self):
        """Call only after the framework has drained older readers and copies."""
        self._retirement_batches = [(e, refs) for e, refs in self._retirement_batches if not e.query()]

    def _update(self, indices, records, replace):
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(self._records) for i in indices)
        ):
            raise ValueError("invalid updated expert indices")
        for i, r in zip(indices, records, strict=True):
            if _signature(self._records[i]) != _signature(r):
                raise ValueError("replacement must preserve weight layout")
        _validate_slots(records, self._dual)
        if replace:
            future = self._records.copy()
            for i, r in zip(indices, records):
                future[i] = r
            _validate_slots(future, self._dual)
        if self._pending.difference(indices):
            raise RuntimeError("retry all experts from the incomplete update")
        self._check_bindings()
        self._valid = False
        self._pending.update(indices)
        self._dirty_tables |= replace
        self._retired.extend(records)
        if replace:
            self._retired.extend(self._records[i] for i in indices)
            for i in indices:
                self._exported.pop(i, None)
            self._records = future
            self._remember_storage()
        else:
            copies = [
                (dst, src)
                for i, new in zip(indices, records)
                for dst, src in zip(_slots(self._records[i], self._dual), _slots(new, self._dual))
                if _extent(dst) != _extent(src)
            ]
            targets = [_extent(dst) for dst, _ in copies]
            staged = []
            for dst, src in copies:
                begin, end = _extent(src)
                interval = (begin, end)
                if interval in self._staged_aliases:
                    src = self._staged_aliases[interval]
                elif any(begin < stop and start < end for start, stop in targets):
                    # Keep the pre-write value across a partial failure and retry.
                    import torch

                    src = (
                        _fp4_cast(_fp4_cast(_bound_weight(src), 2).clone(), 29)
                        if src.dtype == torch.uint8
                        else src.clone()
                    )
                    self._staged_aliases[interval] = src
                    self._retired.append(src)
                staged.append((dst, src))
            for dst, src in staged:
                _copy_prepared(dst, src)
        self.commit_prepared_update()

    @_mutation
    def update(self, l1, l2, *, input_layout="nd", planar_l1=None):
        if self.list_mode:
            return self.update_experts(range(self.count), l1, l2, input_layout=input_layout, planar_l1=planar_l1)
        self._update([0], [_prepare(l1, l2, self.shared, input_layout, planar_l1)], False)

    @_mutation
    def update_experts(
        self,
        experts=_MISSING,
        l1=_MISSING,
        l2=_MISSING,
        *,
        indices=_MISSING,
        replace=False,
        input_layout="nd",
        planar_l1=None,
    ):
        if (indices is _MISSING) == (experts is _MISSING) or l1 is _MISSING or l2 is _MISSING:
            raise TypeError("supply one of indices or experts and both l1 and l2")
        selected = list(experts if indices is _MISSING else indices)
        if not self.list_mode:
            raise ValueError("update_experts requires a list owner")
        is_list, records = _records(l1, l2, self.shared, input_layout, planar_l1)
        if not is_list:
            raise ValueError("supply lists of changed expert tensors")
        self._update(selected, records, replace)

    @_mutation
    def refresh_gmm1(self, indices=None, experts=None, *, source_layout=None, source=None):
        if isinstance(indices, str):
            source, indices = indices, experts
        elif experts is not None:
            indices = experts
        source = source_layout or source or "planar"
        self._check_bindings()
        if source not in ("planar", "fused"):
            raise ValueError("source must be planar or fused")
        indices = list(range(self.count) if indices is None else indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < self.count for i in indices)
        ):
            raise ValueError("invalid expert indices")
        if not self._valid:
            raise RuntimeError("restore the incomplete weight update before refreshing GMM1")
        self._valid = False
        self._pending.update(indices if self.list_mode else [0])
        for i in indices if self.list_mode else [0]:
            first, _, planar = self._records[i]
            if not self._dual:
                continue
            src, dst = (planar, first) if source == "planar" else (first, planar)
            updated = _reorder(src, source == "fused")
            self._retired.append(updated)
            for d, value in zip(dst, updated):
                _copy_prepared(d, value)
        self.commit_prepared_update()


def prepare_weights_for_mega_moe(l1_weights, l2_weights, *, shared=False, input_layout="nd", planar_l1=None):
    return PreparedWeights(l1_weights, l2_weights, shared, input_layout, planar_l1)
