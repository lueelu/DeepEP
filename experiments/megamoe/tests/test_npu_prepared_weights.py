"""Single-device EPLB checks against real NZ storage and a captured table reader."""

import pytest


def test_npu_eplb_layout_refresh_and_graph_table_replacement():
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    def raw(n, k, seed):
        gen = torch.Generator().manual_seed(seed)

        def data(shape):
            return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen).npu()

        return data((n, k // 2)), data((n, k // 64, 2))

    def sliced(n, k, seed):
        w, s = (torch.stack(ts) for ts in zip(raw(n, k, seed), raw(n, k, seed + 1)))
        assert w[1:2].is_contiguous() and w[1:2].storage_offset() > 0
        return [(w[0], s[0]), (w[1:2], s[1:2])]

    first = sliced(512, 1024, 1)
    second = sliced(1024, 256, 3)
    owner = prepare_weights_for_mega_moe(first, second)
    planar, _ = owner.expert_views(layout="planar")
    fused, _ = owner.expert_views(layout="fused")
    addresses = [t.data_ptr() for pair in (planar, fused) for group in pair for t in group]

    def read_weight(t):
        assert torch_npu.get_npu_format(t) in (29, 50)
        return torch_npu.npu_format_cast(
            t, 2, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        ).cpu()

    views = owner.expert_views(1)
    expected_w, expected_s = (t[0].cpu() for t in first[1])
    for actual, expected in zip(views["planar_l1"], (expected_w, expected_s)):
        value = read_weight(actual) if actual.dtype == torch.uint8 else actual.view(torch.uint8).cpu()
        assert torch.equal(value, expected)
    for actual, expected in zip(views["l1"], (expected_w, expected_s)):
        gate0, gate1, up0, up1 = expected.split(128, dim=0)
        value = read_weight(actual) if actual.dtype == torch.uint8 else actual.view(torch.uint8).cpu()
        assert torch.equal(value, torch.cat((gate0, up0, gate1, up1)))
    assert torch.equal(read_weight(views["l2"][0]), second[1][0][0].cpu())
    assert torch.equal(views["l2"][1].view(torch.uint8).cpu(), second[1][1][0].cpu())

    # Modify both the packed FP4 bytes and E8M0 bytes through exported zero-copy views.
    for group in planar:
        group[1].copy_(group[0])
    owner.refresh_gmm1([1], source_layout="planar")
    for i, t in enumerate((read_weight(fused[0][1]), fused[1][1].view(torch.uint8).cpu())):
        gate0, gate1, up0, up1 = first[0][i].cpu().split(128, dim=0)
        assert torch.equal(t, torch.cat((gate0, up0, gate1, up1)))
    owner.refresh_gmm1([1], source_layout="fused")
    assert torch.equal(read_weight(planar[0][1]), first[0][0].cpu())
    assert torch.equal(planar[1][1].view(torch.uint8).cpu(), first[0][1].cpu())
    assert addresses == [t.data_ptr() for pair in (planar, fused) for group in pair for t in group]

    tables = [owner.select(layout)[2] for layout in (True, False)]
    before = [t.cpu() for t in tables]
    saved = [torch.empty_like(t) for t in tables]
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for dst, src in zip(saved, tables):
            dst.copy_(src)
    owner.update_experts([1], [raw(512, 1024, 5)], [raw(1024, 256, 6)], replace=True)
    graph.replay()
    torch.npu.synchronize()
    for idx, (src, dst, old) in enumerate(zip(tables, saved, before)):
        assert owner.select(idx == 0)[2].data_ptr() == src.data_ptr()
        assert torch.equal(dst.cpu(), src.cpu())
        assert torch.equal(dst.cpu()[:, 0], old[:, 0])
        assert torch.all(dst.cpu()[:, 1] != old[:, 1])
    owner.release_retired()


def test_npu_import_singleton_axes_and_update_physical_nz():
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    def cast(t, fmt):
        return torch_npu.npu_format_cast(
            t, fmt, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )

    def projection(n, k, value, explicit):
        raw = torch.full((n, k // 2), value, device="npu", dtype=torch.uint8)
        scale = torch.full((n, k // 64, 2), 127, device="npu", dtype=torch.uint8)
        return cast(raw.unsqueeze(0) if explicit else raw, 29), scale if explicit else scale.unsqueeze(0)

    first = [projection(512, 1024, i + 1, bool(i)) for i in range(2)]
    second = [projection(1024, 256, i + 3, bool(i)) for i in range(2)]
    owner = prepare_weights_for_mega_moe(first, second, input_layout="planar_nz")
    tables = [owner.select(f)[2] for f in (False, True)]
    before = [t.cpu() for t in tables]
    for replace in (False, True):
        value = 7 if replace else 5
        owner.update_experts(
            indices=[0],
            l1=[projection(512, 1024, value, True)],
            l2=[projection(1024, 256, value, True)],
            input_layout="planar_nz",
            replace=replace,
        )
        owner.refresh_gmm1([0], source_layout="fused")
        current = owner.expert_views(0)
        for key in ("l1", "planar_l1", "l2"):
            w = current[key][0]
            assert torch.all(cast(getattr(w, "_megamoe_nz_owner", w), 2).cpu() == value)
        for fused, table, old in zip((False, True), tables, before):
            assert owner.select(fused)[2].data_ptr() == table.data_ptr()
            now = table.cpu()
            assert torch.equal(old[:, 1], now[:, 1])
            assert bool(torch.all(old[:, 0] != now[:, 0])) if replace else torch.equal(old[:, 0], now[:, 0])
    owner.release_retired()


@pytest.mark.parametrize("ranks", [(2, 2, 2), (3, 3, 3), (3, 3, 2)])
def test_cann_consumes_public_physical_expert_tensors(ranks):
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    def cast(t, fmt):
        return torch_npu.npu_format_cast(
            t, fmt, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )

    def interleave(t):
        gate0, gate1, up0, up1 = t.chunk(4, dim=0)
        return torch.cat((gate0, up0, gate1, up1))

    pairs = {key: [] for key in ("planar", "fused", "second")}
    expected = {key: [] for key in pairs}
    for i, rank in enumerate(ranks):
        gen = torch.Generator().manual_seed(301 + i)
        for key, n, k in (("planar", 512, 1024), ("second", 1024, 256)):
            w = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
            s = torch.randint(120, 129, (n, k // 64, 2), generator=gen, dtype=torch.uint8)
            expected[key].append((w, s))
        expected["fused"].append(tuple(interleave(t) for t in expected["planar"][i]))
        for key in pairs:
            w, s = expected[key][i]
            pairs[key].append((cast((w.unsqueeze(0) if rank == 3 else w).npu(), 29), s.npu()))
    owner = prepare_weights_for_mega_moe(
        pairs["fused"], pairs["second"], input_layout="prepared", planar_l1=pairs["planar"]
    )
    tables = [owner.select(f)[2] for f in (False, True)]
    before = [t.cpu() for t in tables]

    def check():
        for layout in ("planar", "fused"):
            for key, (weights, scales) in zip((layout, "second"), owner.expert_views(layout=layout)):
                for i, (weight, scale) in enumerate(zip(weights, scales)):
                    original = pairs[key][i][0]
                    assert weight is original and weight.ndim == ranks[i] and weight.dtype == torch.uint8
                    assert weight.data_ptr() == original.untyped_storage().data_ptr()
                    assert torch_npu.get_npu_format(weight) in (29, 50)
                    raw, sf = expected[key][i]
                    assert torch.equal(cast(weight, 2).cpu().reshape_as(raw), raw)
                    assert torch.equal(scale.view(torch.uint8).cpu(), sf)
        for f, t, old in zip((False, True), tables, before):
            assert owner.select(f)[2].data_ptr() == t.data_ptr() and torch.equal(t.cpu(), old)

    check()
    for pair in owner.expert_views(layout="planar"):
        for group in pair:
            group[1].copy_(group[0])
    owner.refresh_gmm1([1], source_layout="planar")
    for values in expected.values():
        values[1] = values[0]
    check()
