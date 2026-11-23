"""V2 layout routing and production DMA/VF CPU oracle; no NPU performance claims."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("bits", [32, 64])
def test_production_simt_layout_dma_and_counts(tmp_path, bits):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    executable = tmp_path / ("layout.exe" if os.name == "nt" else "layout")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            f"-DEP_NUM_TOPK_IDX_BITS={bits}",
            "-I",
            str(ROOT / "csrc/direct"),
            str(Path(__file__).with_name("simt_layout_oracle.cpp")),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    try:
        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=120)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked oracle execution (4551)")
        raise
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SIMT layout CPU oracle passed" in result.stdout


@pytest.mark.parametrize("num_sms", [0, 8, 16, 32, 48, 64])
@pytest.mark.parametrize("mode", ["unset", "0", "1", "bad", "fused", "old-library", "cached"])
def test_v2_layout_inherits_dispatch_cores_and_keeps_baseline(num_sms, mode):
    from tests.api.test_clos_compatibility import run_host

    resolved = num_sms or 32
    cached = mode == "cached"
    flag = "1" if mode in ("1", "fused", "old-library", "cached") else mode
    error = "ValueError" if mode in ("bad", "fused") else "RuntimeError" if mode == "old-library" else "Submitted"
    run_host(f"""
        import os
        from contextlib import nullcontext
        from deep_ep.backends.direct import elastic as m
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        class Submitted(Exception): pass
        class Tensor:
            def __init__(self, shape): self.shape, self.device, self.dtype = shape, 'npu', 'torch.bfloat16'
            def size(self, dim): return self.shape[dim]
            def element_size(self): return 2
        for key in list(os.environ):
            if key.startswith('DEEPEP_CLOS_'): del os.environ[key]
        if {flag!r} != 'unset': os.environ['DEEPEP_CLOS_V2_SIMT_LAYOUT'] = {flag!r}
        os.environ['DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY'] = '{int(mode == "fused")}'
        for name in ('_validate_dispatch_scope', '_validate_dispatch_tensors'):
            setattr(m, name, lambda *a: None)
        m._get_allocation_stream_id = lambda _: 0
        m._try_record_event = lambda _: None
        m._current_npu_stream = lambda: 'caller'
        m._use_stream_after_event = lambda *a: nullcontext()
        m._use_comm_stream = lambda *a: nullcontext()
        m.load_native_module = lambda: SimpleNamespace(v2_simt_layout_version={int(mode != "old-library")})
        x, topk, mask = Tensor((129,2816)), Tensor((129,8)), Tensor((129,8))
        calls = []
        def layout(*args):
            calls.append(('old', args))
            return Tensor((8,)), None, Tensor((256,)), mask
        def simt(*args):
            calls.append(('simt', args))
            return Tensor((8,)), None, Tensor((256,)), mask
        def submit(*args):
            assert args[12] == {resolved}
            raise Submitted
        policy = resolve_udma_policy(True, 8, 'udma', None, None, None, None, None)
        obj = SimpleNamespace(_ensure_alive=lambda: None, _validate_handle=lambda _: None,
            use_fp8_dispatch=False, num_max_tokens_per_rank=129, num_ranks=8, hidden=2816, num_topk=8,
            get_buffer_size_hint=lambda *a: 0, group=None, allow_hybrid_mode=True,
            num_bytes=1, num_device_aiv=64, _udma_auto_policy=policy, transport='udma',
            _communication_stream='comm', _use_single_source_combine=False,
            _native=SimpleNamespace(get_dispatch_layout=layout, get_dispatch_layout_v2_simt=simt,
                intranode_dispatch_with_plan=submit, intranode_dispatch_with_handle=submit))
        handle = SimpleNamespace(num_sms={resolved}, num_experts=256, num_max_tokens_per_rank=129,
            expert_alignment=1, do_expand=False, topk_idx=topk, num_recv_tokens=0,
            _dispatch_layout=(object(),object(),None,None,mask,None), _dispatch_plan=object())
        try:
            m.ElasticBuffer.dispatch(obj, x, handle=handle if {cached} else None,
                topk_idx=None if {cached} else topk, num_experts=256, num_sms={num_sms}, do_expand=False)
        except {error}: pass
        else: raise AssertionError('expected {error}')
        if {mode!r} == '1': assert calls == [('simt', (topk, 256, {resolved}, 0))]
        elif {mode!r} in ('unset', '0'): assert calls == [('old', (topk, 256, 0))]
        else: assert calls == []
    """)
