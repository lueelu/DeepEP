"""Public cached-dispatch defaults and overrides, without a device runtime."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tests.api.test_clos_compatibility import run_host


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("option", [None, "default", "CACHED_WEIGHT_PACK", "CACHED_OMIT_TOPK_OUTPUT"])
def test_cached_metadata_keeps_route_ownership_and_expansion_consumers(expanded, option):
    run_host(f"""
        import os
        from contextlib import nullcontext
        from deep_ep.backends.direct import elastic as m
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        class Submitted(Exception): pass
        class Tensor:
            def __init__(self, shape, dtype='torch.bfloat16', device='npu'):
                self.shape,self.dtype,self.device=shape,dtype,device
            def size(self, dim): return self.shape[dim]
            def element_size(self): return 2
        torch.empty=lambda shape,**kw:Tensor(shape,**kw)
        for key in ('CACHED_WEIGHT_PACK','CACHED_OMIT_TOPK_OUTPUT'):
            if {option!r} == 'default':
                os.environ.pop('DEEPEP_CLOS_'+key, None)
            else:
                os.environ['DEEPEP_CLOS_'+key] = str(int(key=={option!r}))
        for name in ('_validate_dispatch_scope','_validate_dispatch_tensors'):
            setattr(m,name,lambda *a:None)
        m._get_allocation_stream_id=lambda _:0
        m._try_record_event=lambda _:None
        m._current_npu_stream=lambda:'caller'
        m._use_stream_after_event=lambda *a:nullcontext()
        m._use_comm_stream=lambda *a:nullcontext()
        m.load_native_module=lambda:SimpleNamespace(dispatch_metadata_output_version=2)
        x,idx,weights=Tensor((65,2816)),Tensor((65,8)),Tensor((65,8))
        head,plan=object(),object()
        def submit(*args):
            assert args[2] is idx and args[3] is weights
            if {option is not None}:
                assert args[-2] is head and args[-1] is {not expanded}
                assert args[-4] is plan
            else:
                assert args[-2] is plan
            raise Submitted
        policy=resolve_udma_policy(True,8,'udma',None,None,None,None,None)
        obj=SimpleNamespace(_ensure_alive=lambda:None,_validate_handle=lambda _:None,
            use_fp8_dispatch=False,num_max_tokens_per_rank=65,num_ranks=8,hidden=2816,num_topk=8,
            get_buffer_size_hint=lambda *a:0,group=None,allow_hybrid_mode=True,
            num_bytes=1,num_device_aiv=64,_udma_auto_policy=policy,transport='udma',
            _communication_stream='comm',_use_single_source_combine=False,
            _native=SimpleNamespace(intranode_dispatch_with_plan=submit))
        handle=SimpleNamespace(num_sms=8,num_experts=256,num_max_tokens_per_rank=65,
            expert_alignment=1,do_expand={expanded},topk_idx=idx,num_recv_tokens=0,
            _dispatch_layout=(object(),object(),None,None,Tensor((65,8)),head),_dispatch_plan=plan)
        try:
            m.ElasticBuffer.dispatch(obj,x,handle=handle,topk_weights=weights,num_sms=8,do_expand={expanded})
        except Submitted: pass
        else: raise AssertionError('missing native dispatch')
    """)


def test_native_packing_defaults_preserve_unsupported_paths(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    root = Path(__file__).resolve().parents[2]
    source = (root / "csrc/direct/ops/dispatch.cpp").read_text()
    begin = source.index("    const bool default_channel_metadata =")
    selection = source[begin : source.index("    if (cached_weight_pack)", begin)]
    program = r"""
#include "common/dispatch_channel_metadata.h"
struct Optional { bool present; constexpr bool has_value() const { return present; } };
constexpr unsigned choose(bool use_udma=true, bool token=false, bool capturing=false,
                          unsigned world_size=32, unsigned num_topk=8, bool cached_mode=true,
                          bool weight=true, bool plan=true, bool head=true, int override_value=-1) {
    struct Shmem { bool token; constexpr bool TokenSchedule() const { return token; } } shmem{token};
    const Optional topk_weights{weight}, cached_dispatch_plan{plan}, cached_send_head{head};
    const auto enabled_option = [=](const char*, bool fallback) {
        return override_value < 0 ? fallback : override_value == 1;
    };
    // SELECTION
    return unsigned(default_channel_metadata) | (unsigned(cached_weight_pack) << 1);
}
static_assert(choose() == 3);
static_assert(choose(true,false,false,16) == 3);
static_assert(choose(true,false,false,8) == 3);
static_assert(choose(true,false,false,256,128) == 3);
static_assert(choose(false) == 0);
static_assert(choose(true,true) == 0);
static_assert(choose(true,false,true) == 0);
static_assert(choose(true,false,false,512) == 0);
static_assert(choose(true,false,false,32,0) == 0);
static_assert(choose(true,false,false,32,129) == 0);
static_assert(choose(true,false,false,32,8,false) == 1);
static_assert(choose(true,false,false,32,8,true,false) == 1);
static_assert(choose(true,false,false,32,8,true,true,false) == 1);
static_assert(choose(true,false,false,32,8,true,true,true,false) == 1);
static_assert(choose(true,false,false,32,8,true,true,true,true,0) == 1);
"""
    path = tmp_path / "metadata_defaults.cpp"
    path.write_text(program.replace("// SELECTION", selection))
    result = subprocess.run(
        [compiler, "-std=c++17", "-I", str(root / "csrc/direct"), "-c", str(path), "-o", str(tmp_path / "defaults.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
