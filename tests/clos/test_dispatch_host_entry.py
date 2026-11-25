"""Fresh dispatch Host entry routing and unchanged result/ownership contracts."""

import pytest
import os
from pathlib import Path
import shutil
import subprocess

from tests.api.test_clos_compatibility import run_host


@pytest.mark.parametrize("entry", ["default", "0", "old-wheel", "bad"])
@pytest.mark.parametrize("simt", [False, True])
@pytest.mark.parametrize("allocation", [False, True])
@pytest.mark.parametrize("weights", [False, True])
def test_combined_entry_preserves_layout_handle_outputs_and_allocation(entry, simt, allocation, weights):
    run_host(f"""
        import os
        from contextlib import nullcontext
        from deep_ep.backends.direct import elastic as m
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        for key in list(os.environ):
            if key.startswith('DEEPEP_CLOS_'): del os.environ[key]
        os.environ['DEEPEP_CLOS_V2_SIMT_LAYOUT'] = '{int(simt)}'
        if {entry!r} in ('0','bad'): os.environ['DEEPEP_CLOS_DISPATCH_HOST_ENTRY'] = {entry!r}
        m.load_native_module = lambda: SimpleNamespace(
            dispatch_host_entry_version={int(entry != "old-wheel")},v2_simt_layout_version=1)
        for name in ('_validate_dispatch_scope', '_validate_dispatch_tensors'):
            setattr(m, name, lambda *a: None)
        m._get_allocation_stream_id = lambda value: None if value else 7
        m._try_record_event = lambda _: None
        m._use_stream_after_event = lambda *a: nullcontext()
        m._record_recv_outputs_on_current_stream = lambda *a, **kw: None
        class Tensor:
            dtype = 'torch.bfloat16'
            def __init__(self, shape): self.shape = shape
            def size(self, dim): return self.shape[dim]
            def element_size(self): return 2
            def clone(self): return Tensor(self.shape)
        x, idx, weight = Tensor((129,2816)), Tensor((129,8)), Tensor((129,8))
        recv, recv_idx, recv_weight = Tensor((65,2816)), Tensor((65,8)), Tensor((65,8))
        rank, channel, incoming = Tensor((8,8)), Tensor((8,32)), Tensor((8,32))
        src, head, plan, mask = Tensor((65,)), Tensor((129,8)), Tensor((8,32,8)), Tensor((129,8))
        public = [Tensor((1,)) for _ in range(5)]
        calls = []
        result = (recv,None,recv_idx,recv_weight if {weights} else None,[1]*32,
                  rank,channel,incoming,src,head,plan)
        def combined(*args):
            calls.append(args)
            assert args == (x,None,idx,weight if {weights} else None,256,1,0,32,
                    6,256,None if {allocation} else 7,129,1,32,False,
                policy.for_launch(32,5632,'dispatch',bf16=True,expanded=False).native_plan(),{simt}), repr(args)
            assert {entry!r} == 'default'
            return (result, public, mask)
        def layout(*args):
            assert {entry!r} in ('0','old-wheel')
            calls.append('layout')
            return Tensor((8,)), None, Tensor((256,)), mask
        def legacy(*args):
            assert {entry!r} in ('0','old-wheel')
            assert args[5] is mask and args[12] == 32
            calls.append('dispatch')
            return result, public
        policy = resolve_udma_policy(True,8,'udma',None,None,None,None,None)
        finish = []
        def complete(*args, **kwargs):
            finish.append((args,kwargs)); return 'done'
        obj = SimpleNamespace(_ensure_alive=lambda: None, use_fp8_dispatch=False,
            num_max_tokens_per_rank=129,num_ranks=8,hidden=2816,num_topk=8,
            get_buffer_size_hint=lambda *a: 0,group=None,allow_hybrid_mode=True,
            num_bytes=1,num_device_aiv=64,_udma_auto_policy=policy,transport='udma',
            _communication_stream='comm',_use_single_source_combine=False,
            _handle_token=object(),_finish_comm_event=complete,
            _native=SimpleNamespace(intranode_dispatch_from_topk=combined,
                get_dispatch_layout=layout,get_dispatch_layout_v2_simt=layout,
                intranode_dispatch_with_handle=legacy))
        try:
            out = m.ElasticBuffer.dispatch(obj,x,topk_idx=idx,topk_weights=weight if {weights} else None,
                num_experts=256,num_sms=32,allocate_on_comm_stream={allocation},
                async_with_compute_stream=True)
        except ValueError as exc:
            assert {entry!r} == 'bad' and 'DISPATCH_HOST_ENTRY' in str(exc) and not calls
            raise SystemExit(0)
        assert {entry!r} != 'bad'
        assert len(calls)==1 if {entry!r} == 'default' else calls == ['layout','dispatch']
        assert out[:3] == (recv,recv_idx,recv_weight if {weights} else None)
        handle = out[3]
        assert handle._dispatch_layout == (rank,channel,incoming,src,mask,head)
        assert handle._dispatch_plan is plan and handle._buffer_owner is obj
        assert handle.topk_idx is not idx and handle.topk_idx.shape == idx.shape
        assert handle.num_recv_tokens_per_expert_list == [1]*32
        assert handle.psum_num_recv_tokens_per_scaleup_rank is public[0]
        assert out[4]=='done' and finish[0][0][1][-1] is handle
    """)


def test_native_composed_entry_order_forwarding_and_core_rejection(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for Host entry oracle")
    root = Path(__file__).resolve().parents[2]
    production = (root / "csrc/direct/runtime/buffer_runtime.cpp").read_text(encoding="utf-8")
    start = production.index("BufferRuntime::FreshDispatchHostResult BufferRuntime::IntranodeDispatchFromTopk(")
    body = production[start : production.index("\n}\n", start) + 3]
    source = tmp_path / "host_entry.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>
#define TORCH_CHECK(test, ...) if (!(test)) throw std::runtime_error("invalid cores")
namespace torch { struct Tensor { int id; }; }
namespace c10 { template<class T> using optional = std::optional<T>; constexpr auto nullopt = std::nullopt; }
struct BufferRuntime {
    int num_device_aiv_ = 64;
    bool expected_simt;
    bool used_simt = false;
    c10::optional<int64_t> expected_allocation;
    std::vector<int> order;
    using Layout = std::tuple<torch::Tensor,c10::optional<torch::Tensor>,torch::Tensor,torch::Tensor>;
    using FreshDispatchHostResult = std::tuple<std::tuple<torch::Tensor>,std::vector<torch::Tensor>,torch::Tensor>;
    Layout GetDispatchLayout(torch::Tensor idx,int64_t experts,c10::optional<int64_t> allocation) {
        assert(idx.id==3 && experts==256 && allocation==expected_allocation);
        order.push_back(1);
        return {{11},c10::nullopt,{12},{13}};
    }
    Layout GetDispatchLayoutV2Simt(torch::Tensor idx,int64_t experts,int64_t cores,c10::optional<int64_t> allocation) {
        assert(expected_simt && cores==32);
        used_simt = true;
        return GetDispatchLayout(idx,experts,allocation);
    }
    template<class... Args> auto IntranodeDispatchWithHandle(Args... args) {
        assert(order == std::vector<int>{1});
        order.push_back(2);
        auto a=std::make_tuple(args...);
        assert(std::get<0>(a).id==1 && std::get<1>(a)->id==2 && std::get<2>(a).id==3 && std::get<3>(a)->id==4);
        assert(std::get<4>(a).id==11 && std::get<5>(a).id==13 && std::get<6>(a).id==12);
        assert(std::get<7>(a)==0 && std::get<10>(a)==2 && std::get<11>(a)==100);
        assert(std::get<12>(a)==32 && std::get<13>(a)==6 && std::get<14>(a)==256 && !std::get<15>(a));
        assert(std::get<16>(a)==expected_allocation && std::get<17>(a)==129);
        assert(std::get<18>(a)==2 && std::get<19>(a)==8 && std::get<21>(a));
        assert(std::get<22>(a)==std::vector<int64_t>({1,2,3}));
        return std::make_tuple(std::make_tuple(torch::Tensor{21}),std::vector<torch::Tensor>{{22}});
    }
    FreshDispatchHostResult IntranodeDispatchFromTopk(
        torch::Tensor,c10::optional<torch::Tensor>,torch::Tensor,c10::optional<torch::Tensor>,
        int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,c10::optional<int64_t>,
        c10::optional<int64_t>,int64_t,int64_t,bool,const std::vector<int64_t>&,bool);
};
"""
        + body
        + r"""
int main() {
    for (bool simt : {false,true}) for (bool comm : {false,true}) {
        BufferRuntime runtime;
        runtime.expected_simt=simt;
        runtime.expected_allocation=comm ? c10::nullopt : c10::optional<int64_t>{7};
        auto result=runtime.IntranodeDispatchFromTopk({1},torch::Tensor{2},{3},torch::Tensor{4},
            256,2,100,32,6,256,runtime.expected_allocation,129,2,8,true,{1,2,3},simt);
        assert(std::get<0>(std::get<0>(result)).id==21 && std::get<1>(result)[0].id==22 && std::get<2>(result).id==13);
        assert(runtime.order==std::vector<int>({1,2}));
        assert(runtime.used_simt==simt);
        runtime.order.clear();
        try {
            runtime.IntranodeDispatchFromTopk({1},torch::Tensor{2},{3},torch::Tensor{4},
                256,2,100,65,6,256,runtime.expected_allocation,129,2,8,true,{1,2,3},simt);
            assert(false);
        } catch (const std::runtime_error&) { assert(runtime.order.empty()); }
    }
}
""",
        encoding="utf-8",
    )
    target = tmp_path / ("host_entry.exe" if os.name == "nt" else "host_entry")
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(target)], check=True, timeout=120)
    try:
        subprocess.run([str(target)], check=True, timeout=30)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("Windows application control blocked compiled Host entry oracle")
        raise
