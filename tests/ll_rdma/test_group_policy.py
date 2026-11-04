"""Host tiling policy and write-only host-boundary invariants (CPU only)."""
from pathlib import Path
import os
import shutil
import subprocess
from types import SimpleNamespace
from contextlib import contextmanager
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_tiling_keys_and_host_boundaries(tmp_path):
    compiler = shutil.which("g++")
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if compiler is None and not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG or install g++ for the C++ policy model")
    command = [compiler] if compiler else [zig, "c++"]
    cpp = tmp_path / "policy.cpp"
    cpp.write_text(r'''
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h"
#include "csrc/kernels/combine/low_latency/combine_layout.h"
#include "csrc/kernels/combine/low_latency/ep_memory_server_dedup_combine_tiling.h"
using namespace DispatchDedup;
int main() {
    static_assert(offsetof(EpServerDedupTilingData, tilingKey)==21);
    static_assert(offsetof(CombineDedup::EpServerDedupCombineTilingData, tilingKey)==15);
    static_assert(sizeof(EpServerDedupTilingData)==32);
    static_assert(sizeof(CombineDedup::EpServerDedupCombineTilingData)==16);
    for(unsigned p : {2U,4U,8U,16U,32U,64U,128U,256U}) {
        unsigned group=p<=16?4:16;
        assert(LlGroupForWorld(p)==group && LlTilingKeyForWorld(p)==group);
        assert(LlTilingKeyValid(p,group));
        assert(!LlTilingKeyValid(p,group==4?16:4) && !LlTilingKeyValid(p,0));
        for(unsigned rank=0;rank<p;++rank) for(unsigned peer=0;peer<p;++peer) {
            assert(LlSameHost(rank,peer)==(rank/16==peer/16));
            // Every partial member / relay / ready peer is local under both keys.
            if(rank/group==peer/group) assert(LlSameHost(rank,peer));
        }
        EpMemoryServerDedupLayout d; CombineDedup::Layout c;
        if(p<16) {
            assert(BuildDispatchServerDedupLayout(p,4,256,4,128,&d)!=0);
            assert(!CombineDedup::BuildLayout(p,4,256,4,128,c));
        }
    }
    assert(LlSameHost(15,0) && !LlSameHost(15,16) && LlSameHost(16,31));
    assert(!LlSameHost(0,1,1) && LlSameHost(0,1,2)); // real two-rank primitive layouts
}
''')
    exe = tmp_path / "policy.exe"
    # Quote-only lookup avoids shadowing libc++'s <version> with repo/version.
    subprocess.run([*command, "-std=c++17", "-UNDEBUG", "-iquote", str(ROOT),
                    str(cpp), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=60)


def test_diagnostic_metadata_uses_selected_group(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from tests.elastic import test_ep_low_latency as st
    from deep_ep import _native
    from deep_ep.utils import ll_stage_timer
    import torch.distributed as dist
    saved = []
    monkeypatch.setattr(dist, "get_world_size", lambda: 16)
    monkeypatch.setattr(_native, "require_native", lambda _: SimpleNamespace(low_latency_ranks_per_group=lambda p: 4 if p<=16 else 16))
    @contextmanager
    def capture(*a, **k):
        yield k['prepared']
    monkeypatch.setattr(ll_stage_timer, "capture", capture)
    monkeypatch.setattr(ll_stage_timer, 'StageTrace', lambda *a: SimpleNamespace(
        complete=lambda: None, save=lambda path, meta: saved.append(meta)))
    monkeypatch.setattr(st, 'enqueue_exp_prefix', lambda *a: None)
    monkeypatch.setattr(st, "completion_boundary", lambda *a: None)
    monkeypatch.setattr(st, "agree_check", lambda action, *a: action())
    monkeypatch.setattr(st, "case_config", lambda *a: {"B": 32})
    args = SimpleNamespace(run_directory=str(tmp_path), case_id=0, ll_timer_iters=1,
        ll_timer_mode="observe", ll_timer_frequency_hz=None, seed=0,
        warmup=0, num_iters=0, delay_rows=1, delay_cols=1, group_backend='hccl')
    event = SimpleNamespace(current_stream_wait=lambda: None)
    st.diagnostic_calls(object(), lambda: (event,), lambda output: None, args, "dispatch", 0, 256, "normal",
                        prelude_buffers=(object(), object()))
    assert saved[0]["world_size"] == 16 and saved[0]["config"]["logical_group_size"] == 4


def test_wrong_physical_rank_layout_rejected_in_plan(tmp_path):
    import os, sys
    env = dict(os.environ)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        env.pop(key, None)
    result = subprocess.run([sys.executable, str(ROOT / "tests/ll_rdma/run_ll.py"),
        "--backend", "rdma", "--devices", "0,1", "--validation",
        "--config", "tests/ll_rdma/target16.json", "--num-processes", "8", "--num-nodes", "2", "--plan"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode and "16 contiguous ranks per host" in result.stderr
