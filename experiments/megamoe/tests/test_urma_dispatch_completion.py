"""Real diagnostic observer/timer with CPU local-flag mocks; no hardware latency claim."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("coarse", [False, True])
def test_peer_receive_observer(tmp_path, enabled, coarse):
    cxx = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(cxx[0]):
        pytest.skip("C++ compiler required")
    root = Path(__file__).resolve().parents[3]
    inc = root / "experiments/megamoe/csrc/kernels/ubx/urma"
    (tmp_path / "kernel_operator.h").write_text(
        r"""
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
namespace AscendC {
constexpr int AIC=0,AIV=1;
inline uint64_t cycles=100,flushes=0,reads=0;
inline uint64_t GetSystemCycle(){return cycles++;}
inline uint32_t GetBlockIdx(){return 0;}
inline uint32_t GetTaskRation(){return 2;}
inline uint32_t GetSubBlockIdx(){return 1;}
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T>struct GlobalTensor {void SetGlobalBuffer(T*){}};
template<class T,CacheLine,DcciDst>void DataCacheCleanAndInvalid(GlobalTensor<T>&){++flushes;}
template<class T>T ReadGmByPassDCache(T* p){++reads;return *p;}
}
constexpr int g_coreType=AscendC::AIV;
""",
        encoding="utf-8",
    )
    cpp = r"""
#include "dispatch_completion.hpp"
#include <cassert>
#include <vector>
using namespace MegaMoeUrma;
int main(){
    const uint32_t capacity=8,stride=kTraceHeaderElements+capacity*kTraceRecordElements;
    std::vector<uint64_t> trace(kTimerCores*stride,0),aggregate(kTimerElements,0),flags(16*128/8,0);
    Timer timer;timer.Init(reinterpret_cast<uint64_t>(aggregate.data()),reinterpret_cast<uint64_t>(trace.data()),capacity,COARSE);
    timer.Begin(Dispatch);timer.End(Dispatch,1024,true);
    Arguments a{};a.config.cvRows=256;a.config.maxPeerChunks=4;a.config.hidden=3584;a.config.epoch=7;
    a.symmetric=reinterpret_cast<uint64_t>(flags.data());
    DispatchReceiveObserver observer;observer.Init(timer);observer.Arm(a,2,600);
    const auto firstClock=AscendC::cycles;
    flags[(1*4+2)*16]=7; // another source is NOT completion
    observer.Poll();
    flags[(2*4+2)*16]=6; // old epoch is NOT completion
    observer.Poll();
    flags[(2*4+0)*16]=flags[(2*4+1)*16]=7;observer.Poll();
    assert(AscendC::cycles==firstClock);
    flags[(2*4+2)*16]=7;observer.Poll();
#ifdef DEEPEP_MEGAMOE_TIMER
    assert(AscendC::reads==4 && AscendC::cycles==firstClock+1);
    for(auto value:trace)assert(value==0); // deferred trace store
    observer.Poll();assert(AscendC::reads==4);
    timer.WriteTrace(TraceEvent::Reduce,firstClock-1,firstClock+10,TraceMeta{});
    timer.FinishTrace();
    const auto* point=trace.data()+2*stride+kTraceHeaderElements+kTraceRecordElements;
    assert(point[2]==65 && point[0]==firstClock && point[1]==firstClock && point[3]==2 && point[4]==2);
    assert(point[8]==600 && point[12]==600*(3584+112));
    if(!COARSE){
        auto* row=aggregate.data()+(2*kTimerPhases+Dispatch)*kTimerRecordElements;
        assert(row[5]==2 && row[7]==firstClock && row[8]==600*(3584+112) && row[9]==600);
        assert(row[3]==1024); // outgoing numerator preserved
    }
    std::vector<uint64_t> emptyAggregate(kTimerElements,0),emptyTrace(kTimerCores*stride,0);
    Timer emptyTimer;emptyTimer.Init(reinterpret_cast<uint64_t>(emptyAggregate.data()),reinterpret_cast<uint64_t>(emptyTrace.data()),capacity,COARSE);
    DispatchReceiveObserver empty;empty.Init(emptyTimer);
    const auto reads=AscendC::reads;empty.Arm(a,1,0);empty.Poll();assert(AscendC::reads==reads);
    emptyTimer.FinishTrace();
    const auto* emptyPoint=emptyTrace.data()+2*stride+kTraceHeaderElements;
    assert(emptyPoint[2]==65&&emptyPoint[8]==0&&emptyPoint[12]==0);
    if(!COARSE){auto* row=emptyAggregate.data()+(2*kTimerPhases+Dispatch)*16;assert(row[5]==2&&row[6]==1&&row[7]==0&&row[8]==0);}
#else
    assert(AscendC::reads==0 && AscendC::cycles==100);
    for(auto value:trace)assert(value==0);
    for(auto value:aggregate)assert(value==0);
#endif
    Timer off;off.Init(0);DispatchReceiveObserver disabled;disabled.Init(off);
    const auto disabledReads=AscendC::reads;const auto cycles=AscendC::cycles;
    disabled.Arm(a,1,128);disabled.Poll();
    assert(AscendC::reads==disabledReads && AscendC::cycles==cycles);
}
"""
    src = tmp_path / "test.cpp"
    src.write_text(cpp, encoding="utf-8")
    exe = tmp_path / ("test.exe" if os.name == "nt" else "test")
    defines = ["-DCOARSE=" + str(int(coarse))] + (["-DDEEPEP_MEGAMOE_TIMER"] if enabled else [])
    subprocess.run(
        [*cxx, "-std=c++17", "-UNDEBUG", *defines, "-I", str(tmp_path), "-I", str(inc), str(src), "-o", str(exe)],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_deferred_completion_point_can_overlap_reduce_but_not_hide_bad_scopes():
    import numpy as np
    from deep_ep_experimental.megamoe import urma_trace as t

    capacity = 4
    stride = t.HEADER_WORDS + capacity * t.RECORD_WORDS
    raw = np.zeros((t.ENGINE_COUNT, stride), dtype=np.uint64)
    for i in range(t.ENGINE_COUNT):
        raw[i, :5] = [1, capacity, 0, 0, i]
    raw[2, 2] = 2
    for ordinal, (start, end, event) in enumerate([(100, 200, 25), (150, 150, 65)]):
        row = raw[2, t.HEADER_WORDS + ordinal * t.RECORD_WORDS : t.HEADER_WORDS + (ordinal + 1) * t.RECORD_WORDS]
        row[:3] = [start, end, event]
        row[3:12] = t.UNKNOWN
        row[3] = 1
    doc = t.decode_trace(raw.reshape(-1), rank=0, clock_mhz=1000)
    assert any(e.get("args", {}).get("event_id") == 65 for e in doc["traceEvents"])
    raw[2, t.HEADER_WORDS + t.RECORD_WORDS + 1] = 151
    with pytest.raises(ValueError, match="overlapping|out-of-order"):
        t.decode_trace(raw.reshape(-1), rank=0, clock_mhz=1000)
