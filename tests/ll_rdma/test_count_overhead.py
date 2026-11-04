"""Contracts for count UB lifetimes, publication and prefix reuse."""
from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEVICE = ROOT / "csrc/kernels/dispatch/low_latency/device"


def _source(name):
    return (DEVICE / name).read_text(encoding="utf-8")


def _block(source, marker):
    """Return the actual braced production block after a unique marker."""
    start = source.index("{", source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError(f"Unclosed block: {marker}")


def test_count_prefix_is_loaded_only_by_its_output_owner():
    counts = _source("dispatch_server_dedup_counts.h")
    complete = _block(counts, "void ServerDedupCounts::CompleteCumsum()")
    owner = _block(complete, "if (receiver == 0U)")
    before_owner = complete[:complete.index("if (receiver == 0U)")]
    assert "WaitCumSumFlag();" in before_owner  # Every receiver still retires its ready row.
    assert "prefixBuf_" not in before_owner
    assert "DataCopyPad(prefix," not in before_owner
    assert "InitBuffer(prefixBuf_" in owner
    assert owner.index("DataCopyPad(prefix,") < owner.index("SyncFunc<HardEvent::MTE2_S>()")
    assert owner.index("SyncFunc<HardEvent::MTE2_S>()") < owner.index("prefix.GetValue(")
    assert "DataCopyPad(expertTokenNumsOutGMTensor_" in owner
    assert "LocalTensor<int32_t> Prefix()" not in counts
    receiver = _block(_source("dispatch_server_dedup_recv.h"), "void Process(")
    assert receiver.index("pipe_->Reset();") < receiver.index("DataCopyPad(prefix, prefixGlobal")
    assert receiver.index("ServerDedupReceiveSync<HardEvent::MTE2_S>()") < receiver.index("tasks_.Build(prefix")


def test_direct_count_records_live_until_the_existing_reuse_barrier():
    counts = _source("dispatch_server_dedup_counts.h")
    send = _block(counts, "void ServerDedupCounts::SendCnt()")
    assert "routeExchange_" not in counts
    remote = _block(send, "if (remoteOwner)")
    assert remote.index("CountRemoteRank(first,") < remote.index("LlTransport::DirectCountFrame")
    assert remote.index("LlTransport::DirectCountFrame") < remote.index("SyncFunc<HardEvent::S_V>()")
    assert send.index("LlTransport::DirectCountFrame") < send.index("CountByVectorRange(start, end, maskBytes)")
    assert "Reset(" not in send
    assert "tpipe_->InitBuffer(statusBuf_, ReservedBytes())" in send
    assert "kEpServerDedupCountUbBytes - ReservedBytes()" in send
    assert "return stagePool_;" in _block(counts, "ScratchPool()")
    assert "PipeBarrier<PIPE_ALL>();" in _block(counts, "void Finish()")
    kernel = (DEVICE.parent / "dispatch_server_dedup_kernel.h").read_text(encoding="utf-8")
    assert "sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes(), profile_, counts_.Retirement())" in kernel
    complete = kernel.index("counts_.CompleteCumsum();")
    consume = kernel.index("receiver_.Process(profile_);")
    assert "PipeBarrier<PIPE_ALL>();" in kernel[complete:consume]
    loaded = send.index("DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_")
    assert "SyncFunc<HardEvent::MTE2_V>();" in send[loaded:send.index("CountRemoteRank(first,")]


def test_prefix_subtotal_executes_production_scalar_scan(tmp_path):
    """Compile the production body with a delayed GatherMask dependency model."""
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for C++ CPU validation")
    body = _block(_source("dispatch_server_dedup_counts.h"), "void ServerDedupCounts::GatherSumRecvCnt(")
    assert "\n    Sum(" not in body
    assert body.index("GatherMask(") < body.index("SyncFunc<AscendC::HardEvent::V_S>()")
    assert body.index("SyncFunc<AscendC::HardEvent::V_S>()") < body.index("prefix.GetValue(")
    source = tmp_path / "count-prefix.cpp"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
namespace AscendC { enum class HardEvent { S_V, V_S }; }
using AscendC::HardEvent;
static float* pendingDst=nullptr;
static std::vector<float> pending;
template<class T> struct LocalTensor {
    T* ptr;
    T GetValue(uint32_t i) const {
        assert(!pendingDst);T result;std::memcpy(&result,ptr+i,sizeof(T));return result;
    }
    void SetValue(uint32_t i,T value) const {
        assert(!pendingDst);std::memcpy(ptr+i,&value,sizeof(T));
    }
    template<class U> LocalTensor<U> ReinterpretCast() const {return {reinterpret_cast<U*>(ptr)};}
};
struct GatherParams { uint16_t repeats,records,srcStride,dstStride; };
static void GatherMask(LocalTensor<float> out,LocalTensor<float> input,LocalTensor<uint32_t> pattern,
                       bool,uint32_t mask,GatherParams shape,uint64_t& elements) {
    assert(!pendingDst && mask==2U && pattern.GetValue(0)==2U);
    assert(shape.repeats==1U && shape.srcStride==1U && shape.dstStride==0U);
    pending.resize(shape.records);
    for(unsigned i=0;i<shape.records;++i) pending[i]=input.GetValue(i*8U+1U);
    pendingDst=out.ptr;elements=shape.records;
}
struct Fixture {
    LocalTensor<float> statusFp32Tensor_;
    uint32_t recStatusNumPerCore_=0,published=0;
    template<HardEvent event> void SyncFunc() {
        if constexpr(event==HardEvent::V_S) {
            assert(pendingDst);std::memcpy(pendingDst,pending.data(),pending.size()*sizeof(float));
            pendingDst=nullptr;pending.clear();
        } else assert(!pendingDst);
    }
    void PublishRecvCntSum(float value) {std::memcpy(&published,&value,sizeof(value));}
    void Run(LocalTensor<float>& gatherMaskOutTensor,LocalTensor<uint32_t>& gatherTmpTensor) {
''' + body + r'''
    }
};
static uint32_t Bits(float value) {uint32_t bits;std::memcpy(&bits,&value,4);return bits;}
int main() {
    unsigned cases=0;
    for(unsigned count: {1U,7U,8U,13U,16U,31U,64U,128U}) for(unsigned mode=0;mode<4;++mode) {
        std::vector<float> records(count*8U),prefix(count+8U);
        std::vector<uint32_t> expected(count),pattern(8U);
        uint32_t sum=0;
        for(unsigned i=0;i<count;++i) {
            uint32_t value=mode==0U?0U:mode==1U?16384U:mode==2U?(i%5U==0U?i+1U:0U):(i*127U+3U)%16385U;
            std::memcpy(records.data()+i*8U+1U,&value,4);sum+=value;expected[i]=sum;
            const uint32_t ready=0x3f800000U;std::memcpy(records.data()+i*8U,&ready,4);
        }
        const uint32_t guard=0x5a5a5a5aU;
        for(unsigned i=count;i<prefix.size();++i) std::memcpy(prefix.data()+i,&guard,4);
        Fixture fixture{{records.data()},count};
        LocalTensor<float> output{prefix.data()};LocalTensor<uint32_t> temporary{pattern.data()};
        fixture.Run(output,temporary);
        assert(fixture.published==sum && !pendingDst);
        for(unsigned i=0;i<count;++i) assert(Bits(prefix[i])==expected[i]);
        for(unsigned i=count;i<prefix.size();++i) assert(Bits(prefix[i])==guard);
        ++cases;
    }
    std::cout<<"PASS production count prefix: "<<cases<<" partitions, exact subtotal bits, delayed Vector completion, UB guards\n";
}
''', encoding="utf-8")
    executable = tmp_path / "count-prefix.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", str(source), "-o", str(executable)],
                   check=True, timeout=180)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)
    assert "PASS production count prefix: 32 partitions" in result.stdout


def test_shared_prefix_publication_and_consumers_keep_the_ready_dependency():
    counts = _source("dispatch_server_dedup_counts.h")
    publish = _block(counts, "void ServerDedupCounts::CalRecvAndSetFlag()")
    write = "DataCopyPad(workspaceGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams)"
    assert publish.count(write) == 1
    assert "workspaceGlobal[index" not in publish
    assert publish.index(write) < publish.index("SyncFunc<AscendC::HardEvent::MTE3_V>()")
    assert publish.index("PipeBarrier<PIPE_MTE3>()") < publish.index("layout_.cumsumFlagOffset")
    owner = _block(_block(counts, "void ServerDedupCounts::CompleteCumsum()"), "if (receiver == 0U)")
    assert "DataCopyPad(prefix, workspace, copy, pad);" in owner
    receive = _block(_source("dispatch_server_dedup_recv.h"), "void Process(")
    assert "counts.cntOffset + counts.prefixOffset" in receive
    assert "counts.prefixStride" not in receive
    # Subtotals use one writer-owned slot per receiver, consumers read one contiguous
    # burst, and the slot is reclaimed only after the following shared barrier.
    get_cumsum = _block(counts, "uint32_t ServerDedupCounts::GetCumSum(")
    assert "DataCopyParams sumIntriParams{static_cast<uint16_t>(activeCumSumCores_), 1, 0, 0}" in get_cumsum
    assert "newAivId * UB_ALIGN" not in get_cumsum
    assert "layout_.cumsumOffset + newAivId * UB_ALIGN" in _block(counts, "void ServerDedupCounts::PublishRecvCntSum(")
    wait_flag = _block(counts, "void ServerDedupCounts::WaitCumSumFlag()")
    assert "newAivId * aivUsedCumSum_ * UB_ALIGN" in wait_flag
    assert "if (newAivId < activeCumSumCores_)" in wait_flag
    assert "layout_.cumsumOffset + newAivId * UB_ALIGN" in wait_flag


def test_shared_prefix_layout_and_delayed_fragments(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for C++ CPU validation")
    split = _block(_source("dispatch_server_dedup_counts.h"), "void SplitToCore(")
    source = tmp_path / "shared-prefix.cpp"
    source.write_text(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h"
using namespace DispatchDedup;
static void SplitToCore(uint32_t total,uint32_t cores,uint32_t core,
                        uint32_t& start,uint32_t& end,uint32_t& count) {
''' + split + r'''
}
int main() {
    constexpr uint32_t guard=0x5a5a5a5aU;
    unsigned cases=0;
    for(uint32_t cores: {16U,28U,56U,72U}) for(uint32_t experts: {32U,64U,384U,1024U}) {
        const uint32_t receivers=cores/2U,active=std::min(receivers,experts);
        const auto layout=DispatchServerDedupCountLayout(experts,cores);
        const uint64_t matrix=uint64_t(receivers)*receivers*32U;
        // Capacity and every control offset remain bit-for-bit compatible.
        assert(layout.cumsumOffset==uint64_t(experts)*512U);
        assert(layout.prefixOffset==layout.cumsumOffset+matrix);
        assert(layout.prefixStride==(uint64_t(experts)*4U+31U)/32U*32U);
        assert(layout.cumsumFlagOffset==layout.prefixOffset+receivers*layout.prefixStride);
        const uint64_t oldBytes=layout.cntOffset+layout.cumsumFlagOffset+matrix;
        assert(layout.counterOffset==(oldBytes+1023U)/1024U*1024U);
        assert(layout.stateBytes==layout.counterOffset+1024U);
        for(unsigned iteration=0;iteration<4;++iteration) {
            std::vector<uint32_t> gm(layout.stateBytes/4U+8U,guard),expected(experts),published(experts,guard);
            uint32_t sum=0;
            for(uint32_t i=0;i<experts;++i) {sum+=(i+iteration)%5U==0U?i%17U:0U;expected[i]=sum;}
            std::vector<std::vector<bool>> ready(receivers,std::vector<bool>(active,false));
            const uint64_t prefix=(layout.cntOffset+layout.prefixOffset)/4U;
            // Reverse completion leaves producer0 last. No receiver can see a
            // complete prefix until every producer publishes its existing flag.
            for(uint32_t next=active;next>0U;--next) {
                const uint32_t producer=next-1U;
                uint32_t start,end,count;SplitToCore(experts,receivers,producer,start,end,count);
                assert(count && end-start==count);
                for(uint32_t i=start;i<end;++i) {assert(gm[prefix+i]==guard);gm[prefix+i]=expected[i];published[i]=expected[i];}
                for(uint32_t consumer=0;consumer<receivers;++consumer) ready[consumer][producer]=true;
                for(uint32_t consumer=0;consumer<receivers;++consumer) {
                    const bool acquired=std::all_of(ready[consumer].begin(),ready[consumer].end(),[](bool v){return v;});
                    assert(acquired==(producer==0U));
                    if(acquired) for(uint32_t i=0;i<experts;++i) assert(gm[prefix+i]==expected[i]);
                }
            }
            assert(published==expected); // sendCountsOut retains exact values.
            for(uint64_t i=0;i<gm.size();++i)
                if(i<prefix || i>=prefix+experts) assert(gm[i]==guard);
            ++cases;
        }
    }
    std::cout<<"PASS shared prefix: "<<cases<<" layouts/routes, delayed disjoint fragments, all consumers, untouched reservation\n";
}
''', encoding="utf-8")
    executable = tmp_path / "shared-prefix.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", "-I" + str(ROOT / "csrc"),
                    str(source), "-o", str(executable)], check=True, timeout=180)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)
    assert "PASS shared prefix: 64 layouts/routes" in result.stdout
