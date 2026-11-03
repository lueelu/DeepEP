"""Execute the reference device recorder on CPU and audit its placement contract.

These checks do not establish CANN resource usage or device performance.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
LL = ROOT / 'csrc/kernels/dispatch/low_latency'
COMMON = LL / 'common'


def compiler():
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    native = shutil.which('g++') or shutil.which('clang++')
    if not zig and not native:
        pytest.skip('C++17 compiler required for the reference recorder CPU model')
    return [zig, 'c++'] if zig else [native]


def headers(path):
    # OFF deliberately has no clock, cache-flush, fence or pipeline API. It must
    # compile without referencing any of them, rather than only branch at run time.
    (path / 'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
namespace AscendC {
inline uint32_t core = 0, cores = 16;
inline uint32_t GetBlockIdx() { return core; }
inline uint32_t GetBlockNum() { return cores; }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
inline uint64_t tick = uint64_t(1) << 60, clocks = 0, fences = 0;
inline uint64_t GetSystemCycle() { ++clocks; return tick += 17; }
enum class MemDsbT { DDR };
template<MemDsbT> void DataSyncBarrier() { ++fences; }
#endif
}
''', encoding='utf-8')
    (path / 'shmem.h').write_text(r'''
#pragma once
#include <cstdint>
inline uint64_t flushes = 0, flushedLength = 0;
inline uint8_t* flushedAddress = nullptr;
inline void dcci_cachelines(uint8_t* address, uint64_t length) {
    ++flushes; flushedAddress = address; flushedLength = length;
}
''', encoding='utf-8')


@pytest.mark.parametrize('enabled', [False, True])
def test_record_is_core_local_exact_and_off_has_no_device_work(tmp_path, enabled):
    headers(tmp_path)
    source = tmp_path / 'reference.cpp'
    source.write_text(r'''
#include "ll_reference_profile_device.h"
#include <cassert>
#include <cstring>
using namespace LlReferenceProfile;
int main() {
    static_assert(kSchema == 4 && kWordsPerCore == 640 && kSlots == 80 && kFields == 8);
    static_assert(kWordsPerCore * sizeof(uint64_t) == 5120);
    constexpr uint64_t guard = 0x55aaccff12345678ULL;
    alignas(512) uint64_t raw[74 * kWordsPerCore];
    for (uint32_t cores : {16U, 56U, 72U}) {
        AscendC::cores = cores;
        for (auto& word : raw) word = guard;
        auto* base = raw + kWordsPerCore;
        for (uint32_t i=0; i<cores*kWordsPerCore; ++i) base[i]=0;
        for (uint32_t core=0; core<cores; ++core) {
            AscendC::core = core;
            auto address = reinterpret_cast<uint8_t*>(base);
            const auto entry = DeviceProfile::EntryClock(address);
            DeviceProfile profile;
            profile.Init(address);
            profile.Value<0>(entry);
            profile.Mark<1>();
            profile.Value<33>(0); // observed zero, different from absent word 32
            profile.Value<34>(0);
            profile.Value<37>(0);
            profile.Value<41>(0);
            profile.Value<43>(0);
            profile.Mark<21>();
            profile.Finish(31, 64, 1);
            auto* row = base + core*kWordsPerCore;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            assert(profile.Enabled());
            assert(row[0] > (uint64_t(1) << 60));
            assert(row[1]-row[0] == 17 && row[21]-row[1] == 17 && row[57]-row[21] == 17);
            constexpr uint64_t mask = (uint64_t(1)<<0)|(uint64_t(1)<<1)|(uint64_t(1)<<21)|
                (uint64_t(1)<<33)|(uint64_t(1)<<34)|(uint64_t(1)<<37)|(uint64_t(1)<<41)|
                (uint64_t(1)<<43)|(uint64_t(1)<<57);
            assert(row[58] == mask && row[32] == 0 && row[33] == 0);
            assert(row[50]==31 && row[51]==core && row[52]==64 && row[53]==cores && row[54]==1);
            assert(row[55]==kSchema && row[56]==1 && row[59]==kMagic && row[60]==1 && row[61]==1);
            assert(row[22]==0 && row[23]==0 && row[44]==0 && row[49]==0 && row[62]==0 && row[63]==0);
            assert(flushedAddress == reinterpret_cast<uint8_t*>(row) && flushedLength == 1023);
            // A helper with inclusive end must flush only this core's eight
            // 64B cachelines, not the following core or the trailing guard.
            assert((reinterpret_cast<uintptr_t>(flushedAddress)+flushedLength)/64 ==
                   reinterpret_cast<uintptr_t>(row)/64+15);
#else
            static_assert(sizeof(DeviceProfile) == 1);
            assert(!profile.Enabled() && entry == 0 && profile.Clock() == 0);
            for (uint32_t i=0; i<kWordsPerCore; ++i) assert(row[i] == 0);
#endif
        }
        for (uint32_t i=0; i<kWordsPerCore; ++i) {
            assert(raw[i] == guard);
            assert(base[cores*kWordsPerCore+i] == guard);
        }
    }
    DeviceProfile disabled;
    disabled.Init(nullptr);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const auto clockCount = AscendC::clocks, flushCount=flushes, fenceCount=AscendC::fences;
    assert(clockCount == (16+56+72)*4 && flushCount == 16+56+72 && fenceCount == flushCount);
#endif
    assert(!disabled.Enabled() && DeviceProfile::EntryClock(nullptr)==0 && disabled.Clock()==0);
    disabled.Value<33>(123);
    disabled.Mark<0>();
    disabled.Finish(0,16,1);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    assert(clockCount==AscendC::clocks && flushCount==flushes && fenceCount==AscendC::fences);
#endif
}
'''.replace('#include <cstring>', '#include <cstring>\n#include <initializer_list>'), encoding='utf-8')
    exe = tmp_path / 'reference.exe'
    subprocess.run(compiler() + ['-std=c++17', '-O2', '-UNDEBUG',
        *(['-DDEEPEP_LL_REFERENCE_PROFILE=1'] if enabled else []),
        '-I' + str(tmp_path), '-I' + str(COMMON), str(source), '-o', str(exe)],
        check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)


def test_legacy_and_reference_cannot_compile_together(tmp_path):
    headers(tmp_path)
    source = tmp_path / 'conflict.cpp'
    source.write_text('#include "ll_reference_profile_device.h"\nint main() {}\n', encoding='utf-8')
    result = subprocess.run(compiler() + ['-std=c++17', '-fsyntax-only',
        '-DDEEPEP_LL_REFERENCE_PROFILE=1', '-DDEEPEP_LL_STAGE_TIMER=1',
        '-I' + str(tmp_path), '-I' + str(COMMON), str(source)],
        capture_output=True, text=True, timeout=180)
    assert result.returncode != 0
    assert 'has been removed' in result.stderr


def read(relative):
    return (LL / relative).read_text(encoding='utf-8')


def test_reference_points_preserve_business_boundaries_and_async_tail():
    entry = (ROOT / 'csrc/kernels/dispatch/low_latency.cpp').read_text(encoding='utf-8')
    assert entry.index('DeviceProfile::EntryClock') < entry.index('util_set_ffts_config')
    assert entry.index('kernel.Process();') < entry.index('profile.Finish(')
    kernel = read('dispatch_server_dedup_kernel.h')
    for begin, business, end in ((2,'RotateDispatchServerDedupMagic',3), (6,'counts_.SendCnt()',7),
            (8,'sender_.Process(',9), (10,'counts_.RecvCnt()',11), (12,'counts_.CalcCumsum()',13),
            (14,'counts_.CompleteCumsum()',15), (16,'PipeBarrier<PIPE_ALL>()',17),
            (18,'receiver_.Process(',19), (20,'counts_.Finish()',21)):
        assert kernel.index(f'profile_->Mark<{begin}>') < kernel.index(business) < kernel.index(f'profile_->Mark<{end}>')
    send = read('device/dispatch_server_dedup_send.h')
    assert send.index('SendWithoutDedup(input);') < send.index('profile->Mark<39>') < send.index('rdmaBatch_.Flush(')
    assert send.index('rdmaBatch_.Flush(') < send.index('profile->Mark<40>')
    assert 'routes / sendCores_ + (GetBlockIdx() < routes % sendCores_' in send
    counts = read('device/dispatch_server_dedup_counts.h')
    assert counts.index('profile_->Mark<24>') < counts.index('while (sumOfFlag != compareTarget)') < counts.index('profile_->Mark<25>')
    prefix = counts[counts.index('const uint32_t offset = GetCumSum(newAivId);'):]
    assert prefix.index('profile_->Mark<27>') < prefix.index('if (packedFrames_) WaitDispatchClearStatus()')
    recv = read('device/dispatch_server_dedup_recv.h')
    retry = recv.split('do {', 1)[1].split('} while (probe.readyMask == 0U);', 1)[0]
    packet = recv.split('for (uint32_t i = count; i > 0U; --i)', 1)[1].split('++consumed;', 1)[0]
    assert not re.search(r'profile|Clock|Mark|GetSystemCycle', retry)
    assert not re.search(r'profile|Clock|Mark|GetSystemCycle', packet)
    assert recv.count('profile->Clock()') == 5
    assert kernel.index('profile_->Mark<21>') < kernel.index('receiver_.ExportPacketTrace')
    assert '#ifdef DEEPEP_LL_REFERENCE_PROFILE\n        // Three clocks' in recv
    assert recv.index('profile->Mark<35>') < recv.index('profile->Mark<36>') < recv.index('profile->Value<33>(pollCycles)')
    zero_tasks = recv.split('if (tasks_.Count() == 0U)', 1)[1].split('return;', 1)[0]
    for word in (33, 34, 41):
        assert f'profile->Value<{word}>(0U)' in zero_tasks


def test_recorder_has_no_ub_event_or_new_pipeline_barrier():
    device = read('common/ll_reference_profile_device.h')
    assert not re.search(r'\b(?:LocalTensor|TBuf|TQue|TPipe|HardEvent|PipeBarrier|SetFlag|WaitFlag)\b', device)
    assert 'record_[Word] = value;' in device
    assert 'valid_ |= uint64_t(1) << Word;' in device
    assert 'uint64_t values[' not in device
    assert 'kPacketLogWord * sizeof(uint64_t) - 1U' in device
    # No device recorder reads its output buffer to update any duration/count.
    assert not re.search(r'=\s*record_\[', device)
