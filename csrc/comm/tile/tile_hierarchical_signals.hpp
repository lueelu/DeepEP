// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once
#include "comm/common.hpp"
#include "kernel_operator.h"
#include "shmem.h"

namespace deepep::comm::tile {
// Scratch/event belong exclusively to the calling AIV and must be idle on entry.
class TileHierarchicalSignals {
public:
    using Transport = comm::CopyTransport;
    DEEPEP_COMM_DEVICE TileHierarchicalSignals(uintptr_t base, __ubuf__ uint32_t* scratch, uint32_t eventId = 0)
        : base(reinterpret_cast<__gm__ uint8_t*>(base)), scratch(scratch), eventId(eventId) {}

    DEEPEP_COMM_DEVICE void FinishMte()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
    DEEPEP_COMM_DEVICE void Store(uint64_t offset, uint32_t epoch)
    {
        AscendC::GlobalTensor<uint32_t> value;
        value.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(base + offset));
        value.SetValue(0, epoch);
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(value);
        __asm__ __volatile__("");
    }
    DEEPEP_COMM_DEVICE void Wait(uint64_t offset, uint32_t epoch)
    {
        aclshmem_uint32_wait_until(reinterpret_cast<__gm__ uint32_t*>(base + offset), ACLSHMEM_CMP_GE, epoch);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    DEEPEP_COMM_DEVICE void Quiet(uint32_t peer, Transport transport, uint32_t qp)
    {
        if (transport == Transport::Udma) {
            aclshmemx_udma_qp_quiet(peer, qp);
        } else {
            aclshmemx_roce_qp_quiet(peer, qp, scratch, eventId);
        }
    }
    DEEPEP_COMM_DEVICE void PutFlag(uint64_t dst, uint64_t src, uint32_t peer, Transport transport, uint32_t qp)
    {
        auto out = reinterpret_cast<__gm__ uint32_t*>(base + dst);
        auto value = reinterpret_cast<__gm__ uint32_t*>(base + src);
        if (transport == Transport::Udma) {
            aclshmemx_udma_qp_put_nbi<uint32_t>(out, value, scratch, 1, peer, qp, eventId);
        } else {
            aclshmemx_roce_qp_put_nbi<uint32_t>(out, value, scratch, 1, peer, qp, eventId);
        }
    }

private:
    __gm__ uint8_t* base;
    __ubuf__ uint32_t* scratch;
    uint32_t eventId;
};

} // namespace deepep::comm::tile
