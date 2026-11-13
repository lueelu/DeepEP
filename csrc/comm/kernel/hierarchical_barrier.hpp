/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See csrc/comm/LICENSE.catccos for the full text of the License.
 */
#ifndef DEEPEP_COMM_KERNEL_HIERARCHICAL_BARRIER_HPP
#define DEEPEP_COMM_KERNEL_HIERARCHICAL_BARRIER_HPP

#include <cstddef>
#include <cstdint>
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "shmem.h"

namespace deepep::comm::kernel {

// Host-ordered barrier for hierarchical AllToAll. The caller first completes
// its data kernel and all QP operations, then launches one MIX block. Only AIV 0
// exchanges signals; the other AIV and AIC have no work. Do not overlap calls
// using the same workspace or QPs.
class HierarchicalBarrier {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr uint32_t SLOT_BYTES = 64;
    static constexpr uint32_t SLOT_WORDS = SLOT_BYTES / sizeof(uint32_t);

    struct Arguments {
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t groupSize;
        uint32_t epoch;
    };

    struct Params : Arguments {
        __gm__ uint32_t* ptrWorkspace = nullptr;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(const Arguments& args, uint8_t* workspace)
            : Arguments(args),
              ptrWorkspace(reinterpret_cast<__gm__ uint32_t*>(
                  (reinterpret_cast<uintptr_t>(workspace) + SLOT_BYTES - 1) & ~(uintptr_t(SLOT_BYTES) - 1)))
        {}
    };

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        uint32_t slots = 1; // Local send value, isolated from incoming signals.
        for (uint32_t shift = 1; shift < args.rankSize; shift *= 2) {
            ++slots;
        }
        // SHMEM calloc guarantees 16-byte allocation alignment. Reserve enough
        // space to align the signal region to a full cache line.
        return static_cast<size_t>(slots) * SLOT_BYTES + SLOT_BYTES - 1;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace) { return Params{args, workspace}; }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params& params)
    {
        (void)params;
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params& params)
    {
        if (AscendC::GetBlockIdx() != 0 || params.rankSize <= 1) {
            return;
        }
        Catlass::Arch::Resource<ArchTag> resource;
        auto ub = resource.ubBuf.template GetBufferByByte<uint32_t>(0);
        auto scratch = reinterpret_cast<__ubuf__ uint32_t*>(ub.GetPhyAddr());
        auto src = params.ptrWorkspace;
        AscendC::GlobalTensor<uint32_t> sendValue;
        sendValue.SetGlobalBuffer(src);
        sendValue.SetValue(0, params.epoch);
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(sendValue);
        __asm__ __volatile__("");

        // Dissemination: each round doubles the set of ranks known to have
        // arrived. Each receiving slot has one writer and a separate cache line.
        uint32_t round = 0;
        for (uint32_t shift = 1; shift < params.rankSize; shift *= 2, ++round) {
            const uint32_t peer = (params.rankIdx + shift) % params.rankSize;
            auto dst = params.ptrWorkspace + (round + 1) * SLOT_WORDS;
            if (params.rankIdx / params.groupSize == peer / params.groupSize) {
                aclshmemx_udma_qp_put_nbi<uint32_t>(dst, src, scratch, 1, peer, 0, 0);
                aclshmemx_udma_qp_quiet(peer, 0);
            } else {
                aclshmemx_roce_qp_put_nbi<uint32_t>(dst, src, scratch, 1, peer, 0, 0);
                aclshmemx_roce_qp_quiet(peer, 0, scratch, 0);
            }
            // A fast rank may already send the next epoch. Never clear slots
            // between calls; GE accepts that progress without missing a signal.
            aclshmem_uint32_wait_until(dst, ACLSHMEM_CMP_GE, params.epoch);
        }
    }
};

} // namespace deepep::comm::kernel
#endif
