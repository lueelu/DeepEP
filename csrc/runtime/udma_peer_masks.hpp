// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include "kernels/common.hpp"
#include "kernels/udma_layout.hpp"

namespace deepep::udma_admission {
inline constexpr uint64_t PeerMask(uint32_t rank, uint32_t peer, uint32_t world)
{
    uint64_t mask = first_hit_schedule::CombineQpMask(rank, peer, world);
    // HT Dispatch QP ownership remains eight-way. LL uses direct MTE peer
    // mappings and its separate logical-group protocol, not these QP lanes.
    if (rank / 8U != peer / 8U) {
        // Dispatch owns each destination lane's pair; admission is symmetric.
        mask |= uint64_t{3} << (2U * (peer % 8U));
        mask |= uint64_t{3} << (2U * (rank % 8U));
    }
    return mask;
}
static_assert(ascend_deepep::udma_layout::kQpCount == 32U);
}  // namespace deepep::udma_admission
