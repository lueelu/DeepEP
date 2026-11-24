// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

// MegaMoE URMA device entries shared by the runtime DeepJIT source. Routed and
// shared variants compile separately because a DeepJIT binary carries exactly
// one kernel; the by-value Arguments ABI is unchanged and the SuperKernel
// entries read the same packed argument block.
#pragma once

#include <cstdint>

#include "basic_api/kernel_basic_intf.h"
#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_specialization.hpp"

#if defined(__NPU_DEVICE__)
#include "kernels/mega_moe_urma/mega_moe_urma.hpp"
#endif

namespace MegaMoeUrmaEntry {

#if defined(__NPU_DEVICE__)
using namespace AscendC;

__aicore__ inline void MegaMoeUrmaPrologue(MegaMoeUrma::Arguments& args)
{
    InitSocState();
    util_set_ffts_config(args.ffts);
    SetSysWorkspaceForce(reinterpret_cast<GM_ADDR>(args.systemWorkspace));
}

// Capacity and model geometry are compile-time arguments of the generated JIT
// source. AIC/AIV counts and the UB budget keep their platform defaults, so the
// seven-argument form below is the complete public specialization.
template <uint32_t kCapacity, uint32_t kHidden, uint32_t kIntermediate, uint32_t kTopK, uint32_t kLocalExperts,
          uint32_t kEp, uint32_t kWeightsType, uint32_t kCubeCores = MegaMoeUrma::kAic,
          uint32_t kVectorCores = 2U * MegaMoeUrma::kAic>
__aicore__ inline void RoutedBody(MegaMoeUrma::Arguments& args)
{
    using Spec = MegaMoeUrma::MegaMoeSpec<kCapacity, kHidden, kIntermediate, kTopK, kLocalExperts, kEp, kWeightsType,
                                          kCubeCores, kVectorCores>;
    MegaMoeUrmaPrologue(args);
    MegaMoeUrma::ProcessRouted<Spec>(args);
}

template <uint32_t kCapacity, uint32_t kHidden, uint32_t kIntermediate, uint32_t kTopK, uint32_t kLocalExperts,
          uint32_t kEp, uint32_t kWeightsType, uint32_t kCubeCores = MegaMoeUrma::kAic,
          uint32_t kVectorCores = 2U * MegaMoeUrma::kAic>
__aicore__ inline void SharedBody(MegaMoeUrma::Arguments& args)
{
    using Spec = MegaMoeUrma::MegaMoeSpec<kCapacity, kHidden, kIntermediate, kTopK, kLocalExperts, kEp, kWeightsType,
                                          kCubeCores, kVectorCores>;
    MegaMoeUrmaPrologue(args);
    MegaMoeUrma::ProcessShared<Spec>(args);
}

template <bool kShared, uint32_t kSkSlot, uint32_t kCapacity, uint32_t kHidden, uint32_t kIntermediate,
          uint32_t kTopK, uint32_t kLocalExperts, uint32_t kEp, uint32_t kWeightsType,
          uint32_t kCubeCores = MegaMoeUrma::kAic, uint32_t kVectorCores = 2U * MegaMoeUrma::kAic>
__aicore__ inline void MegaMoeUrmaSk(const MegaMoeUrma::Arguments* args, sk::SkSystemArgs* system_args)
{
    system_args->SkInit();
    // The stages consume a mutable argument block while the SuperKernel
    // parameter view is read-only.
    MegaMoeUrma::Arguments block = *args;
    if constexpr (kShared)
        SharedBody<kCapacity, kHidden, kIntermediate, kTopK, kLocalExperts, kEp, kWeightsType, kCubeCores,
                   kVectorCores>(block);
    else
        RoutedBody<kCapacity, kHidden, kIntermediate, kTopK, kLocalExperts, kEp, kWeightsType, kCubeCores,
                   kVectorCores>(block);
}
#endif

}  // namespace MegaMoeUrmaEntry
