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

__aicore__ inline void RoutedBody(MegaMoeUrma::Arguments& args)
{
    MegaMoeUrmaPrologue(args);
    MegaMoeUrma::ProcessRouted(args);
}

__aicore__ inline void SharedBody(MegaMoeUrma::Arguments& args)
{
    MegaMoeUrmaPrologue(args);
    MegaMoeUrma::ProcessShared(args);
}

template <bool kShared, uint32_t kSkSlot>
__aicore__ inline void MegaMoeUrmaSk(const MegaMoeUrma::Arguments* args, sk::SkSystemArgs* system_args)
{
    system_args->SkInit();
    // The stages consume a mutable argument block while the SuperKernel
    // parameter view is read-only.
    MegaMoeUrma::Arguments block = *args;
    if constexpr (kShared)
        SharedBody(block);
    else
        RoutedBody(block);
}
#endif

}  // namespace MegaMoeUrmaEntry
