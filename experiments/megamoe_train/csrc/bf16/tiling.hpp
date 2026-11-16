// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include <cstdint>
#include <string>

#include <torch/extension.h>

struct MegaMoeTilingData;

namespace deepep::megamoe_train::ops::detail {

struct MegaMoeHostTiling {
    uint64_t workspace_bytes;
    uint64_t peer_bytes;
};

// Spell out both trailing arguments so old bool-only calls cannot silently
// become swiglu_limit=1.0 after a merge.
MegaMoeHostTiling MakeMegaMoeHostTiling(MegaMoeTilingData& tiling, int64_t m, int64_t k, int64_t n, int64_t topk,
                                        int64_t world_size, int64_t expert_per_rank, int64_t max_recv_tokens, int rank,
                                        bool bf16, double swiglu_limit, bool bf16_direct_dispatch,
                                        const std::string& save_gmm1_output = "auto", uint32_t topk_weight_type = 0);

MegaMoeHostTiling GetBf16EarlyTopkWeightBufferBytes(const MegaMoeTilingData& tiling);

}  // namespace deepep::megamoe_train::ops::detail
