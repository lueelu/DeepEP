// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#pragma once

#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"

// Fixed-shape specification for one routed or shared URMA binary.
//
// The runtime DeepJIT source instantiates the entry body with the plan's
// source-row capacity and model geometry, so this file's arithmetic sees those
// values as compile-time constants. Dynamic per-invocation values (BS, rank,
// addresses, epoch, clamp, received rows and task counts) stay in Arguments.
//
// The remaining ABI types are fixed by construction and are not template
// parameters: int32 top-k indices, FP8 E4M3 routed activations/weights,
// E8M0 scales and BF16 output; top-k weights are FP32.
namespace MegaMoeUrma {

template <uint32_t Capacity, uint32_t Hidden, uint32_t Intermediate, uint32_t TopKCount, uint32_t LocalExperts,
          uint32_t Ep, uint32_t WeightsType, uint32_t CubeCores = kAic, uint32_t VectorCores = 2U * kAic>
struct MegaMoeSpec {
    static_assert(Ep >= kMinWorld && Ep <= kWorld, "URMA requires EP4/EP8/EP16");
    static_assert(LocalExperts >= 1U && LocalExperts <= kMaxLocalExperts);
    // Qualify the capacity constant so this does not resolve to kExperts below.
    static_assert(Ep * LocalExperts <= MegaMoeUrma::kExperts);
    static_assert(TopKCount >= 1U && TopKCount <= 8U);
    static_assert(Hidden >= 256U && Hidden <= 8192U && Hidden % 256U == 0U);
    static_assert(Intermediate >= 256U && Intermediate <= 16384U && Intermediate % 256U == 0U);
    static_assert(WeightsType <= 1U);
    static_assert(Capacity >= 1U);

    static constexpr uint32_t kSourceRowCapacity = Capacity;
    static constexpr uint32_t kHidden = Hidden;
    static constexpr uint32_t kIntermediate = Intermediate;
    static constexpr uint32_t kGmm1Columns = 2U * Intermediate;
    static constexpr uint32_t kTopK = TopKCount;
    static constexpr uint32_t kLocalExperts = LocalExperts;
    static constexpr uint32_t kEp = Ep;
    static constexpr uint32_t kExperts = Ep * LocalExperts;
    // 0 selects FP32 route weights; 1 appends one FP32 weight per source row to
    // the scale PUT payload.
    static constexpr uint32_t kWeightsType = WeightsType;
    static constexpr uint32_t kDispatchScaleStride =
        WeightsType ? ((Hidden / 32U + 31U) / 32U) * 32U + 32U : Hidden / 32U;
    static constexpr uint32_t kCubeCores = CubeCores;
    static constexpr uint32_t kVectorCores = VectorCores;
    static constexpr uint32_t kCommunicationWorkers = Ep - 1U;
    static constexpr uint32_t kActivationWorkers = CubeCores - kCommunicationWorkers;
    static constexpr uint32_t kReduceWorkers = VectorCores - kCommunicationWorkers;
};

}  // namespace MegaMoeUrma
