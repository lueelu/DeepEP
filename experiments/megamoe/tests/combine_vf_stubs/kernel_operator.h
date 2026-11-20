// Copyright (c) 2026, Lu Lu
#pragma once

// CPU semantics for the register primitives used by the production Combine VFs.
// Deliberately execute the actual VFs, including masks, Gather and BF16 unpack,
// rather than replacing either VF with a second implementation of its algorithm.
#define __simd_vf__

namespace AscendC {
constexpr uint32_t GetVecLen()
{
    return 256U;
}
enum class CMPMODE { EQ, NE, LT };
namespace Reg {
enum class MaskPattern { ALL };
enum class MaskMergeMode { ZEROING, MERGING };
enum class HistogramsBinType { BIN0 };
enum class HistogramsType { FREQUENCY };
enum class ReduceType { SUM };
enum class PostLiteral { POST_MODE_UPDATE };
enum class LoadDist { DIST_BRC_B32, DIST_UNPACK_B16 };
enum class RegLayout { ZERO };
enum class SatMode { UNKNOWN };
struct CastTrait {
    RegLayout layout;
    SatMode saturation;
    MaskMergeMode merge;
    RoundMode rounding;
};
struct MaskReg {
    std::array<bool, 256> lanes{};
};
template <typename T>
struct RegTensor {
    std::array<T, GetVecLen() / sizeof(T)> lanes{};
};
struct UnalignRegForStore {};
template <typename T, MaskPattern>
MaskReg CreateMask()
{
    MaskReg mask;
    std::fill_n(mask.lanes.begin(), GetVecLen() / sizeof(T), true);
    return mask;
}
template <typename T>
MaskReg UpdateMask(uint32_t& remaining)
{
    MaskReg mask;
    const uint32_t count = std::min(remaining, GetVecLen() / static_cast<uint32_t>(sizeof(T)));
    std::fill_n(mask.lanes.begin(), count, true);
    remaining -= count;
    return mask;
}
template <typename T>
void Duplicate(RegTensor<T>& dst, T value, const MaskReg& mask)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i)
        if (mask.lanes[i]) dst.lanes[i] = value;
}
template <typename T>
void Arange(RegTensor<T>& dst, T first)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i) dst.lanes[i] = first + static_cast<T>(i);
}
template <typename T>
void Muls(RegTensor<T>& dst, const RegTensor<T>& src, T value, const MaskReg& mask)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i)
        if (mask.lanes[i]) dst.lanes[i] = src.lanes[i] * value;
}
template <typename T>
void Adds(RegTensor<T>& dst, const RegTensor<T>& src, T value, const MaskReg& mask)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i)
        if (mask.lanes[i]) dst.lanes[i] = src.lanes[i] + value;
}
template <typename T>
void Mins(RegTensor<T>& dst, const RegTensor<T>& src, T limit, const MaskReg& mask)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i)
        if (mask.lanes[i]) dst.lanes[i] = std::min(src.lanes[i], limit);
}
template <typename S, typename D, HistogramsBinType, HistogramsType>
void Histograms(RegTensor<D>& dst, const RegTensor<S>& src, const MaskReg& mask)
{
    for (size_t i = 0; i < src.lanes.size(); ++i)
        if (mask.lanes[i] && src.lanes[i] < dst.lanes.size()) ++dst.lanes[src.lanes[i]];
}
template <typename T>
void Gather(RegTensor<T>& dst, T* src, const RegTensor<uint32_t>& offsets, const MaskReg& mask)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i)
        if (mask.lanes[i]) {
            ReduceTest::CheckUbAccess(src + offsets.lanes[i], sizeof(T), false);
            dst.lanes[i] = src[offsets.lanes[i]];
        }
}
template <class T>
void ShiftRights(RegTensor<T>& dst, const RegTensor<T>& src, int16_t n, const MaskReg& mask)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i)
        if (mask.lanes[i]) dst.lanes[i] = src.lanes[i] >> n;
}
template <class T>
void And(RegTensor<T>& dst, const RegTensor<T>& a, const RegTensor<T>& b, const MaskReg& mask)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i)
        if (mask.lanes[i]) dst.lanes[i] = a.lanes[i] & b.lanes[i];
}
template <typename T, CMPMODE Mode>
void Compares(MaskReg& dst, const RegTensor<T>& src, T expected, const MaskReg& mask)
{
    dst = {};
    for (size_t i = 0; i < src.lanes.size(); ++i)
        dst.lanes[i] = mask.lanes[i] && (Mode == CMPMODE::EQ   ? src.lanes[i] == expected
                                         : Mode == CMPMODE::LT ? src.lanes[i] < expected
                                                               : src.lanes[i] != expected);
}
template <typename T, MaskMergeMode Merge = MaskMergeMode::ZEROING>
void Add(RegTensor<T>& dst, const RegTensor<T>& left, const RegTensor<T>& right, const MaskReg& mask)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i) {
        if (mask.lanes[i])
            dst.lanes[i] = left.lanes[i] + right.lanes[i];
        else if constexpr (Merge == MaskMergeMode::ZEROING)
            dst.lanes[i] = T{};
    }
}
template <typename T>
void Mul(RegTensor<T>& dst, const RegTensor<T>& left, const RegTensor<T>& right, const MaskReg& mask)
{
    for (size_t i = 0; i < dst.lanes.size(); ++i)
        if (mask.lanes[i]) dst.lanes[i] = left.lanes[i] * right.lanes[i];
}
template <ReduceType, typename T>
void Reduce(RegTensor<T>& dst, const RegTensor<T>& src, const MaskReg& mask)
{
    T sum{};
    for (size_t i = 0; i < src.lanes.size(); ++i)
        if (mask.lanes[i]) sum += src.lanes[i];
    // Only lane zero is guaranteed by this test double. Reading other result
    // lanes as if Reduce broadcast the sum would expose the poison below.
    dst.lanes.fill(static_cast<T>(0xBAADU));
    dst.lanes[0] = sum;
}
template <typename T, PostLiteral>
void StoreUnAlign(T*& dst, const RegTensor<T>& src, UnalignRegForStore&, uint32_t count)
{
    ReduceTest::CheckUbAccess(dst, count * sizeof(T), true);
    std::copy_n(src.lanes.begin(), count, dst);
    dst += count;
}
enum class GatherMaskMode { STORE_REG };
enum class SpecialPurposeReg { AR };
inline uint32_t packedLanes;
template <SpecialPurposeReg>
void ClearSpr()
{
    packedLanes = 0;
}
template <class T, GatherMaskMode>
void GatherMask(RegTensor<T>& dst, const RegTensor<T>& src, const MaskReg& m)
{
    packedLanes = 0;
    for (uint32_t i = 0; i < src.lanes.size(); ++i)
        if (m.lanes[i]) dst.lanes[packedLanes++] = src.lanes[i];
}
template <class T>
void StoreUnAlign(T*& dst, const RegTensor<T>& src, UnalignRegForStore& stream)
{
    StoreUnAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, src, stream, packedLanes);
}
template <typename T>
void StoreUnAlignPost(T*, UnalignRegForStore&, int = 0)
{
}
template <typename T, LoadDist Dist>
void DataCopy(RegTensor<T>& dst, T* src)
{
    if constexpr (Dist == LoadDist::DIST_BRC_B32) {
        ReduceTest::CheckUbAccess(src, sizeof(T), false);
        dst.lanes.fill(*src);
    } else {
        static_assert(sizeof(T) == 2U);
        ReduceTest::CheckUbAccess(src, GetVecLen() / 2U, false);
        for (size_t i = 0; i < dst.lanes.size() / 2U; ++i) dst.lanes[2U * i] = src[i];
    }
}
template <typename D, typename S, const CastTrait& Trait>
void Cast(RegTensor<D>& dst, const RegTensor<S>& src, const MaskReg& mask)
{
    static_assert(Trait.layout == RegLayout::ZERO && sizeof(D) == 4U && sizeof(S) == 2U);
    for (size_t i = 0; i < dst.lanes.size(); ++i)
        dst.lanes[i] = mask.lanes[i] ? static_cast<D>(static_cast<float>(src.lanes[2U * i])) : D{};
}
template <typename T>
void LoadAlign(RegTensor<T>& dst, T* src)
{
    ReduceTest::CheckUbAccess(src, GetVecLen(), false);
    std::copy_n(src, dst.lanes.size(), dst.lanes.begin());
}
template <typename T>
void StoreAlign(T* dst, const RegTensor<T>& src, const MaskReg& mask)
{
    for (size_t i = 0; i < src.lanes.size(); ++i)
        if (mask.lanes[i]) {
            ReduceTest::CheckUbAccess(dst + i, sizeof(T), true);
            dst[i] = src.lanes[i];
        }
}
}  // namespace Reg
}  // namespace AscendC
