// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once
#include "kernel_operator.h"
#include "layout/mega_moe_layout.hpp"

namespace MegaMoeBF16Impl {
template <uint32_t RawRows = 128>
class SavedGateUpMxFp8 {
public:
    static constexpr uint32_t kChunkRows = RawRows == 192 ? 64 : RawRows;
    using Layout = MegaMoeBF16Training::SavedGateUpQuantUbLayoutFor<RawRows, kChunkRows>;

private:
    using DataTypeOut = fp8_e4m3fn_t;
    static constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
    static constexpr uint16_t MAX_EXP_FOR_FP8 = 0xff;
    static constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
    // CANN ShiftRights requires a signed int16_t scalar shift count.
    static constexpr int16_t SHR_NUM_FOR_BF16 = 7;
    static constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
    static constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x40;
    static constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64LL;
    static constexpr uint32_t vlForHalfNumber_ = 256U / sizeof(bfloat16_t);
    static constexpr uint16_t elementAfterReduce_ = 256U / 32;
    static constexpr uint16_t fpEmax_ = 0x0400;  // W8A8 E4M3FN maximum exponent.
    AscendC::LocalTensor<int8_t> gateData_{AscendC::TPosition::VECOUT, Layout::gateData, Layout::kElements};
    AscendC::LocalTensor<int8_t> upData_{AscendC::TPosition::VECOUT, Layout::upData, Layout::kElements};
    AscendC::LocalTensor<uint16_t> compactScale_{AscendC::TPosition::VECCALC, Layout::compactScale, Layout::kGroups};
    AscendC::LocalTensor<int8_t> gateScale_{AscendC::TPosition::VECOUT, Layout::gateScale, Layout::kRows * 32};
    AscendC::LocalTensor<int8_t> upScale_{AscendC::TPosition::VECOUT, Layout::upScale, Layout::kRows * 32};
    AscendC::LocalTensor<uint16_t> maxExp_{AscendC::TPosition::VECCALC, Layout::maxExp, Layout::kGroups};
    AscendC::LocalTensor<uint16_t> halfScale_{AscendC::TPosition::VECCALC, Layout::halfScale, Layout::kGroups};
    AscendC::LocalTensor<int8_t> gateHigh_{AscendC::TPosition::VECOUT, Layout::gateHigh, Layout::maskPlaneBytes};
    AscendC::LocalTensor<int8_t> upLow_{AscendC::TPosition::VECOUT, Layout::upLow, Layout::maskPlaneBytes};
    AscendC::LocalTensor<int8_t> upHigh_{AscendC::TPosition::VECOUT, Layout::upHigh, Layout::maskPlaneBytes};

public:
    __aicore__ inline void Save(AscendC::LocalTensor<bfloat16_t> gate, AscendC::LocalTensor<bfloat16_t> up,
                                GM_ADDR gateGm, GM_ADDR upGm, GM_ADDR gateScaleGm, GM_ADDR upScaleGm,
                                GM_ADDR gateHighGm, GM_ADDR upLowGm, GM_ADDR upHighGm, float limit, uint16_t rows,
                                uint16_t columns, uint32_t width, uint64_t offset)
    {
        // Keep M192 raw planes intact. Reuse a disjoint 64-row quantization
        // scratch region after all previous chunk stores have completed.
        for (uint16_t row = 0; row < rows; row += kChunkRows) {
            if (row != 0) {
                AscendC::SyncFunc<AscendC::HardEvent::MTE3_V>();
            }
            const uint16_t remaining = rows - row;
            const uint16_t chunkRows = remaining < kChunkRows ? remaining : kChunkRows;
            SaveChunk(gate[static_cast<uint32_t>(row) * columns], up[static_cast<uint32_t>(row) * columns], gateGm,
                      upGm, gateScaleGm, upScaleGm, gateHighGm, upLowGm, upHighGm, limit, chunkRows, columns, width,
                      offset + static_cast<uint64_t>(row) * width);
        }
    }

private:
    __aicore__ inline void SaveChunk(AscendC::LocalTensor<bfloat16_t> gate, AscendC::LocalTensor<bfloat16_t> up,
                                     GM_ADDR gateGm, GM_ADDR upGm, GM_ADDR gateScaleGm, GM_ADDR upScaleGm,
                                     GM_ADDR gateHighGm, GM_ADDR upLowGm, GM_ADDR upHighGm, float limit, uint16_t rows,
                                     uint16_t columns, uint32_t width, uint64_t offset)
    {
        // Host requires F%32==0; the 256-column tile therefore has whole groups.
        // Both BF16 inputs remain intact until the caller computes SwiGLU.
        const bool saveMasks = gateHighGm != nullptr;
        if (saveMasks) {
            ComputeClipMasks(gate, up, rows, columns, limit);
        }
        // Match W8A8 GroupMatmulSwiglu: Cast's SatMode::SAT only controls
        // FP8 overflow when CTRL[60] selects per-instruction saturation.
        // Preserve the caller's mode for the unchanged BF16 SwiGLU path.
        const int64_t overflowMode = AscendC::GetCtrlSpr<60, 60>();
        AscendC::SetCtrlSpr<60, 60>(0);
        Quantize(gate, gateData_, gateScale_, rows, columns);
        Quantize(up, upData_, upScale_, rows, columns);
        AscendC::SyncFunc<AscendC::HardEvent::V_S>();
        AscendC::SetCtrlSpr<60, 60>(overflowMode);
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
        Store(gateGm, gateData_, rows, columns, width, offset);
        Store(upGm, upData_, rows, columns, width, offset);
        Store(gateScaleGm, gateScale_, rows, columns / 32, width / 32, offset / 32);
        Store(upScaleGm, upScale_, rows, columns / 32, width / 32, offset / 32);
        if (saveMasks) {
            // One bit per column, uint32 words in increasing lane order.
            // GM is compact; each UB source row is padded to 32 bytes.
            Store(gateHighGm, gateHigh_, rows, columns / 8, width / 8, offset / 8);
            Store(upLowGm, upLow_, rows, columns / 8, width / 8, offset / 8);
            Store(upHighGm, upHigh_, rows, columns / 8, width / 8, offset / 8);
        }
        // These private quant buffers do not alias raw gate/up. SwiGLU may
        // run while MTE3 writes them. The caller's final MTE3 completion wait
        // retires data, scale and mask stores before the next tile reuses scratch.
    }

private:
    __aicore__ inline void ComputeClipMasks(AscendC::LocalTensor<bfloat16_t> gate, AscendC::LocalTensor<bfloat16_t> up,
                                            uint16_t rows, uint16_t columns, float limit)
    {
        // The b16 UNPACK load places 64 consecutive values in the low b16
        // of 64 b32 lanes. Cast ZERO preserves that order exactly.
        // StoreUnAlign<uint32_t>(MaskReg) packs each b32 lane's predicate
        // into one bit and advances by 8 bytes; four calls fill a 256-wide row.
        constexpr uint16_t lanes = 256U / sizeof(float);
        // Cast takes its trait by template reference; it needs static storage.
        static constexpr AscendC::Reg::CastTrait cast = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::UNKNOWN};
        auto g = (__ubuf__ bfloat16_t*)gate.GetPhyAddr();
        auto u = (__ubuf__ bfloat16_t*)up.GetPhyAddr();
        auto gh = (__ubuf__ uint32_t*)gateHigh_.GetPhyAddr();
        auto ul = (__ubuf__ uint32_t*)upLow_.GetPhyAddr();
        auto uh = (__ubuf__ uint32_t*)upHigh_.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> limitReg, negLimitReg;
            AscendC::Reg::Duplicate(limitReg, limit);
            AscendC::Reg::Duplicate(negLimitReg, -limit);
            for (uint16_t row = 0; row < rows; ++row) {
                auto ghRow = gh + row * 8;
                auto ulRow = ul + row * 8;
                auto uhRow = uh + row * 8;
                AscendC::Reg::UnalignRegForStore ghAlign, ulAlign, uhAlign;
                uint32_t remaining = columns;
                for (uint16_t col = 0; col < columns; col += lanes) {
                    auto valid = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::RegTensor<bfloat16_t> gBf, uBf;
                    AscendC::Reg::RegTensor<float> gFp, uFp;
                    AscendC::Reg::MaskReg gHigh, uLow, uHigh;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        gBf, g + row * columns + col);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        uBf, u + row * columns + col);
                    AscendC::Reg::Cast<float, bfloat16_t, cast>(gFp, gBf, valid);
                    AscendC::Reg::Cast<float, bfloat16_t, cast>(uFp, uBf, valid);
                    // FP32 limit: rounding it to BF16 changes boundary decisions.
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(gHigh, gFp, limitReg, valid);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::LT>(uLow, uFp, negLimitReg, valid);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(uHigh, uFp, limitReg, valid);
                    AscendC::Reg::StoreUnAlign<uint32_t>(ghRow, gHigh, ghAlign);
                    AscendC::Reg::StoreUnAlign<uint32_t>(ulRow, uLow, ulAlign);
                    AscendC::Reg::StoreUnAlign<uint32_t>(uhRow, uHigh, uhAlign);
                }
                AscendC::Reg::StoreUnAlignPost<uint32_t>(ghRow, ghAlign, 0);
                AscendC::Reg::StoreUnAlignPost<uint32_t>(ulRow, ulAlign, 0);
                AscendC::Reg::StoreUnAlignPost<uint32_t>(uhRow, uhAlign, 0);
            }
        }
    }

    __aicore__ inline void Store(GM_ADDR base, AscendC::LocalTensor<int8_t> src, uint16_t rows, uint16_t columns,
                                 uint32_t width, uint64_t offset)
    {
        AscendC::GlobalTensor<int8_t> dst;
        dst.SetGlobalBuffer((__gm__ int8_t*)base + offset);
        // Apply one cache policy to saved data, scales and clipping masks.
        dst.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::DataCopyExtParams p{rows, columns, 0, width - columns, 0};
        AscendC::DataCopyPad(dst, src, p);
    }

    __aicore__ inline void Quantize(AscendC::LocalTensor<bfloat16_t> src, AscendC::LocalTensor<int8_t> data,
                                    AscendC::LocalTensor<int8_t> scales, uint16_t rows, uint16_t columns)
    {
        uint32_t count = static_cast<uint32_t>(rows) * columns, groups = count / 32;
        auto input = (__ubuf__ bfloat16_t*)src.GetPhyAddr();
        auto maxAddr = (__ubuf__ uint16_t*)maxExp_.GetPhyAddr();
        auto scaleAddr = (__ubuf__ uint16_t*)compactScale_.GetPhyAddr();
        auto halfAddr = (__ubuf__ uint16_t*)halfScale_.GetPhyAddr();
        uint16_t loops = (count + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2);
        ComputeMaxExp(input, maxAddr, count, loops);
        ComputeScale(maxAddr, scaleAddr, halfAddr, groups, (groups + vlForHalfNumber_ - 1) / vlForHalfNumber_);
        ComputeDataForQuantTargetFp8(input, halfAddr, (__ubuf__ int8_t*)data.GetPhyAddr(), count, loops);
        // UB DMA sources need 32-byte rows. GM scales stay compact [row,F/32].
        auto packed = (__ubuf__ int8_t*)compactScale_.GetPhyAddr();
        auto padded = (__ubuf__ int8_t*)scales.GetPhyAddr();
        __VEC_SCOPE__
        {
            for (uint16_t r = 0; r < rows; ++r) {
                uint32_t remaining = columns / 32;
                auto mask = AscendC::Reg::UpdateMask<int8_t>(remaining);
                AscendC::Reg::RegTensor<int8_t> value;
                AscendC::Reg::UnalignReg align;
                auto address = packed + r * (columns / 32);
                AscendC::Reg::DataCopyUnAlignPre(align, address);
                AscendC::Reg::DataCopyUnAlign(value, align, address);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(padded + r * 32, value, mask);
            }
        }
    }
    __aicore__ inline void ComputeMaxExp(__ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr,
                                         uint32_t totalCountInUB, uint16_t loopNum)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<bfloat16_t> vdExp0;
            AscendC::Reg::RegTensor<bfloat16_t> vdExp1;
            AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16;
            AscendC::Reg::RegTensor<bfloat16_t> vdExp1BF16;
            AscendC::Reg::RegTensor<uint16_t> vdExpExtract0;
            AscendC::Reg::RegTensor<uint16_t> vdExpExtract1;

            AscendC::Reg::RegTensor<uint16_t> expMaskBF16;
            AscendC::Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

            AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
            AscendC::Reg::MaskReg scaleMask1;
            AscendC::Reg::MaskReg scaleMask2;
            AscendC::Reg::UnalignReg u1;
            static constexpr AscendC::Reg::CastTrait castTraitHalf2Bf16 = {
                AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::CAST_TRUNC};
            for (uint16_t i = 0; i < loopNum; i++) {
                scaleMask1 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
                scaleMask2 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                    vdExp0, vdExp1, srcAddr,
                    vlForHalfNumber_ * 2);  // copy two chunks from srcAddr to regbase
                AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);
                AscendC::Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
                AscendC::Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

                AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
            }
            AscendC::Reg::DataCopyUnAlignPost(maxExpAddr, u1, 0);
        }
        return;
    }

    __aicore__ inline void ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                        __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale)  // 128*8  8
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
            AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);  // MAX_EXP_FOR_BF16表示bf16正无穷 大小：128
            AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
            AscendC::Reg::RegTensor<bfloat16_t> vdExp0, vdExp1;
            AscendC::Reg::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
            AscendC::Reg::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
            AscendC::Reg::Duplicate(maxExpValue, fpEmax_);      // 0x0780 大小：128 对应bf16指数位后四位
            AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);  // 0x7f00 大小：128
            AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);  // 0x00ff 大小：128
            AscendC::Reg::Duplicate(zeroRegTensor, 0);                  // 0 大小：128
            AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);   // 0x7f81 大小：128
            AscendC::Reg::MaskReg invalidDataMask, specialDataMask;
            AscendC::Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);    // 0x0040 大小：128
            for (uint16_t i = 0; i < loopNumScale; i++) {                           // 8
                preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);  // 128*8
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    vdMaxExp, maxExpAddr, vlForHalfNumber_);  // 每次搬运128个数到vdMaxExp
                // 得到不等于INF的结果掩码 cmpResult
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask,
                                                                      preMaskScale);  // INF/NAN
                // 得到不等于0的结果掩码 zeroMask
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
                // 得到小于或等于0x0780的结果掩码 invalidDataMask
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                      preMaskScale);
                // 将vdMaxExp中小于或等于0x0780的结果替换成0x0780
                AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
                AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);  // sharedExp = vdMaxExp - 0x0780
                AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
                // 逻辑右移7位 当前指数位在减去0x0780后，已移至最低位
                // 将scaleValue中INF的结果替换成0x00ff
                AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
                // 将scaleValue中原来是0的结果替换成0
                AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                               vlForHalfNumber_ >> 1, preMaskScale);
                // 将scaleValue中数取低半部分，搬运到mxScaleLocalAddr uint16--int8

                // 得到sharedExp等于0x7f00的结果掩码 specialDataMask
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                                                                      preMaskScale);
                AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);  // halfScale = 0x7f00 - sharedExp
                // 将halfScale中原等于INF的数值替换成0x7f81
                AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
                // 将halfScale中原等于0的数值替换成0
                AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
                // 将halfScale中原等于0x7f00的数值替换成0x0040
                AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    halfScaleLocalAddr, halfScale, vlForHalfNumber_, preMaskScale);
                // 将128个数搬运到halfScaleLocalAddr uint16--uint16
            }
        }
        return;
    }

    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t* srcAddr,
                                                        __ubuf__ uint16_t* halfScaleLocalAddr,
                                                        __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum)
    {
        uint32_t totalCountInUB2 = totalCountInUB * 2;  // 128*256*2
        using T = bfloat16_t;
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
            AscendC::Reg::MaskReg maskAll =
                AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();  // 全1
            AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
            AscendC::Reg::RegTensor<float> floatScaleForMul;
            AscendC::Reg::RegTensor<T> vdExp0, vdExp1, vdExp0Convert, vdExp1Convert;
            AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
            AscendC::Reg::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
            AscendC::Reg::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;
            AscendC::Reg::RegTensor<bfloat16_t> vdBf16Exp0FP4, vdBf16Exp1FP4;

            static constexpr AscendC::Reg::CastTrait castTraitZero = {
                // src与dst类型位宽不同时,写入索引0的位置，其余位置置零
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN};
            static constexpr AscendC::Reg::CastTrait castTraitOne = {
                // src与dst类型位宽不同时,写入索引1的位置，其余位置置零
                AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN};
            static constexpr AscendC::Reg::CastTrait castTrait32to8 = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::CAST_RINT};
            // 当结果超出范围时，截断到最大值或最小值
            // 返回最接近参数的整数，如果有2个整数同样接近，则会返回偶数的那个
            for (uint16_t i = 0; i < loopNum; i++) {                       // 128
                dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);   // 128*256
                dataMask2 = AscendC::Reg::UpdateMask<T>(totalCountInUB);   // 128*256
                dataMask3 = AscendC::Reg::UpdateMask<T>(totalCountInUB2);  // 128*256*2
                dataMask4 = AscendC::Reg::UpdateMask<T>(totalCountInUB2);  // 128*256*2
                // DIST_DINTLV_B16:双搬入模式，基于元素的交错搬运，从src中读取2*VL长度数据，将偶数索引的元素存入dst0，将奇数索引的元素存入dst1
                AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                    vdExp0, vdExp1, srcAddr,
                    vlForHalfNumber_ * 2);  // copy two chunks from srcAddr to regbase 128*256个数
                // 将halfScale中的8个数uint16广播到halfScaleForMul中，halfScale[0]*16 halfScale[1]*16...
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                             elementAfterReduce_);
                // vdExp0*halfScaleForMul
                AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
                // vdExp1*halfScaleForMul
                AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
                // 交叉vdExp0、vdExp1数据，并将前后分别存入vdExp0、vdExp1
                AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);  // 交叉数据
                AscendC::Reg::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0,
                                                            dataMask1);  // vdExp0转float,值在前半部分
                AscendC::Reg::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0,
                                                           dataMask1);  // vdExp0转float，值在后半部分
                // 交叉vdExp0FP32Zero、vdExp0FP32One数据，并将前后分别存入vdExp0FP32Zero、vdExp0FP32One
                AscendC::Reg::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                AscendC::Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                // vdExp0FP32Zero转int8
                AscendC::Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
                // vdExp0FP32One转int8
                AscendC::Reg::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1,
                                                            dataMask2);  // vdExp1转float，前半部分值
                AscendC::Reg::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1,
                                                           dataMask2);  // vdExp1转float，后半部分值
                // 交叉vdExp1FP32Zero、vdExp1FP32One数据，并将前后分别存入vdExp1FP32Zero、vdExp1FP32One
                AscendC::Reg::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                AscendC::Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                // vdExp1FP32Zero转int8
                AscendC::Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
                // vdExp1FP32One转int8
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    // 将src中有效元素的低8bit（四分之一）数据连续存储于dst中
                    outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
                // 64个数
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
            }
        }
        return;
    }
};
}  // namespace MegaMoeBF16Impl
