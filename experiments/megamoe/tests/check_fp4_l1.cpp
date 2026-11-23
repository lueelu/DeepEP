// Copyright (c) 2026, Lu Lu
// Check the packed FP4 bytes after the production VF and UB-to-L1 gather on Ascend950.
#include "basic_api/kernel_basic_intf.h"
#if defined(__NPU_DEVICE__)
#include "kernels/mega_moe/catlass_gmm/block_prologue_a8w4.hpp"
#else
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#endif

constexpr unsigned slots = 4, inputSlot = 16384, outputSlot = inputSlot * 2;
__global__ __mix__(1, 2) void check_fp4(GM_ADDR input, GM_ADDR output, uint16_t vectors)
{
#if defined(__NPU_DEVICE__)
    using namespace AscendC;
    using namespace Catlass::Gemm::Block;
    InitSocState();
    MegaMoeResource mem;
    GlobalTensor<int8_t> packed;
    packed.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(input));
    auto l1 = mem.l1Buf.GetBufferByByte<int8_t>(0);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() != 0) return;
        for (unsigned s = 0; s < slots; ++s) {
            auto in = mem.ubBuf.GetBufferByByte<int8_t>(s * inputSlot);
            auto out = mem.ubBuf.GetBufferByByte<int8_t>(slots * inputSlot + s * 256);
            DataCopy(in, packed[s * inputSlot], vectors * 128U);
            PipeBarrier<PIPE_ALL>();
            PackedNzFp4::Decode(reinterpret_cast<__ubuf__ uint8_t*>(in.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ uint8_t*>(out.GetPhyAddr()), vectors);
        }
        PipeBarrier<PIPE_ALL>();
        for (unsigned s = 0; s < slots; ++s) {
            auto out = mem.ubBuf.GetBufferByByte<int8_t>(slots * inputSlot + s * 256);
            PackedNzFp4::StoreL1(l1[s * outputSlot], out, vectors);
        }
        CrossCoreSetFlag<4, PIPE_MTE3>(6);
    } else {
        // A signed-byte identity GEMM makes every decoded L1 byte observable in GM, including FP8 -0.
        auto identity = l1[slots * outputSlot];
        DataCopy(identity, packed[slots * inputSlot], 32U * 32U);
        PipeBarrier<PIPE_ALL>();
        auto b = mem.l0BBuf.GetBufferByByte<int8_t>(0);
        LoadData2DParamsV2 load{};
        load.kStep = 1;
        load.mStep = load.srcStride = load.dstStride = 2;
        LoadData(b, identity, load);
        CrossCoreWaitFlag<4, PIPE_MTE1>(6);
        auto a = mem.l0ABuf.GetBufferByByte<int8_t>(0);
        auto c = mem.l0CBuf.GetBufferByByte<int32_t>(0);
        GlobalTensor<int32_t> result;
        result.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
        unsigned blocks = (vectors + 1U) / 2U;
        load.mStep = load.srcStride = load.dstStride = blocks;
        MmadParams mm{};
        mm.m = blocks * 16U;
        mm.k = mm.n = 32;
        mm.cmatrixInitVal = true;
        DataCopyCO12DstParams store{};
        store.mSize = store.srcStride = mm.m;
        store.nSize = store.dstStride = 32;
        store.quantPre = QuantMode_t::NoQuant;
        store.nz2ndEn = true;
        SetFixpipeNz2ndFlag(1, 1, 1);
        for (unsigned s = 0; s < slots; ++s) {
            LoadData(a, l1[s * outputSlot], load);
            PipeBarrier<PIPE_ALL>();
            Mmad(c, a, b, mm);
            PipeBarrier<PIPE_ALL>();
            DataCopy(result[s * outputSlot], c, store);
            PipeBarrier<PIPE_ALL>();
        }
    }
#endif
}

#if !defined(__NPU_DEVICE__)
#define CHECK(call) do { if ((call) != ACL_SUCCESS) { std::fprintf(stderr, "%s failed\n", #call); return 1; } } while (0)
int main(int argc, char** argv)
{
    int device = argc > 1 ? std::atoi(argv[1]) : 0;
    std::vector<uint8_t> input(slots * inputSlot + 32 * 32);
    std::vector<int32_t> output(slots * outputSlot);
    for (unsigned i = 0; i < slots * inputSlot; ++i) input[i] = i * 17U + i / 128U + i / inputSlot * 53U;
    for (unsigned i = 0; i < 32; ++i) input[slots * inputSlot + i * 33U] = 1;
    aclrtStream stream{};
    void *src{}, *dst{};
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(device));
    CHECK(aclrtCreateStream(&stream));
    CHECK(aclrtMalloc(&src, input.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    const size_t bytes = output.size() * sizeof(int32_t);
    CHECK(aclrtMalloc(&dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(src, input.size(), input.data(), input.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    for (unsigned vectors : {1U, 2U, 3U, 5U, 64U, 127U, 128U}) {
        CHECK(aclrtMemset(dst, bytes, 0x55, bytes));
        check_fp4<<<1, 0, stream>>>(static_cast<uint8_t*>(src), static_cast<uint8_t*>(dst), vectors);
        CHECK(aclrtSynchronizeStream(stream));
        CHECK(aclrtMemcpy(output.data(), bytes, dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        for (unsigned s = 0; s < slots; ++s) for (unsigned j = 0; j < vectors * 256U; ++j) {
            unsigned nibble = input[s * inputSlot + j / 2] >> (4U * (j % 2)) & 15U;
            int expected = static_cast<int8_t>(((nibble & 8U) << 4) | ((nibble & 7U) << 2));
            if (output[s * outputSlot + j] != expected) {
                std::fprintf(stderr, "%u vectors, slot %u, byte %u: %d != %d\n", vectors, s, j,
                             output[s * outputSlot + j], expected);
                return 1;
            }
        }
    }
    CHECK(aclrtFree(dst));
    CHECK(aclrtFree(src));
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(device));
    CHECK(aclFinalize());
    std::puts("Packed NZ FP4 bytes and L1 writeback matched in all four slots.");
}
#endif
