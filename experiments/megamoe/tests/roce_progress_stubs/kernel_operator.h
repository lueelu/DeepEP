// Copyright (c) 2026, Lu Lu

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#define __aicore__
#define __gm__
#define __ubuf__

namespace RoceProgressTest {
inline uint32_t copies = 0U;
inline uint32_t syncs = 0U;
inline uint8_t* refreshed = nullptr;
inline bool copyPending = false;
}  // namespace RoceProgressTest

inline void trap()
{
    throw std::runtime_error("CQE failure");
}

inline void dcci_cachelines(uint8_t* address, uint32_t bytes)
{
    assert(bytes == 32U);
    RoceProgressTest::refreshed = address;
}

namespace AscendC {
enum class TPosition : uint8_t { VECOUT };
enum class HardEvent { MTE2_S };

template <typename T>
struct LocalTensor {
    struct {
        uint8_t logicPos = 0U;
        uint64_t bufferAddr = 0U;
        uint32_t dataLen = 0U;
    } address_;

    T* GetPhyAddr() const
    {
        return reinterpret_cast<T*>(address_.bufferAddr);
    }
};

template <typename T>
struct GlobalTensor {
    T* data = nullptr;
    void SetGlobalBuffer(T* address)
    {
        data = address;
    }
};

struct DataCopyExtParams {
    uint32_t blockCount;
    uint32_t blockLen;
    uint32_t srcStride;
    uint32_t dstStride;
    uint32_t reserved;
};

template <typename T>
struct DataCopyPadExtParams {
    bool enabled;
    uint32_t left;
    uint32_t right;
    T value;
};

template <typename T>
void DataCopyPad(LocalTensor<T>& dst, GlobalTensor<T>& src, DataCopyExtParams copy, DataCopyPadExtParams<T>)
{
    assert(!RoceProgressTest::copyPending);
    assert(RoceProgressTest::refreshed == reinterpret_cast<uint8_t*>(src.data));
    assert(copy.blockCount == 1U && copy.blockLen == 32U);
    assert(copy.blockLen <= dst.address_.dataLen);
    std::memcpy(dst.GetPhyAddr(), src.data, copy.blockLen);
    RoceProgressTest::refreshed = nullptr;
    RoceProgressTest::copyPending = true;
    ++RoceProgressTest::copies;
}
}  // namespace AscendC
