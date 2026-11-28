// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <cstdint>
#include "catlass/catlass.hpp"
#include "shmem.h"

namespace Catccos::Comm::Tile {

CATLASS_HOST_DEVICE
bool UseStripedPeer(uint32_t self, uint32_t peer, uint32_t ranksPerNode)
{
    return ranksPerNode != 0 && self / ranksPerNode != peer / ranksPerNode;
}

// Source and destination must both be contiguous. Both QPs belong to the same
// remote peer and have a single posting AIV. QP0 gets any remainder after division by four.
template <bool Put>
CATLASS_DEVICE
void StripedRows(__gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t rows, uint32_t rowBytes, uint32_t peer)
{
    const uint32_t secondRows = rows / 4;
    const uint32_t firstBytes = (rows - secondRows) * rowBytes;
    const uint32_t secondBytes = secondRows * rowBytes;
    auto scratch = static_cast<__ubuf__ uint8_t*>(nullptr);
    if (firstBytes != 0) {
        if constexpr (Put) {
            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_S>(dst, src, scratch, firstBytes, peer, 0, 0);
        } else {
            aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_S>(dst, src, scratch, firstBytes, peer, 0, 0);
        }
    }
    if (secondBytes != 0) {
        if constexpr (Put) {
            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_S>(dst + firstBytes, src + firstBytes, scratch, secondBytes, peer, 1, 0);
        } else {
            aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_S>(dst + firstBytes, src + firstBytes, scratch, secondBytes, peer, 1, 0);
        }
    }
}

CATLASS_DEVICE
void StripedPutRows(__gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t rows, uint32_t rowBytes, uint32_t peer)
{
    StripedRows<true>(dst, src, rows, rowBytes, peer);
}

CATLASS_DEVICE
void StripedGetRows(__gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t rows, uint32_t rowBytes, uint32_t peer)
{
    StripedRows<false>(dst, src, rows, rowBytes, peer);
}

CATLASS_DEVICE
void FinishStripedPeer(uint32_t peer)
{
    aclshmemx_udma_qp_quiet(peer, 0);
    aclshmemx_udma_qp_quiet(peer, 1);
}

}  // namespace Catccos::Comm::Tile
