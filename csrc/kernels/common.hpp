// Copyright (c) 2026, Lu Lu
// Modified by lishaoxun 2026

#pragma once
#include <cstdint>

// Scheduling helpers shared by Notify and Combine.
// MoonEP's Legacy half-world mapping for EP<=64 and EP256, and Staged 4x4
// two-frame mapping for EP128. Keep pure arithmetic host-testable.
namespace first_hit_schedule {
__aicore__ inline constexpr uint32_t Width(uint32_t world)
{
    return world < 16U ? world : 16U;
}
__aicore__ inline constexpr uint32_t Index(uint32_t rank, uint32_t world)
{
    return world == 128U ? (rank / 8U % 4U) * 4U + rank % 4U
                         : (rank / (world / 2U)) * (Width(world) / 2U) + rank % (Width(world) / 2U);
}
__aicore__ inline constexpr uint32_t Peer(uint32_t rank, uint32_t world, uint32_t group, uint32_t index,
                                          bool incoming = false)
{
    if (world == 128U) {
        const uint32_t toggle = group / 4U, subround = group % 4U;
        const uint32_t row = rank % 64U / 32U;
        const uint32_t side = rank % 8U / 4U;
        const uint32_t peer_side = side ^ (subround % 2U);
        const uint32_t peer_row = (row + subround / 2U) % 2U;
        const uint32_t frame = rank / 64U ^ (incoming ? peer_side : side) ^ toggle;
        return frame * 64U + (peer_row * 4U + index / 4U) * 8U + peer_side * 4U + index % 4U;
    }
    const uint32_t half = world / 2U, half_width = Width(world) / 2U;
    const uint32_t groups = world / Width(world), own = rank % half / half_width;
    const uint32_t peer_group = (own + (incoming ? groups - group : group)) % groups;
    return index / half_width * half + peer_group * half_width + index % half_width;
}
// Peer 的出向逆映射：expert 发送 backward 时必须传目的 gateway rank。
__aicore__ inline constexpr uint32_t Group(uint32_t rank, uint32_t world, uint32_t peer)
{
    if (world == 128U) {
        const uint32_t side = rank % 8U / 4U, peer_side = peer % 8U / 4U;
        const uint32_t row = rank % 64U / 32U, peer_row = peer % 64U / 32U;
        const uint32_t toggle = rank / 64U ^ peer / 64U ^ side;
        return toggle * 4U + (row ^ peer_row) * 2U + (side ^ peer_side);
    }
    const uint32_t half = world / 2U, half_width = Width(world) / 2U;
    const uint32_t groups = world / Width(world), own = rank % half / half_width;
    return (peer % half / half_width + groups - own) % groups;
}
// Combine aggregates the existing topology order into groups of up to 64.
__aicore__ inline constexpr uint32_t CombineGroup(uint32_t rank, uint32_t world, uint32_t peer)
{
    return (Group(rank, world, peer) * Width(world) + Index(peer, world)) / (world < 64U ? world : 64U);
}
}  // namespace first_hit_schedule
