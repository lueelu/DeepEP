#pragma once

#include <array>
#include <charconv>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/udma_schedule.h"

namespace ascend_deepep::udma {

// Host-only, immutable reservation for a known set of peer schedules. Visit,
// dtype and copy/weight modes do not change the QP ownership mapping.
class QpReservation {
public:
    QpReservation() = default;
    QpReservation(const std::string& spec, uint32_t rank, uint32_t world, uint32_t group)
    {
        if (spec.empty()) return;
        if (!world || world > 257 || rank >= world || !group || group > world || world % group)
            throw std::invalid_argument("QP reservation requires a valid world and fixed group size");
        masks_.resize(world, 0);
        std::size_t begin = 0;
        while (begin < spec.size()) {
            const auto end = spec.find(';', begin);
            const auto item = spec.substr(begin, end == std::string::npos ? end : end - begin);
            const auto colon = item.find(':');
            const std::array<std::string, 4> names = {"rank", "group", "group_staggered", "group_round_robin"};
            uint32_t order = 0;
            while (order < names.size() && names[order] != item.substr(0, colon)) ++order;
            uint32_t values[3]{};
            std::size_t offset = colon == std::string::npos ? item.size() : colon + 1;
            for (unsigned i = 0; i < 3; ++i) {
                const auto next = item.find(':', offset);
                const char* first = item.data() + offset;
                const char* last = item.data() + (next == std::string::npos ? item.size() : next);
                const auto parsed = std::from_chars(first, last, values[i]);
                if (parsed.ec != std::errc{} || parsed.ptr != last || (i < 2) != (next != std::string::npos))
                    throw std::invalid_argument("DEEPEP_CLOS_QP_PLANS expects order:P:W:Q entries separated by semicolons");
                offset = next == std::string::npos ? item.size() : next + 1;
            }
            const auto p = values[0], w = values[1], q = values[2];
            if ((w != 1 && w != 2) || (q != 1 && q != 2) || w * q > 2 || !Valid(world, group, order, p, w * q))
                throw std::invalid_argument("QP reservation contains an unsupported schedule");
            for (uint32_t peer = 0; peer < world; ++peer)
                masks_[peer] |= Mask(rank, peer, world, group, order, p, w * q);
            if (end == std::string::npos) break;
            begin = end + 1;
            if (begin == spec.size()) throw std::invalid_argument("empty QP reservation entry");
        }
        // Compile coverage once. Launch validation then costs a single bit test,
        // including when different schedules happen to use the same physical QPs.
        for (uint32_t order = 0; order < 4; ++order)
            for (uint32_t p = 1; p <= kMaxRankPeers; p *= 2)
                for (uint32_t lanes = 1; lanes <= 2; ++lanes) {
                    if (!Valid(world, group, order, p, lanes)) continue;
                    bool covered = true;
                    for (uint32_t peer = 0; peer < world; ++peer)
                        covered &= (Mask(rank, peer, world, group, order, p, lanes) & ~masks_[peer]) == 0;
                    if (covered) allowed_ |= Bit(order, p, lanes);
                }
    }

    const std::vector<uint64_t>& Masks() const { return masks_; }
    bool Allows(uint32_t order, uint32_t p, uint32_t w, uint32_t q) const
    {
        if (masks_.empty()) return true;
        if (order > 3 || !ValidPeers(p) || (w != 1 && w != 2) || (q != 1 && q != 2) || w * q > 2) return false;
        return (allowed_ & Bit(order, p, w * q)) != 0;
    }

private:
    static bool Valid(uint32_t world, uint32_t group, uint32_t order, uint32_t p, uint32_t lanes)
    {
        if (order > 3 || !ValidPeers(p)) return false;
        if (p > kMaxPeers && order != kRankOrder) return false;
        const auto owners = Workers(world, p, order, group);
        return owners * lanes <= kQpCount &&
            (order != kGroupRoundRobinOrder ||
             (p <= 4 && ((group + owners - 1) / owners) * (world / group) <= kGroupPeerCapacity));
    }
    static uint64_t Bit(uint32_t order, uint32_t p, uint32_t lanes)
    {
        const uint32_t slot = p == 1 ? 0 : p == 2 ? 1 : p == 4 ? 2 : p == 8 ? 3 : 4;
        return uint64_t{1} << ((order * 5 + slot) * 2 + lanes - 1);
    }
public:
    static constexpr uint64_t Mask(uint32_t rank, uint32_t peer, uint32_t world, uint32_t group,
                         uint32_t order, uint32_t p, uint32_t lanes)
    {
        if (rank == peer) return 0;
        // QP 0 is also used by SDK control traffic. Both endpoints must create
        // the union of their outgoing mappings, not just the local sender's QP.
        const auto bits = (uint64_t{1} << lanes) - 1;
        return 1 | (bits << (Owner(rank, peer, world, p, order, group) * lanes)) |
            (bits << (Owner(peer, rank, world, p, order, group) * lanes));
    }
private:
    std::vector<uint64_t> masks_;
    uint64_t allowed_ = 0;
};

} // namespace ascend_deepep::udma
