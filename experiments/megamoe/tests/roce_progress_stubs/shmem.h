// Copyright (c) 2026, Lu Lu

#pragma once

#include <array>
#include "kernel_operator.h"

#if !defined(ROCE_PROGRESS_TEST_HOST_PASS)

// Model only the pinned HNS metadata/ready/doorbell interfaces used by the
// adapter. These doubles check resource ownership and completion bookkeeping;
// they do not emulate NIC ordering or Ascend cache behavior.
enum class aclshmemi_rdma_backend_t { HNS_1825 };
struct aclshmemi_rdma_cq_ctx {
    uint64_t buf_addr;
    uint64_t tail_addr;
    uint32_t cqe_size;
    uint32_t depth;
};
struct aclshmemi_rdma_sq_ctx {
    uint64_t head_addr;
    uint64_t tail_addr;
};
struct aclshmemi_rdma_info {
    uint32_t qp_num;
    uint64_t scq_ptr;
    uint64_t sq_ptr;
};
struct aclshmemi_hns_1825_cqe_t {
    uint32_t owner_id_qpn;
    uint32_t op_sr_wqebb;
    uint32_t byte_cnt;
    uint32_t imm_data;
    uint32_t rsvd_dw5;
    uint32_t wqe_num;
    uint32_t vlan_queue_index;
    uint8_t syndrome;
    uint8_t rsvd;
    uint16_t wqe_counter;
};
static_assert(sizeof(aclshmemi_hns_1825_cqe_t) == 32U);

namespace RoceProgressTest {
constexpr uint32_t kPeer = 1U;
constexpr uint32_t kQp = 1U;
constexpr uint32_t kQpCount = 2U;
constexpr uint32_t kSlot = kPeer * kQpCount + kQp;
alignas(64) inline std::array<uint8_t, 256> scratch{};
alignas(64) inline std::array<uint8_t, 64U * 128U> ring{};
inline std::array<aclshmemi_rdma_sq_ctx, 4> sq{};
inline std::array<aclshmemi_rdma_cq_ctx, 4> cq{};
inline uint32_t head = 0U;
inline uint32_t sqTail = 0U;
inline uint32_t cqTail = 0U;
inline uint32_t doorbells = 0U;
inline uint32_t dumps = 0U;
inline uint32_t dumpedSyndrome = 0U;
inline aclshmemi_rdma_info info{};

inline void Reset(uint32_t initialTail = 0U, uint32_t depth = 8U, uint32_t stride = 64U)
{
    head = initialTail;
    sqTail = initialTail;
    cqTail = initialTail;
    copies = syncs = doorbells = dumps = dumpedSyndrome = 0U;
    refreshed = nullptr;
    copyPending = false;
    sq.fill({});
    cq.fill({});
    ring.fill(0xffU);
    scratch.fill(0xcdU);
    sq[kSlot].head_addr = reinterpret_cast<uint64_t>(&head);
    sq[kSlot].tail_addr = reinterpret_cast<uint64_t>(&sqTail);
    cq[kSlot].buf_addr = reinterpret_cast<uint64_t>(ring.data());
    cq[kSlot].tail_addr = reinterpret_cast<uint64_t>(&cqTail);
    cq[kSlot].cqe_size = stride;
    cq[kSlot].depth = depth;
    info.qp_num = kQpCount;
    info.scq_ptr = reinterpret_cast<uint64_t>(cq.data());
    info.sq_ptr = reinterpret_cast<uint64_t>(sq.data());
}

inline void Publish(uint32_t index, uint32_t opcode = 4U, bool readyOwner = true, uint8_t syndrome = 0U, uint32_t wqe = UINT32_MAX)
{
    const uint32_t depth = cq[kSlot].depth;
    const uint32_t stride = cq[kSlot].cqe_size == 0U ? 64U : cq[kSlot].cqe_size;
    aclshmemi_hns_1825_cqe_t entry{};
    const bool owner = ((index & depth) != 0U) == readyOwner;
    entry.owner_id_qpn = (static_cast<uint32_t>(owner) << 31U) | 23U;
    entry.op_sr_wqebb = opcode << 27U;
    entry.syndrome = syndrome;
    entry.wqe_counter = static_cast<uint16_t>(wqe == UINT32_MAX ? index : wqe);
    std::memcpy(ring.data() + (index & (depth - 1U)) * stride, &entry, sizeof(entry));
}
}  // namespace RoceProgressTest

inline aclshmemi_rdma_info* aclshmemi_qp_info_fetch()
{
    return &RoceProgressTest::info;
}

inline uint32_t aclshmemi_hns_1825_read_u32_gm(uint64_t address)
{
    return *reinterpret_cast<uint32_t*>(address);
}

inline void aclshmemi_hns_1825_write_u32_gm(uint64_t address, uint32_t value)
{
    *reinterpret_cast<uint32_t*>(address) = value;
}

template <AscendC::HardEvent Event>
inline void aclshmemi_hns_1825_sync_func(uint32_t syncId)
{
    static_assert(Event == AscendC::HardEvent::MTE2_S);
    assert(syncId == 7U && RoceProgressTest::copyPending);
    RoceProgressTest::copyPending = false;
    ++RoceProgressTest::syncs;
}

inline bool aclshmemi_roce_hns_1825_check_cqe_owner(aclshmemi_hns_1825_cqe_t* entry, uint32_t tail, uint32_t depth)
{
    assert(!RoceProgressTest::copyPending);
    const uint32_t owner = (entry->owner_id_qpn >> 31U) & 1U;
    const uint32_t expected = (tail & depth) == 0U;
    return (expected ^ owner) != 0U;
}

inline void aclshmemi_roce_hns_1825_dump_cqe(aclshmemi_hns_1825_cqe_t* entry)
{
    ++RoceProgressTest::dumps;
    RoceProgressTest::dumpedSyndrome = entry->syndrome;
}

template <aclshmemi_rdma_backend_t Backend>
inline void aclshmemi_roce_ring_cq_doorbell(uint32_t peer, uint32_t qp, uint32_t tail,
                                            AscendC::LocalTensor<uint64_t>& ub64, AscendC::LocalTensor<uint32_t>& ub32,
                                            uint32_t syncId)
{
    static_assert(Backend == aclshmemi_rdma_backend_t::HNS_1825);
    using namespace RoceProgressTest;
    assert(peer == kPeer && qp == kQp && syncId == 7U);
    assert(ub32.GetPhyAddr() == reinterpret_cast<uint32_t*>(scratch.data()));
    assert(ub64.GetPhyAddr() == reinterpret_cast<uint64_t*>(scratch.data() + 64U));
    assert(!copyPending);
    cqTail = tail;
    ++doorbells;
    std::memset(scratch.data(), 0xee, 64U);  // SDK reuses staging UB for its CI write.
}

#endif  // !ROCE_PROGRESS_TEST_HOST_PASS
