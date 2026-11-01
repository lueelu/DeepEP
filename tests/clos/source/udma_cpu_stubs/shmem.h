#pragma once
#include "kernel_operator.h"
#include <vector>
#include <map>
#include <set>
enum { ACLSHMEMX_UDMA_ODR_RO, ACLSHMEMX_UDMA_ODR_SO, ACLSHMEM_CMP_EQ };
struct aclshmemx_udma_op_config_t {
    unsigned cqe, a, b, order;
};
struct aclshmemx_submit_state_t {
    unsigned pending_count = 0;
};
struct aclshmemx_defer_t {
    aclshmemx_submit_state_t& state;
    explicit aclshmemx_defer_t(aclshmemx_submit_state_t& s) : state(s) {}
};
struct aclshmemx_submit_t {
    aclshmemx_submit_state_t& state;
    explicit aclshmemx_submit_t(aclshmemx_submit_state_t& s) : state(s) {}
};
struct aclshmemi_udma_queue_state_t {
    unsigned reserved = 0;
};
namespace shm {
inline constexpr unsigned UDMA_SQ_BASKBLK_CNT = 32768;
}
inline constexpr unsigned ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE = 64;
enum class aclshmemi_udma_opcode_t { UDMA_OP_WRITE };
struct aclshmemi_sqe_ctx_t {
    uint64_t words[6];
};
struct aclshmemi_sge_ctx_t {
    uint64_t words[2];
};
struct aclshmemi_udma_wq_ctx_t {
    uint64_t state_addr = 0, buf_addr = 1;
    unsigned wqe_size = 64, depth = 32768;
};
using QpContext = aclshmemi_udma_wq_ctx_t;
struct MemoryInfo {
    unsigned peer, qp;
};
struct Info {
    unsigned qp_num = 32;
};
namespace model {
struct Request {
    uint8_t* dst;
    uint8_t* src;
    uint32_t bytes, peer, qp, order;
};
inline Info info;
inline unsigned alias_groups = 1;
inline aclshmemi_udma_queue_state_t states[32][3];
inline QpContext contexts[257][32];
inline MemoryInfo memory[257][32];
inline bool corrupt_wqe = false;
inline bool direct_output = false;
inline std::function<void(unsigned, size_t)> on_metadata_store;
inline uint64_t decoded_wqes = 0;
inline std::vector<Request> staged, sent;
inline std::vector<unsigned> batches;
inline std::vector<std::pair<unsigned, unsigned>> quiesced;
inline unsigned active_qp = 0;
inline unsigned active_qp_count = 1;
inline unsigned cross_syncs = 0;
inline uint8_t* local_workspace = nullptr;
inline size_t workspace_bytes = 0;
inline std::vector<uint8_t*> remote_workspaces;
inline std::set<std::pair<unsigned, size_t>> metadata_stores;
inline size_t metadata_store_bytes = 0;
inline void reset(unsigned groups)
{
    alias_groups = groups;
    active_qp_count = 1;
    staged.clear();
    sent.clear();
    batches.clear();
    quiesced.clear();
    for (unsigned p = 0; p < 257; ++p)
        for (unsigned q = 0; q < 32; ++q) {
            contexts[p][q].state_addr = reinterpret_cast<uint64_t>(&states[q][p % groups]);
            memory[p][q] = {p, q};
        }
}
inline uint64_t state(const Request& r)
{
    return contexts[r.peer][r.qp].state_addr;
}
template <class T, const aclshmemx_udma_op_config_t& C>
Request request(T* d, T* s, uint32_t n, uint32_t p, uint32_t q)
{
    CHECK(q >= active_qp && q < active_qp + active_qp_count && n > 0);
    return {reinterpret_cast<uint8_t*>(d), reinterpret_cast<uint8_t*>(s), n * uint32_t(sizeof(T)), p, q, C.order};
}
}  // namespace model
inline Info* aclshmemi_udma_qp_info_fetch()
{
    return &model::info;
}
inline unsigned aclshmemi_udma_compute_slot(unsigned p, unsigned)
{
    return p;
}
inline QpContext* aclshmemi_udma_get_qp_ctx(Info*, unsigned slot, unsigned qp)
{
    CHECK(qp < 32);
    return &model::contexts[slot][qp];
}
inline MemoryInfo* aclshmemi_udma_get_mem_info(Info*, unsigned slot, unsigned qp)
{
    return &model::memory[slot][qp];
}
// Opaque, peer/QP-specific template values make copying the wrong header
// detectable independently of the routing oracle and of production bitfields.
template <class T, aclshmemi_udma_opcode_t, class P, const aclshmemx_udma_op_config_t& C>
void aclshmemi_udma_fill_sqe_base_ctx(P header, uint8_t* remote, MemoryInfo* m, unsigned)
{
    auto* w = reinterpret_cast<uint64_t*>(header);
    w[0] = 0xa0b0c00000000000ULL | (uint64_t(m->peer) << 16) | (m->qp << 1) | C.cqe;
    w[1] = 0x1020304000000000ULL + m->peer;
    w[2] = 0x5678000000000000ULL + m->qp;
    w[3] = 0x9abc000000000000ULL + m->peer * 33 + m->qp;
    w[4] = 0x1234567800000000ULL + m->peer * 17;
    w[5] = reinterpret_cast<uint64_t>(remote);
}
inline void aclshmemi_udma_poll_cq_when_sq_overflow(aclshmemi_udma_queue_state_t* s, unsigned, unsigned, unsigned n)
{
    CHECK(n == 32 && model::staged.empty());
    s->reserved = n;
}
inline bool aclshmemi_udma_sq_has_credit(aclshmemi_udma_queue_state_t* s, unsigned n)
{
    return s->reserved >= n;
}
template <bool>
void aclshmemi_sync_npu_v4(unsigned)
{
    AscendC::CheckDrained();
    ++model::cross_syncs;
}
inline void* aclshmem_ptr(uint8_t* ptr, unsigned peer)
{
    // Sender arithmetic tests do not dereference remote addresses. Distinct
    // virtual bases still verify the per-peer symmetric-address translation.
    if (!model::local_workspace)
        return reinterpret_cast<void*>(reinterpret_cast<uint64_t>(ptr) + (uint64_t(peer + 1) << 40));
    CHECK(model::local_workspace && ptr >= model::local_workspace);
    const size_t offset = ptr - model::local_workspace;
    CHECK(offset < model::workspace_bytes && peer < model::remote_workspaces.size());
    return model::remote_workspaces[peer] + offset;
}
inline void aclshmemi_udma_flush_aggregate_wqes(uint8_t* ub, QpContext* context, unsigned slot, unsigned qp, unsigned,
                                                unsigned count, unsigned size)
{
    CHECK(count > 0 && count <= 32 && size == 64 && qp >= model::active_qp &&
          qp < model::active_qp + model::active_qp_count);
    CHECK(context == &model::contexts[slot][qp] && model::staged.empty());
    auto* words = reinterpret_cast<uint64_t*>(ub);
    if (model::corrupt_wqe) words[4] ^= 1;
    for (unsigned i = 0; i < count; ++i) {
        auto* w = words + i * 8;
        const unsigned peer = (w[0] >> 16) & 0xffff;
        CHECK(peer < 257 && ((w[0] >> 1) & 31) == qp);
        CHECK(w[0] == (0xa0b0c00000000000ULL | (uint64_t(peer) << 16) | (qp << 1) | unsigned(i + 1 == count)));
        CHECK(w[1] == 0x1020304000000000ULL + peer);
        CHECK(w[2] == 0x5678000000000000ULL + qp);
        CHECK(w[3] == 0x9abc000000000000ULL + peer * 33 + qp);
        CHECK(w[4] == 0x1234567800000000ULL + peer * 17);
        CHECK(model::contexts[peer][qp].state_addr == context->state_addr);
        CHECK(w[6] > 0 && w[6] <= UINT32_MAX);
        auto* dst = reinterpret_cast<uint8_t*>(
            w[5] - (model::direct_output || model::local_workspace ? 0 : (uint64_t(peer + 1) << 40)));
        model::sent.push_back({dst, reinterpret_cast<uint8_t*>(w[7]), unsigned(w[6]), peer, qp, ACLSHMEMX_UDMA_ODR_RO});
        ++model::decoded_wqes;
    }
    model::batches.push_back(count);
}
namespace simt {
template <class T>
void scalar_put(T* ptr, T value, unsigned peer)
{
    const size_t offset = reinterpret_cast<uint8_t*>(ptr) - model::local_workspace;
    CHECK(offset + sizeof(T) <= model::workspace_bytes);
    CHECK(model::metadata_stores.emplace(peer, offset).second);
    auto* dst = static_cast<uint8_t*>(::aclshmem_ptr(reinterpret_cast<uint8_t*>(ptr), peer));
    std::memcpy(dst, &value, sizeof(T));
    model::metadata_store_bytes += sizeof(T);
    if (model::on_metadata_store) model::on_metadata_store(peer, offset);
}
inline void aclshmem_int32_p(int32_t* p, int32_t v, int pe)
{
    scalar_put(p, v, pe);
}
inline void aclshmem_int64_p(int64_t* p, int64_t v, int pe)
{
    scalar_put(p, v, pe);
}
inline void aclshmem_uint64_p(uint64_t* p, uint64_t v, int pe)
{
    scalar_put(p, v, pe);
}
inline void aclshmem_float_p(float* p, float v, int pe)
{
    scalar_put(p, v, pe);
}
}  // namespace simt
inline void aclshmem_uint64_wait_until(uint64_t*, int, unsigned) {}
inline void aclshmemx_udma_qp_quiet(unsigned p, unsigned q)
{
    CHECK(q >= model::active_qp && q < model::active_qp + model::active_qp_count && model::staged.empty());
    model::quiesced.emplace_back(p, q);
}
template <class T, int, const aclshmemx_udma_op_config_t& C>
void aclshmemx_udma_qp_put_nbi(T* d, T* s, T*, unsigned n, unsigned p, unsigned q, unsigned, aclshmemx_defer_t a)
{
    CHECK(C.cqe == 0 && a.state.pending_count == model::staged.size());
    auto r = model::request<T, C>(d, s, n, p, q);
    CHECK(model::staged.empty() || model::state(r) == model::state(model::staged.front()));
    model::staged.push_back(r);
    ++a.state.pending_count;
}
template <class T, int, const aclshmemx_udma_op_config_t& C>
void aclshmemx_udma_qp_put_nbi(T* d, T* s, T*, unsigned n, unsigned p, unsigned q, unsigned, aclshmemx_submit_t a)
{
    CHECK(C.cqe == 1 && a.state.pending_count == model::staged.size() && a.state.pending_count < 32);
    auto r = model::request<T, C>(d, s, n, p, q);
    CHECK(model::staged.empty() || model::state(r) == model::state(model::staged.front()));
    model::staged.push_back(r);
    model::batches.push_back(model::staged.size());
    model::sent.insert(model::sent.end(), model::staged.begin(), model::staged.end());
    model::staged.clear();
    a.state.pending_count = 0;
}
template <class T, int, const aclshmemx_udma_op_config_t& C>
void aclshmemx_udma_qp_put_nbi(T* d, T* s, T*, unsigned n, unsigned p, unsigned q, unsigned)
{
    CHECK(C.cqe == 1 && C.order == ACLSHMEMX_UDMA_ODR_SO && model::staged.empty());
    model::sent.push_back(model::request<T, C>(d, s, n, p, q));
}
