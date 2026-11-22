// Included by test_sparse_cqe.py with the actual sender credit/final methods.
#include <cstdint>
#define __aicore__ constexpr
using GM_ADDR = uint64_t;
constexpr unsigned kQpCount = 32, kBatchWqes = 32, kWqeBytes=64;
constexpr unsigned PIPE_MTE3 = 3, EVENT_ID5 = 5;
struct Config {};
constexpr Config kDone{};
enum class aclshmemi_udma_opcode_t { UDMA_OP_WRITE };
template<aclshmemi_udma_opcode_t> constexpr uint32_t get_wqe_bb_cnt() { return 1; }
struct State { uint32_t sq_head=0, sq_tail=0, cqe_cnt=0, cq_tail=0, completed_head=0; };
struct Context { uint64_t buf_addr=0; unsigned depth=1024; };
namespace AscendC {
enum class TPosition { VECOUT };
template<class T> struct LocalTensor { struct { uint8_t logicPos=0; uint64_t bufferAddr=0; } address_; };
}
constexpr bool aclshmemi_udma_sq_has_credit(State* s, uint32_t n) {
    return uint32_t(s->sq_head + n - s->sq_tail) + 10U < 1024U;
}
struct Trace {
    static constexpr bool enabled = false;
    enum Field { CreditQuietCalls, CqeRequested, CreditCheckpoints, FinalQuietCycles,
        FinalQuietCalls, CreditChecks, SqUsedSum, SqUsedMax, SqEmptyChecks,
        CreditBlocked, CreditCycles, CreditBlockedCycles, CreditMaxCycles,
        Wqes, Spans, Flushes, ShortFlushes, MetadataWqes, MetadataFlushes, SubmitNoCredit, SqWraps, WqeSubmitCycles };
    constexpr uint64_t Now() { return 0; }
    constexpr void Count(Field) {}
    constexpr void Accumulate(Field,uint64_t) {}
    constexpr void Maximum(Field,uint64_t) {}
};
template<unsigned Capacity, unsigned Qps>
struct Model {
    struct Peer { unsigned peer=0, state=0; Context* context=nullptr; };
    Peer peers_[Capacity*Qps]{};
    State states[Capacity*Qps]{};
    Context contexts[Capacity*Qps]{};
    unsigned count_=Capacity, qp_base_=4, rank_=2;
    GM_ADDR local_base_=0;
    uint64_t local[64]{}, remote[Capacity][64]{};
    uint64_t scratch[256]{}, pressure[8]{};
    uint64_t scratch_=0x4000;
    unsigned last_=0, copied_bytes=0, doorbells=0, wraps=0;
    unsigned checkpoints=0, cqes=0, credit_quiets=0, final_quiets=0;
    bool valid=true, shared=false;
    Trace* trace_=nullptr;
    constexpr explicit Model(bool aliases, uint32_t initial=0) : shared(aliases) {
        for(unsigned i=0;i<32;++i)local[i]=1;
        for(unsigned i=0;i<256;++i) scratch[i]=0x11223344;
        for(unsigned q=0;q<Qps;++q) for(unsigned p=0;p<Capacity;++p) {
            const unsigned index=q*Capacity+p, state=q*Capacity+(aliases?0:p);
            contexts[index].buf_addr=0x10000+state*1024*64;
            peers_[index]={p,state,&contexts[index]};
            states[index].sq_head=states[index].sq_tail=initial;
            states[index].completed_head=initial;
        }
    }
    constexpr void trap() { valid=false; }
    constexpr uint64_t TraceNow() { return 0; }
    constexpr void TraceAdd(Trace::Field,uint64_t) {}
    constexpr void TraceCount(Trace::Field f) {
        if(f==Trace::CreditCheckpoints) ++checkpoints;
        if(f==Trace::CqeRequested) ++cqes;
        if(f==Trace::CreditQuietCalls) ++credit_quiets;
        if(f==Trace::FinalQuietCalls) ++final_quiets;
        if(f==Trace::SqWraps) ++wraps;
    }
    constexpr void Flush() {}
    constexpr void aclshmemi_udma_copy_wqe_from_ub(
        uint64_t dst,AscendC::LocalTensor<uint8_t>& view,unsigned bytes,int event) {
        auto* ctx=peers_[last_].context;
        const auto& state=states[peers_[last_].state];
        const unsigned index=uint32_t(state.sq_head+copied_bytes/64)%1024;
        if(dst!=ctx->buf_addr+index*64 || view.address_.bufferAddr!=scratch_+copied_bytes ||
            bytes%64 || !bytes || index*64+bytes>1024*64 || event!=5)valid=false;
        copied_bytes+=bytes;
        if(copied_bytes>32*64)valid=false;
    }
    constexpr void aclshmemi_udma_post_send_update_info(unsigned head,Context* ctx,State* state) {
        if(ctx!=peers_[last_].context || uint32_t(head-state->sq_head)*64!=copied_bytes)valid=false;
        state->sq_head=head;
        copied_bytes=0;
        ++doorbells;
    }
    constexpr uint64_t* Signal(GM_ADDR,uint64_t offset,unsigned source) {
        if(offset || source!=rank_) valid=false;
        return local;
    }
    template<class T,int Pipe,const Config& C>
    constexpr void aclshmemx_udma_qp_put_nbi(
        T* dst,T* src,uint64_t* ub,unsigned count,unsigned peer,unsigned qp,int event) {
        auto& state=states[peers_[(qp-qp_base_)*Capacity+peer].state];
        const auto word=dst-local;
        if(!aclshmemi_udma_sq_has_credit(&state,1) || count!=1 || event!=5 || *src!=1) valid=false;
        if(ub==pressure) {
            if(word!=int(kQpCount+qp)) valid=false;
            // SDK writes only its independent 64-byte WQE scratch.
            for(unsigned i=0;i<8;++i) ub[i]=0x555;
        } else if(ub==scratch) {
            if(word<0 || word>=32) valid=false;
        } else valid=false;
        if(word>=0 && word<64) remote[peer][word]=*src;
        ++state.sq_head;
        ++state.cqe_cnt;
        state.completed_head=state.sq_head;
    }
    constexpr void aclshmemx_udma_qp_quiet(unsigned peer,unsigned qp) {
        auto& s=states[peers_[(qp-qp_base_)*Capacity+peer].state];
        // Pinned SDK CQE entry_idx is the low 16 bits of the next BB. Reclaim
        // through that CQE only; a later unsignaled RO suffix remains in use.
        if(s.cqe_cnt && s.cqe_cnt!=s.cq_tail) {
            uint32_t next=(s.sq_tail & ~65535U) | (s.completed_head & 65535U);
            if(next<s.sq_tail)next+=65536U;
            s.sq_tail=next;
            s.cq_tail=s.cqe_cnt;
        }
    }
    // PRODUCTION_METHODS
    constexpr bool submit(unsigned p,unsigned n) {
        Admit(p);
        auto& s=states[peers_[p].state];
        if(!aclshmemi_udma_sq_has_credit(&s,n+1)) return false;
        last_=p;
        const auto cqes_before=s.cqe_cnt;
        SubmitBatch(n,n,true);
        if(s.cqe_cnt!=cqes_before || copied_bytes)return false;
        return valid;
    }
    constexpr bool intact() const {
        for(auto word:scratch) if(word!=0x11223344) return false;
        for(const auto& row:remote) for(unsigned i=0;i<32;++i) if(row[i]) return false;
        return valid;
    }
};
template<unsigned C,unsigned Q>
constexpr bool stress(bool shared,uint32_t initial) {
    Model<C,Q> m(shared,initial);
    // Empty sender still publishes every peer/QP marker, with no pressure CQE.
    m.Publish(0,0,2,3);
    m.Quiet();
    if(!m.valid || m.cqes!=C*Q || m.checkpoints || m.final_quiets!=(shared?Q:C*Q))return false;
    for(auto& row:m.remote)for(auto& word:row)word=0;
    // Reuse the same physical SQs across calls; partial, full and short-tail
    // batches interleave peers and QP lanes, exceeding capacity repeatedly.
    for(unsigned round=0;round<3;++round) {
        for(unsigned i=0;i<180;++i) {
            if(!m.submit(i%(C*Q), i%7==0?1:32))return false;
        }
        if(!m.intact())return false; // pressure must not touch pending WQEs/readiness
        m.Publish(0,0,2,3);
        m.Quiet();
        for(unsigned p=0;p<C;++p)for(unsigned q=0;q<Q;++q) {
            if(m.remote[p][3*Q+q]!=1)return false;
            auto& s=m.states[m.peers_[q*C+p].state];
            if(s.cqe_cnt!=s.cq_tail || s.sq_head!=s.sq_tail)return false;
        }
        for(auto& row:m.remote)for(auto& word:row)word=0;
    }
    return m.valid && m.checkpoints>0 && m.cqes==4*C*Q+m.checkpoints && m.doorbells==540 && m.wraps>0;
}
constexpr bool reclaim_prefix(bool small_prefix) {
    Model<2,1> m(true);
    auto& s=m.states[0];
    s.sq_head=1000; s.cqe_cnt=7; s.cq_tail=6;
    s.completed_head=small_prefix?8:300;
    m.Admit(1); // alias peer, same physical state
    if(!m.intact() || m.checkpoints!=(small_prefix?1U:0U))return false;
    return m.credit_quiets==(small_prefix?2U:1U) && aclshmemi_udma_sq_has_credit(&s,33);
}
constexpr bool final_flag_pressure() {
    Model<64,1> m(true);
    m.states[0].sq_head=980;
    // More final markers than the reserved credit. Existing final CQEs must
    // be reclaimed rather than issuing an extra pressure checkpoint.
    m.Publish(0,0,2);
    m.Quiet();
    return m.valid && m.cqes==64 && m.checkpoints==0 && m.credit_quiets>0;
}
constexpr bool no_peers() {
    Model<2,2> m(false); m.count_=0; m.qp_base_=126; // idle AIV63 in a two-QP launch
    m.Publish(0,0,2);m.Quiet();
    return m.valid && !m.cqes && !m.final_quiets;
}
static_assert(stress<2,1>(false,0));
static_assert(stress<4,2>(true,0));
static_assert(stress<2,2>(false,UINT32_MAX-12)); // counter and physical ring wrap
static_assert(reclaim_prefix(false));
static_assert(reclaim_prefix(true));
static_assert(final_flag_pressure());
static_assert(no_peers());
