#include <cstdint>

#include "ascend_deepep/kernels/ub_workspace.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "ascend_deepep/kernels/launch.h"
#include "ascend_deepep/kernels/types.h"

using namespace AscendC;

namespace {

constexpr int32_t kInt32Bytes = 4;
constexpr int32_t kUbAlignBytes = 32;
constexpr int32_t kInt32AlignElems = kUbAlignBytes / kInt32Bytes;
constexpr int32_t kDispatchLayoutUbBytes = static_cast<int32_t>(ascend_deepep::platform::kKernelUbBytes);
constexpr int32_t kReduceBufferCount = 2;
constexpr int32_t kMaxReduceCols = kDispatchLayoutUbBytes / kReduceBufferCount / kInt32Bytes;
static_assert(kMaxReduceCols % kInt32AlignElems == 0);
static_assert(kMaxReduceCols * kReduceBufferCount * kInt32Bytes == kDispatchLayoutUbBytes);

__aicore__ inline int32_t IntMin(int32_t lhs, int32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline int32_t AlignUpBytes(int32_t value, int32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_layout_kernel(
    GM_ADDR topk_idx, GM_ADDR scratch_buffer, GM_ADDR num_tokens_per_rank, GM_ADDR num_tokens_per_expert,
    GM_ADDR is_token_in_rank, ascend_deepep::DispatchLayoutTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    (void)tiling.rank;

    if ASCEND_IS_AIC {
        return;
    }

    const int32_t core_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    const int32_t core_num = get_block_num() * get_subblockdim();

    const int32_t world_size = static_cast<int32_t>(tiling.world_size);
    const int32_t num_tokens = static_cast<int32_t>(tiling.num_tokens);
    const int32_t num_topk = static_cast<int32_t>(tiling.num_topk);
    const int32_t num_experts = static_cast<int32_t>(tiling.num_experts);
    const int32_t rank_stride = static_cast<int32_t>(tiling.rank_stride);
    const int32_t expert_stride = static_cast<int32_t>(tiling.expert_stride);
    const int32_t experts_per_rank = num_experts / world_size;

    const int32_t rank_count_offset_bytes = 0;
    const int32_t expert_count_offset_bytes = AlignUpBytes(rank_stride * kInt32Bytes, kUbAlignBytes);
    const int32_t is_token_in_rank_offset_bytes =
        AlignUpBytes(expert_count_offset_bytes + expert_stride * kInt32Bytes, kUbAlignBytes);
    const int32_t is_token_in_rank_ub_bytes = AlignUpBytes(world_size, kUbAlignBytes);

    ascend_deepep::UbWorkspace resource;
    AscendC::LocalTensor<int32_t> rank_count_ub =
        resource.template GetBufferByByte<int32_t>(static_cast<uint32_t>(rank_count_offset_bytes));
    AscendC::LocalTensor<int32_t> expert_count_ub =
        resource.template GetBufferByByte<int32_t>(static_cast<uint32_t>(expert_count_offset_bytes));
    AscendC::LocalTensor<uint8_t> is_token_in_rank_ub =
        resource.template GetBufferByByte<uint8_t>(static_cast<uint32_t>(is_token_in_rank_offset_bytes));
    AscendC::LocalTensor<uint32_t> is_token_in_rank_zero_ub =
        resource.template GetBufferByByte<uint32_t>(static_cast<uint32_t>(is_token_in_rank_offset_bytes));

    __gm__ int32_t* scratch_gm = reinterpret_cast<__gm__ int32_t*>(scratch_buffer);
    AscendC::GlobalTensor<ascend_deepep::topk_idx_t> topk_gm;
    AscendC::GlobalTensor<uint8_t> is_token_in_rank_gm;
    AscendC::GlobalTensor<int32_t> rank_out_gm;
    AscendC::GlobalTensor<int32_t> expert_out_gm;
    AscendC::GlobalTensor<int32_t> rank_partials_gm;
    AscendC::GlobalTensor<int32_t> expert_partials_gm;
    topk_gm.SetGlobalBuffer(reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(topk_idx));
    is_token_in_rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(is_token_in_rank));
    rank_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(num_tokens_per_rank));
    expert_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(num_tokens_per_expert));
    rank_partials_gm.SetGlobalBuffer(scratch_gm);
    expert_partials_gm.SetGlobalBuffer(scratch_gm + core_num * rank_stride);

    AscendC::Duplicate(rank_count_ub, static_cast<int32_t>(0), rank_stride);
    AscendC::Duplicate(expert_count_ub, static_cast<int32_t>(0), expert_stride);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    for (int32_t token = core_idx; token < num_tokens; token += core_num) {
        AscendC::Duplicate(
            is_token_in_rank_zero_ub, static_cast<uint32_t>(0),
            is_token_in_rank_ub_bytes / static_cast<int32_t>(sizeof(uint32_t)));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        const int32_t route_base = token * num_topk;
        for (int32_t k = 0; k < num_topk; ++k) {
            const int32_t expert = static_cast<int32_t>(topk_gm.GetValue(route_base + k));
            if (expert == -1) {
                continue;
            }

            expert_count_ub.SetValue(expert, expert_count_ub.GetValue(expert) + 1);

            const int32_t dst_rank = expert / experts_per_rank;
            if (is_token_in_rank_ub.GetValue(dst_rank) == 0) {
                rank_count_ub.SetValue(dst_rank, rank_count_ub.GetValue(dst_rank) + 1);
                is_token_in_rank_ub.SetValue(dst_rank, static_cast<uint8_t>(1));
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams is_token_in_rank_store_params(1, static_cast<uint32_t>(world_size), 0, 0, 0);
        AscendC::DataCopyPad(
            is_token_in_rank_gm[token * world_size], is_token_in_rank_ub, is_token_in_rank_store_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    const int32_t rank_row_base = core_idx * rank_stride;
    const int32_t expert_row_base = core_idx * expert_stride;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyExtParams rank_partial_store_params(1, static_cast<uint32_t>(rank_stride * kInt32Bytes), 0, 0, 0);
    AscendC::DataCopyPad(rank_partials_gm[rank_row_base], rank_count_ub, rank_partial_store_params);
    AscendC::DataCopyExtParams expert_partial_store_params(
        1, static_cast<uint32_t>(expert_stride * kInt32Bytes), 0, 0, 0);
    AscendC::DataCopyPad(expert_partials_gm[expert_row_base], expert_count_ub, expert_partial_store_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

    AscendC::SyncAll<true>();

    const int32_t reduce_acc_offset_bytes = 0;
    const int32_t reduce_tmp_offset_bytes = AlignUpBytes(kMaxReduceCols * kInt32Bytes, kUbAlignBytes);
    AscendC::LocalTensor<int32_t> reduce_acc_ub =
        resource.template GetBufferByByte<int32_t>(static_cast<uint32_t>(reduce_acc_offset_bytes));
    AscendC::LocalTensor<int32_t> reduce_tmp_ub =
        resource.template GetBufferByByte<int32_t>(static_cast<uint32_t>(reduce_tmp_offset_bytes));
    AscendC::DataCopyPadExtParams<int32_t> reduce_pad_params{false, 0, 0, 0};

    if ((core_idx & 1) != 0) {
        const int32_t reduce_worker_idx = core_idx / 2;
        const int32_t reduce_worker_num = core_num / 2;
        const int32_t total_blocks = expert_stride / kInt32AlignElems;
        const int32_t blocks_per_worker = total_blocks / reduce_worker_num;
        const int32_t extra_blocks = total_blocks % reduce_worker_num;
        const int32_t worker_blocks = blocks_per_worker + (reduce_worker_idx < extra_blocks ? 1 : 0);
        const int32_t begin_block = reduce_worker_idx * blocks_per_worker + IntMin(reduce_worker_idx, extra_blocks);
        const int32_t max_reduce_blocks = kMaxReduceCols / kInt32AlignElems;
        const int32_t reduce_loop_count = (worker_blocks + max_reduce_blocks - 1) / max_reduce_blocks;
        const int32_t blocks_per_loop = reduce_loop_count == 0 ? 0 : worker_blocks / reduce_loop_count;
        const int32_t extra_loop_blocks = reduce_loop_count == 0 ? 0 : worker_blocks % reduce_loop_count;
        int32_t current_block = begin_block;

        for (int32_t loop = 0; loop < reduce_loop_count; ++loop) {
            const int32_t loop_blocks = blocks_per_loop + (loop < extra_loop_blocks ? 1 : 0);
            const int32_t base = current_block * kInt32AlignElems;
            const int32_t len = loop_blocks * kInt32AlignElems;
            current_block += loop_blocks;
            AscendC::DataCopyExtParams copy_params(1, static_cast<uint32_t>(len * kInt32Bytes), 0, 0, 0);
            AscendC::DataCopyPad(reduce_acc_ub, expert_partials_gm[base], copy_params, reduce_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            for (int32_t source_core = 1; source_core < core_num; ++source_core) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::DataCopyPad(
                    reduce_tmp_ub, expert_partials_gm[source_core * expert_stride + base], copy_params,
                    reduce_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Add(reduce_acc_ub, reduce_acc_ub, reduce_tmp_ub, len);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(expert_out_gm[base], reduce_acc_ub, copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        return;
    }

    if ((core_idx & 1) == 0) {
        const int32_t reduce_worker_idx = core_idx / 2;
        const int32_t reduce_worker_num = (core_num + 1) / 2;
        const int32_t total_blocks = rank_stride / kInt32AlignElems;
        const int32_t blocks_per_worker = total_blocks / reduce_worker_num;
        const int32_t extra_blocks = total_blocks % reduce_worker_num;
        const int32_t worker_blocks = blocks_per_worker + (reduce_worker_idx < extra_blocks ? 1 : 0);
        const int32_t begin_block = reduce_worker_idx * blocks_per_worker + IntMin(reduce_worker_idx, extra_blocks);
        const int32_t max_reduce_blocks = kMaxReduceCols / kInt32AlignElems;
        const int32_t reduce_loop_count = (worker_blocks + max_reduce_blocks - 1) / max_reduce_blocks;
        const int32_t blocks_per_loop = reduce_loop_count == 0 ? 0 : worker_blocks / reduce_loop_count;
        const int32_t extra_loop_blocks = reduce_loop_count == 0 ? 0 : worker_blocks % reduce_loop_count;
        int32_t current_block = begin_block;

        for (int32_t loop = 0; loop < reduce_loop_count; ++loop) {
            const int32_t loop_blocks = blocks_per_loop + (loop < extra_loop_blocks ? 1 : 0);
            const int32_t base = current_block * kInt32AlignElems;
            const int32_t len = loop_blocks * kInt32AlignElems;
            current_block += loop_blocks;
            AscendC::DataCopyExtParams copy_params(1, static_cast<uint32_t>(len * kInt32Bytes), 0, 0, 0);
            AscendC::DataCopyPad(reduce_acc_ub, rank_partials_gm[base], copy_params, reduce_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            for (int32_t source_core = 1; source_core < core_num; ++source_core) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::DataCopyPad(
                    reduce_tmp_ub, rank_partials_gm[source_core * rank_stride + base], copy_params, reduce_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Add(reduce_acc_ub, reduce_acc_ub, reduce_tmp_ub, len);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(rank_out_gm[base], reduce_acc_ub, copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void dispatch_layout_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* topk_idx, uint8_t* scratch_buffer, uint8_t* num_tokens_per_rank,
    uint8_t* num_tokens_per_expert, uint8_t* is_token_in_rank, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::DispatchLayoutTilingData*>(tiling_host);
    dispatch_layout_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kDispatchLayoutUbBytes), stream>>>(
        topk_idx, scratch_buffer, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, *td, ffts_addr);
}
