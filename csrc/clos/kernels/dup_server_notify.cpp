#include <cstdint>

#include "ascend_deepep/kernels/ub_workspace.h"
#include "kernel_operator.h"
#include "shmem.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"

#include "ascend_deepep/kernels/launch.h"
#include "ascend_deepep/kernels/types.h"

using namespace AscendC;

namespace {

constexpr uint32_t kMaxDataCopyBytes = 65504;
constexpr uint32_t kNotifySimtMaxThreads = 1024;
constexpr uint32_t kNotifyUbBytes = ascend_deepep::platform::kKernelUbBytes;
constexpr uint32_t kNotifyResetUbBytes = 32U;
constexpr uint32_t kNotifyUdmaWqeBufferBytes = 128U * 64U;
constexpr uint32_t kNotifySignalUbOffset = kNotifyUdmaWqeBufferBytes;
constexpr uint32_t kNotifySignalStride = 512U;
constexpr uint32_t kNotifyCopyUbOffset = kNotifySignalUbOffset + 32U;
constexpr uint32_t kNotifyCopyTileBytes = kMaxDataCopyBytes;
constexpr uint32_t kNotifyCopyBufferCount = 2U;
constexpr uint32_t kInvalidPowerOfTwoShift = 32U;

// Exchange 开始后 Count/Pack 的 UB 已失效，尾部空间复用为两个本地 GM 搬运槽。
static_assert(
    kNotifyCopyUbOffset + kNotifyCopyBufferCount * kNotifyCopyTileBytes <= ascend_deepep::platform::kKernelUbBytes);

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr aclshmemx_udma_op_config_t kNotifyDataConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kNotifyReadyConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
#endif

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t MaxU32(uint32_t lhs, uint32_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

__aicore__ inline uint32_t NotifySimtThreads(uint32_t work_items)
{
    return MinU32(kNotifySimtMaxThreads, MaxU32(32U, AlignUpU32(work_items, 32U)));
}

template <typename T>
__aicore__ inline void CopyUbToGm(
    AscendC::GlobalTensor<T> dst_gm, AscendC::LocalTensor<T> src_ub, uint32_t element_count)
{
    const uint32_t element_bytes = static_cast<uint32_t>(sizeof(T));
    const uint32_t total_bytes = element_count * element_bytes;
    for (uint32_t byte_offset = 0; byte_offset < total_bytes; byte_offset += kMaxDataCopyBytes) {
        const uint32_t copy_bytes = MinU32(kMaxDataCopyBytes, total_bytes - byte_offset);
        const uint32_t element_offset = byte_offset / element_bytes;
        AscendC::DataCopyExtParams copy_params(1, copy_bytes, 0, 0, 0);
        AscendC::DataCopyPad(dst_gm[element_offset], src_ub[element_offset], copy_params);
    }
}

template <typename T>
__aicore__ inline void CopyGmToUb(
    AscendC::LocalTensor<T> dst_ub, AscendC::GlobalTensor<T> src_gm, uint32_t element_count)
{
    const uint32_t element_bytes = static_cast<uint32_t>(sizeof(T));
    const uint32_t total_bytes = element_count * element_bytes;
    AscendC::DataCopyPadExtParams<T> pad_params{false, 0, 0, 0};
    for (uint32_t byte_offset = 0; byte_offset < total_bytes; byte_offset += kMaxDataCopyBytes) {
        const uint32_t copy_bytes = MinU32(kMaxDataCopyBytes, total_bytes - byte_offset);
        const uint32_t element_offset = byte_offset / element_bytes;
        AscendC::DataCopyExtParams copy_params(1, copy_bytes, 0, 0, 0);
        AscendC::DataCopyPad(dst_ub[element_offset], src_gm[element_offset], copy_params, pad_params);
    }
}

__simt_vf__ __launch_bounds__(kNotifySimtMaxThreads) inline void CountNotifyRoutesSimt(
    __ubuf__ ascend_deepep::topk_idx_t* topk_ub, __gm__ ascend_deepep::topk_idx_t* topk_gm, __ubuf__ int32_t* histogram,
    __ubuf__ int32_t* expert_counts, __ubuf__ int32_t* server_counts, __ubuf__ uint64_t* local_server_mask,
    __gm__ uint64_t* source_server_mask, int32_t token_base, int32_t num_tokens, int32_t num_topk, int32_t num_experts,
    int32_t experts_per_server, int32_t experts_per_server_shift, int32_t expert_stride, int32_t histogram_stride,
    int32_t worker_count, int32_t thread_count, int32_t inputs_in_ub)
{
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const bool use_expert_shift = experts_per_server_shift < static_cast<int32_t>(kInvalidPowerOfTwoShift);
    const int32_t tokens_per_worker = (num_tokens + worker_count - 1) / worker_count;
    const int32_t active_workers = (num_tokens + tokens_per_worker - 1) / tokens_per_worker;

    // Give each worker a private histogram row, avoiding UB atomics while scanning routes.
    if (thread < active_workers) {
        __ubuf__ int32_t* row = histogram + thread * histogram_stride;
        const int32_t token_begin = thread * tokens_per_worker;
        const int32_t token_end =
            token_begin + tokens_per_worker < num_tokens ? token_begin + tokens_per_worker : num_tokens;
        if (inputs_in_ub != 0) {
            for (int32_t local_token = token_begin; local_token < token_end; ++local_token) {
                uint64_t server_mask = 0U;
                const int32_t route_base = local_token * num_topk;
                for (int32_t topk = 0; topk < num_topk; ++topk) {
                    const int64_t expert = static_cast<int64_t>(topk_ub[route_base + topk]);
                    if (expert < 0 || expert >= num_experts) {
                        continue;
                    }
                    const int32_t expert_idx = static_cast<int32_t>(expert);
                    const int32_t dst_server =
                        use_expert_shift ? expert_idx >> experts_per_server_shift : expert_idx / experts_per_server;
                    const uint64_t server_bit = static_cast<uint64_t>(1) << dst_server;
                    row[expert_idx] += 1;
                    if ((server_mask & server_bit) == 0U) {
                        row[expert_stride + dst_server] += 1;
                        server_mask |= server_bit;
                    }
                }
                local_server_mask[local_token] = server_mask;
                source_server_mask[token_base + local_token] = server_mask;
            }
        } else {
            for (int32_t local_token = token_begin; local_token < token_end; ++local_token) {
                uint64_t server_mask = 0U;
                const int32_t global_route_base = (token_base + local_token) * num_topk;
                for (int32_t topk = 0; topk < num_topk; ++topk) {
                    const int64_t expert = static_cast<int64_t>(topk_gm[global_route_base + topk]);
                    if (expert < 0 || expert >= num_experts) {
                        continue;
                    }
                    const int32_t expert_idx = static_cast<int32_t>(expert);
                    const int32_t dst_server =
                        use_expert_shift ? expert_idx >> experts_per_server_shift : expert_idx / experts_per_server;
                    const uint64_t server_bit = static_cast<uint64_t>(1) << dst_server;
                    row[expert_idx] += 1;
                    if ((server_mask & server_bit) == 0U) {
                        row[expert_stride + dst_server] += 1;
                        server_mask |= server_bit;
                    }
                }
                local_server_mask[local_token] = server_mask;
                source_server_mask[token_base + local_token] = server_mask;
            }
        }
    }
    asc_syncthreads();

    // Merge private rows into this AIV's cumulative expert/server counts.
    for (int32_t expert = thread; expert < num_experts; expert += thread_count) {
        int32_t total = 0;
        for (int32_t worker = 0; worker < active_workers; ++worker) {
            total += histogram[worker * histogram_stride + expert];
        }
        expert_counts[expert] += total;
    }
    const int32_t num_servers = num_experts / experts_per_server;
    for (int32_t server = thread; server < num_servers; server += thread_count) {
        int32_t total = 0;
        for (int32_t worker = 0; worker < active_workers; ++worker) {
            total += histogram[worker * histogram_stride + expert_stride + server];
        }
        server_counts[server] += total;
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kNotifySimtMaxThreads) inline void BuildNotifyPrefixesSimt(
    __gm__ int32_t* scratch, __gm__ int32_t* send_token_count, __gm__ int32_t* send_route_count,
    __gm__ int32_t* send_expert_prefix, __ubuf__ int32_t* expert_totals, int32_t core_idx, int32_t core_num,
    int32_t num_servers, int32_t experts_per_server, int32_t expert_stride, int32_t scratch_stride,
    int32_t thread_count)
{
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    for (int32_t server = core_idx; server < num_servers; server += core_num) {
        const int32_t expert_begin = server * experts_per_server;
        // First collect the contribution of every source AIV for this server's experts.
        for (int32_t local_expert = thread; local_expert < experts_per_server; local_expert += thread_count) {
            const int32_t expert = expert_begin + local_expert;
            int32_t total = 0;
            for (int32_t source_core = 0; source_core < core_num; ++source_core) {
                total += scratch[static_cast<int64_t>(source_core) * scratch_stride + expert];
            }
            expert_totals[local_expert] = total;
        }

        // Replace per-AIV token counts with each AIV's deterministic token write offset.
        if (thread == 0) {
            int32_t token_prefix = 0;
            const int32_t server_column = expert_stride + server;
            for (int32_t source_core = 0; source_core < core_num; ++source_core) {
                const int64_t offset = static_cast<int64_t>(source_core) * scratch_stride + server_column;
                const int32_t count = scratch[offset];
                scratch[offset] = token_prefix;
                token_prefix += count;
            }
            send_token_count[server] = token_prefix;
        }
        asc_syncthreads();

        // Build the expert-major route layout exposed to dispatch.
        if (thread == 0) {
            int32_t route_prefix = 0;
            const int64_t prefix_base = static_cast<int64_t>(server) * (experts_per_server + 1);
            for (int32_t local_expert = 0; local_expert < experts_per_server; ++local_expert) {
                const int32_t count = expert_totals[local_expert];
                expert_totals[local_expert] = route_prefix;
                send_expert_prefix[prefix_base + local_expert] = route_prefix;
                route_prefix += count;
            }
            send_expert_prefix[prefix_base + experts_per_server] = route_prefix;
            send_route_count[server] = route_prefix;
        }
        asc_syncthreads();

        // Replace per-AIV expert counts with each AIV's deterministic route write offset.
        for (int32_t local_expert = thread; local_expert < experts_per_server; local_expert += thread_count) {
            const int32_t expert = expert_begin + local_expert;
            int32_t route_prefix = expert_totals[local_expert];
            for (int32_t source_core = 0; source_core < core_num; ++source_core) {
                const int64_t offset = static_cast<int64_t>(source_core) * scratch_stride + expert;
                const int32_t count = scratch[offset];
                scratch[offset] = route_prefix;
                route_prefix += count;
            }
        }
        asc_syncthreads();
    }
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(32) inline void BuildServerRouteBaseSimt(
    __ubuf__ int32_t* server_route_base, int32_t num_servers)
{
    if (threadIdx.x == 0) {
        int32_t route_prefix = 0;
        for (int32_t server = 0; server < num_servers; ++server) {
            const int32_t count = server_route_base[server];
            server_route_base[server] = route_prefix;
            route_prefix += count;
        }
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kNotifySimtMaxThreads) inline void PackNotifyRoutesSimt(
    __ubuf__ ascend_deepep::topk_idx_t* topk_ub, __ubuf__ float* weight_ub, __gm__ ascend_deepep::topk_idx_t* topk_gm,
    __gm__ float* weight_gm, __ubuf__ int32_t* histogram, __ubuf__ int32_t* expert_cursor,
    __ubuf__ int32_t* server_cursor, __ubuf__ int32_t* server_route_base, __gm__ int32_t* send_token_ids,
    __gm__ int32_t* packed_route_position, __ubuf__ uint64_t* local_server_mask, __gm__ float* packed_topk_weights,
    int32_t token_base, int32_t num_tokens, int32_t token_capacity, int32_t send_route_capacity, int32_t num_topk,
    int32_t num_experts, int32_t experts_per_server, int32_t experts_per_server_shift, int32_t expert_stride,
    int32_t histogram_stride, int32_t worker_count, int32_t thread_count, int32_t has_topk_weights,
    int32_t inputs_in_ub)
{
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const bool use_expert_shift = experts_per_server_shift < static_cast<int32_t>(kInvalidPowerOfTwoShift);
    const int32_t tokens_per_worker = (num_tokens + worker_count - 1) / worker_count;
    const int32_t active_workers = (num_tokens + tokens_per_worker - 1) / tokens_per_worker;

    // Convert the Phase 1 private counts into collision-free worker cursors.
    for (int32_t expert = thread; expert < num_experts; expert += thread_count) {
        int32_t prefix = expert_cursor[expert];
        for (int32_t worker = 0; worker < active_workers; ++worker) {
            const int32_t offset = worker * histogram_stride + expert;
            const int32_t count = histogram[offset];
            histogram[offset] = prefix;
            prefix += count;
        }
        expert_cursor[expert] = prefix;
    }
    const int32_t num_servers = num_experts / experts_per_server;
    for (int32_t server = thread; server < num_servers; server += thread_count) {
        int32_t prefix = server_cursor[server];
        for (int32_t worker = 0; worker < active_workers; ++worker) {
            const int32_t offset = worker * histogram_stride + expert_stride + server;
            const int32_t count = histogram[offset];
            histogram[offset] = prefix;
            prefix += count;
        }
        server_cursor[server] = prefix;
    }
    asc_syncthreads();

    // Workers scatter independently while preserving token/top-k order within every expert and server.
    if (thread < active_workers) {
        __ubuf__ int32_t* cursor = histogram + thread * histogram_stride;
        const int32_t token_begin = thread * tokens_per_worker;
        const int32_t token_end =
            token_begin + tokens_per_worker < num_tokens ? token_begin + tokens_per_worker : num_tokens;
        if (inputs_in_ub != 0) {
            for (int32_t local_token = token_begin; local_token < token_end; ++local_token) {
                const int32_t local_route_base = local_token * num_topk;
                const int32_t global_token = token_base + local_token;
                const int32_t global_route_base = global_token * num_topk;

                // Count 阶段已经完成 token 级 server 去重，直接消费该 mask，避免再次按 route 判断首命中。
                const uint64_t server_mask = local_server_mask[local_token];
                for (int32_t dst_server = 0; dst_server < num_servers; ++dst_server) {
                    const uint64_t server_bit = static_cast<uint64_t>(1) << dst_server;
                    if ((server_mask & server_bit) == 0U) {
                        continue;
                    }
                    const int32_t token_index = cursor[expert_stride + dst_server]++;
                    send_token_ids[static_cast<int64_t>(dst_server) * token_capacity + token_index] = global_token;
                }

                for (int32_t topk = 0; topk < num_topk; ++topk) {
                    const int64_t expert = static_cast<int64_t>(topk_ub[local_route_base + topk]);
                    if (expert < 0 || expert >= num_experts) {
                        continue;
                    }
                    const int32_t expert_idx = static_cast<int32_t>(expert);
                    const int32_t dst_server =
                        use_expert_shift ? expert_idx >> experts_per_server_shift : expert_idx / experts_per_server;
                    const int32_t packed_index = cursor[expert_idx]++;
                    packed_route_position[server_route_base[dst_server] + packed_index] = global_route_base + topk;
                    if (has_topk_weights != 0) {
                        packed_topk_weights[static_cast<int64_t>(dst_server) * send_route_capacity + packed_index] =
                            weight_ub[local_route_base + topk];
                    }
                }
            }
        } else {
            for (int32_t local_token = token_begin; local_token < token_end; ++local_token) {
                const int32_t global_token = token_base + local_token;
                const int32_t global_route_base = global_token * num_topk;

                const uint64_t server_mask = local_server_mask[local_token];
                for (int32_t dst_server = 0; dst_server < num_servers; ++dst_server) {
                    const uint64_t server_bit = static_cast<uint64_t>(1) << dst_server;
                    if ((server_mask & server_bit) == 0U) {
                        continue;
                    }
                    const int32_t token_index = cursor[expert_stride + dst_server]++;
                    send_token_ids[static_cast<int64_t>(dst_server) * token_capacity + token_index] = global_token;
                }

                for (int32_t topk = 0; topk < num_topk; ++topk) {
                    const int64_t expert = static_cast<int64_t>(topk_gm[global_route_base + topk]);
                    if (expert < 0 || expert >= num_experts) {
                        continue;
                    }
                    const int32_t expert_idx = static_cast<int32_t>(expert);
                    const int32_t dst_server =
                        use_expert_shift ? expert_idx >> experts_per_server_shift : expert_idx / experts_per_server;
                    const int32_t packed_index = cursor[expert_idx]++;
                    packed_route_position[server_route_base[dst_server] + packed_index] = global_route_base + topk;
                    if (has_topk_weights != 0) {
                        packed_topk_weights[static_cast<int64_t>(dst_server) * send_route_capacity + packed_index] =
                            weight_gm[global_route_base + topk];
                    }
                }
            }
        }
    }
    asc_syncthreads();
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(kNotifySimtMaxThreads) inline void BuildGatewayLaneTablesSimt(
    __gm__ int32_t* send_token_count, __gm__ uint8_t* send_gateway_lane, __gm__ int32_t* packed_route_position,
    __gm__ uint8_t* packed_gateway_lane, int32_t num_servers, int32_t token_capacity, int32_t send_route_capacity,
    int32_t source_lane, int32_t core_idx, int32_t core_num, int32_t thread_count)
{
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t global_thread = core_idx * thread_count + thread;
    const int32_t global_threads = core_num * thread_count;

    for (int32_t linear = global_thread; linear < num_servers * token_capacity; linear += global_threads) {
        const int32_t destination_server = linear / token_capacity;
        const int32_t compact_position = linear - destination_server * token_capacity;
        if (compact_position >= send_token_count[destination_server]) {
            continue;
        }
        const int32_t lane = source_lane;
        send_gateway_lane[linear] = static_cast<uint8_t>(lane);
    }

    for (int32_t packed_index = global_thread; packed_index < send_route_capacity; packed_index += global_threads) {
        const int32_t route_position = packed_route_position[packed_index];
        if (route_position < 0) {
            continue;
        }
        const int32_t lane = source_lane;
        packed_gateway_lane[packed_index] = static_cast<uint8_t>(lane);
    }
    asc_threadfence();
}

__aicore__ inline void LoadNotifyInputs(
    AscendC::GlobalTensor<ascend_deepep::topk_idx_t>& topk_idx_gm, AscendC::GlobalTensor<float>& topk_weights_gm,
    AscendC::LocalTensor<ascend_deepep::topk_idx_t> topk_ub, AscendC::LocalTensor<float> weight_ub,
    uint64_t route_begin, uint32_t route_count, uint32_t load_weights)
{
    CopyGmToUb(topk_ub, topk_idx_gm[route_begin], route_count);
    if (load_weights != 0U) {
        CopyGmToUb(weight_ub, topk_weights_gm[route_begin], route_count);
    }
}

struct NotifyUbLayout {
    uint32_t histogram_elems;
    uint32_t source_server_mask_offset;
    uint32_t topk_slot_bytes;
    uint32_t stage_base_bytes;
};

struct NotifyRuntime {
    uint32_t token_begin;
    uint32_t token_end;
    uint32_t simt_threads;
};

__aicore__ inline void LaunchCountNotifyRoutes(
    AscendC::LocalTensor<ascend_deepep::topk_idx_t> topk_ub, AscendC::LocalTensor<int32_t> histogram_ub,
    AscendC::LocalTensor<int32_t> expert_ub, AscendC::LocalTensor<int32_t> server_ub,
    AscendC::LocalTensor<uint64_t> local_server_mask_ub, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling,
    const NotifyRuntime& runtime, GM_ADDR topk_idx, GM_ADDR source_server_mask)
{
    const uint32_t local_tokens = runtime.token_end - runtime.token_begin;
    asc_vf_call<CountNotifyRoutesSimt>(
        dim3(runtime.simt_threads), reinterpret_cast<__ubuf__ ascend_deepep::topk_idx_t*>(topk_ub.GetPhyAddr()),
        reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(topk_idx),
        reinterpret_cast<__ubuf__ int32_t*>(histogram_ub.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(expert_ub.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(server_ub.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint64_t*>(local_server_mask_ub.GetPhyAddr()),
        reinterpret_cast<__gm__ uint64_t*>(source_server_mask), static_cast<int32_t>(runtime.token_begin),
        static_cast<int32_t>(local_tokens), static_cast<int32_t>(tiling.num_topk),
        static_cast<int32_t>(tiling.num_experts), static_cast<int32_t>(tiling.experts_per_server),
        static_cast<int32_t>(tiling.experts_per_server_shift), static_cast<int32_t>(tiling.expert_stride),
        static_cast<int32_t>(tiling.scratch_stride), static_cast<int32_t>(tiling.pack_workers),
        static_cast<int32_t>(runtime.simt_threads), static_cast<int32_t>(tiling.inputs_in_ub));
}

__aicore__ inline void LaunchPackNotifyRoutes(
    AscendC::LocalTensor<ascend_deepep::topk_idx_t> topk_ub, AscendC::LocalTensor<float> weight_ub,
    AscendC::LocalTensor<int32_t> histogram_ub, AscendC::LocalTensor<int32_t> expert_ub,
    AscendC::LocalTensor<int32_t> server_ub, AscendC::LocalTensor<int32_t> server_route_base_ub,
    AscendC::LocalTensor<uint64_t> local_server_mask_ub, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling,
    const NotifyRuntime& runtime, GM_ADDR topk_idx, GM_ADDR topk_weights, GM_ADDR send_token_ids,
    GM_ADDR packed_route_position, GM_ADDR packed_topk_weights)
{
    const uint32_t local_tokens = runtime.token_end - runtime.token_begin;
    asc_vf_call<PackNotifyRoutesSimt>(
        dim3(runtime.simt_threads), reinterpret_cast<__ubuf__ ascend_deepep::topk_idx_t*>(topk_ub.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(weight_ub.GetPhyAddr()),
        reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(topk_idx), reinterpret_cast<__gm__ float*>(topk_weights),
        reinterpret_cast<__ubuf__ int32_t*>(histogram_ub.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(expert_ub.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(server_ub.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(server_route_base_ub.GetPhyAddr()),
        reinterpret_cast<__gm__ int32_t*>(send_token_ids), reinterpret_cast<__gm__ int32_t*>(packed_route_position),
        reinterpret_cast<__ubuf__ uint64_t*>(local_server_mask_ub.GetPhyAddr()),
        reinterpret_cast<__gm__ float*>(packed_topk_weights), static_cast<int32_t>(runtime.token_begin),
        static_cast<int32_t>(local_tokens), static_cast<int32_t>(tiling.token_capacity),
        static_cast<int32_t>(tiling.send_route_capacity), static_cast<int32_t>(tiling.num_topk),
        static_cast<int32_t>(tiling.num_experts), static_cast<int32_t>(tiling.experts_per_server),
        static_cast<int32_t>(tiling.experts_per_server_shift), static_cast<int32_t>(tiling.expert_stride),
        static_cast<int32_t>(tiling.scratch_stride), static_cast<int32_t>(tiling.pack_workers),
        static_cast<int32_t>(runtime.simt_threads), static_cast<int32_t>(tiling.has_topk_weights),
        static_cast<int32_t>(tiling.inputs_in_ub));
}

__aicore__ inline __gm__ uint8_t* NotifyBufferAt(GM_ADDR symmetric_buffer, uint64_t offset)
{
    return reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + offset;
}

// Notify 收件箱分为两跳：source_server 行落到同号 gateway，source_rank 行再落到专家 rank。
__aicore__ inline __gm__ uint8_t* GatewayPrefixRow(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_server)
{
    return NotifyBufferAt(
        symmetric_buffer, tiling.gateway_prefix_offset + source_server * tiling.gateway_prefix_stride);
}

__aicore__ inline __gm__ uint8_t* GatewayRouteRow(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_server)
{
    return NotifyBufferAt(symmetric_buffer, tiling.gateway_route_offset + source_server * tiling.gateway_route_stride);
}

__aicore__ inline __gm__ uint8_t* GatewayRouteLaneRow(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_server)
{
    return NotifyBufferAt(
        symmetric_buffer, tiling.gateway_route_lane_offset + source_server * tiling.gateway_route_lane_stride);
}

__aicore__ inline __gm__ uint8_t* GatewayChunkEndRow(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_server)
{
    return GatewayPrefixRow(symmetric_buffer, tiling, source_server) +
           (tiling.experts_per_server + 1U) * sizeof(int32_t);
}

__aicore__ inline __gm__ uint64_t* GatewayReadySignal(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_server)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        NotifyBufferAt(symmetric_buffer, tiling.gateway_ready_offset + source_server * kNotifySignalStride));
}

__aicore__ inline __gm__ uint8_t* RankPrefixRow(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_rank)
{
    return NotifyBufferAt(symmetric_buffer, tiling.rank_prefix_offset + source_rank * tiling.rank_prefix_stride);
}

__aicore__ inline __gm__ uint8_t* RankRouteRow(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_rank)
{
    return NotifyBufferAt(symmetric_buffer, tiling.rank_route_offset + source_rank * tiling.rank_route_stride);
}

__aicore__ inline __gm__ uint8_t* RankRouteLaneRow(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_rank)
{
    return NotifyBufferAt(
        symmetric_buffer, tiling.rank_route_lane_offset + source_rank * tiling.rank_route_lane_stride);
}

__aicore__ inline __gm__ uint8_t* RankChunkEndRow(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_rank)
{
    return RankPrefixRow(symmetric_buffer, tiling, source_rank) + (tiling.num_local_experts + 1U) * sizeof(int32_t);
}

__aicore__ inline __gm__ uint64_t* RankReadySignal(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_rank)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        NotifyBufferAt(symmetric_buffer, tiling.rank_ready_offset + source_rank * kNotifySignalStride));
}

__aicore__ inline void StoreNotifySignal(
    __gm__ uint64_t* destination, AscendC::LocalTensor<uint64_t> signal_ub, uint64_t value)
{
    signal_ub.SetValue(0, value);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::GlobalTensor<uint64_t> signal_gm;
    signal_gm.SetGlobalBuffer(destination);
    AscendC::DataCopyExtParams copy_params(1, sizeof(uint64_t), 0, 0, 0);
    AscendC::DataCopyPad(signal_gm, signal_ub, copy_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
}

__aicore__ inline uint32_t BeginNotifyLocalCopyPipeline()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    return 0U;
}

__aicore__ inline uint32_t EnqueueNotifyBytesLocal(
    __gm__ uint8_t* destination, __gm__ uint8_t* source, uint32_t num_bytes,
    AscendC::LocalTensor<uint8_t> staging_pingpong_ub, uint32_t staging_buffer_idx)
{
    if (num_bytes == 0U) {
        return staging_buffer_idx;
    }

    // 本卡不能走 URMA。两个 UB 槽跨 metadata segment 轮换，只有槽位再次使用时才等待上一笔 MTE3。
    AscendC::GlobalTensor<uint8_t> source_gm;
    AscendC::GlobalTensor<uint8_t> destination_gm;
    source_gm.SetGlobalBuffer(source);
    destination_gm.SetGlobalBuffer(destination);
    AscendC::DataCopyPadExtParams<uint8_t> pad_params{false, 0, 0, 0};
    for (uint32_t offset = 0; offset < num_bytes; offset += kNotifyCopyTileBytes) {
        const uint32_t copy_bytes = MinU32(kNotifyCopyTileBytes, num_bytes - offset);
        const auto event_id = staging_buffer_idx == 0U ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<uint8_t> staging_ub = staging_pingpong_ub[staging_buffer_idx * kNotifyCopyTileBytes];
        AscendC::DataCopyExtParams copy_params(1, copy_bytes, 0, 0, 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopyPad(staging_ub, source_gm[offset], copy_params, pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::DataCopyPad(destination_gm[offset], staging_ub, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        staging_buffer_idx = (staging_buffer_idx + 1U) & 1U;
    }

    return staging_buffer_idx;
}

__aicore__ inline void EndNotifyLocalCopyPipeline()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void PutNotifyBytes(
    __gm__ uint8_t* destination, __gm__ uint8_t* source, uint32_t num_bytes, int32_t peer, uint32_t qp_slot,
    __ubuf__ uint8_t* udma_wqe_buffer)
{
    // 当前安装的 SHMEM 未暴露聚合 action 类型，immediate NBI 必须为每个 WQE 生成 CQE。
    for (uint32_t offset = 0; offset < num_bytes; offset += kMaxDataCopyBytes) {
        const uint32_t copy_bytes = MinU32(kMaxDataCopyBytes, num_bytes - offset);
        aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kNotifyDataConfig>(
            destination + offset, source + offset, udma_wqe_buffer, copy_bytes, peer, qp_slot, EVENT_ID5);
    }
}

__aicore__ inline void PutNotifyPayload(
    __gm__ uint8_t* destination_prefix, __gm__ uint8_t* source_prefix, uint32_t prefix_bytes,
    __gm__ uint8_t* destination_chunk_end, __gm__ uint8_t* source_chunk_end, uint32_t chunk_end_bytes,
    __gm__ uint8_t* destination_routes, __gm__ uint8_t* source_routes, uint32_t route_bytes,
    __gm__ uint8_t* destination_route_lanes, __gm__ uint8_t* source_route_lanes, uint32_t route_lane_bytes,
    int32_t peer, uint32_t qp_slot, __ubuf__ uint8_t* udma_wqe_buffer)
{
    PutNotifyBytes(destination_prefix, source_prefix, prefix_bytes, peer, qp_slot, udma_wqe_buffer);
    PutNotifyBytes(destination_chunk_end, source_chunk_end, chunk_end_bytes, peer, qp_slot, udma_wqe_buffer);
    PutNotifyBytes(destination_routes, source_routes, route_bytes, peer, qp_slot, udma_wqe_buffer);
    PutNotifyBytes(destination_route_lanes, source_route_lanes, route_lane_bytes, peer, qp_slot, udma_wqe_buffer);
}

__aicore__ inline void PublishNotifyReady(
    __gm__ uint64_t* destination_signal, __gm__ uint64_t* source_signal, int32_t peer, uint32_t qp_slot,
    __ubuf__ uint8_t* udma_wqe_buffer)
{
    aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kNotifyReadyConfig>(
        destination_signal, source_signal, reinterpret_cast<__ubuf__ uint64_t*>(udma_wqe_buffer), 1U, peer, qp_slot,
        EVENT_ID5);
}

__simt_vf__ __launch_bounds__(kNotifySimtMaxThreads) inline void BuildGatewaySlotMaskSimt(
    __gm__ int32_t* route_position, __gm__ int32_t* route_count, __gm__ int32_t* slot_mask, int32_t source_server,
    int32_t count, int32_t token_capacity, int32_t num_topk, int32_t num_topk_shift, int32_t thread_count)
{
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    if (thread == 0) {
        route_count[source_server] = count;
    }
    const bool use_shift = num_topk_shift < static_cast<int32_t>(kInvalidPowerOfTwoShift);
    for (int32_t route = thread; route < count; route += thread_count) {
        const int32_t position = route_position[route];
        const int32_t token_id = use_shift ? position >> num_topk_shift : position / num_topk;
        const int32_t topk_slot = use_shift ? position & (num_topk - 1) : position - token_id * num_topk;
        if (token_id >= 0 && token_id < token_capacity) {
            // 同一 route_position 只出现一次，因此对不同 slot 的原子加等价于按位 OR。
            asc_atomic_add(slot_mask + source_server * token_capacity + token_id, 1 << topk_slot);
        }
    }
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(32) inline void BuildRecvSegmentsSimt(
    __gm__ int32_t* rank_prefix, __gm__ int32_t* recv_segment_offset, __gm__ int32_t* recv_weight_segment_begin,
    __gm__ int32_t* notify_status, __ubuf__ uint64_t* scan_scratch, int32_t world_size, int32_t num_local_experts,
    int32_t rank_prefix_stride, int32_t recv_route_capacity)
{
    constexpr int32_t kScanThreads = 32;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t num_segments = num_local_experts * world_size;
    if (thread == 0) {
        scan_scratch[kScanThreads] = 0U;
    }
    asc_syncthreads();

    // 每轮并行读取 32 个 (local_expert, source_rank) 计数，并在 UB 中完成分块前缀和。
    for (int32_t tile_begin = 0; tile_begin < num_segments; tile_begin += kScanThreads) {
        const int32_t segment = tile_begin + thread;
        uint64_t count = 0U;
        if (segment < num_segments) {
            const int32_t local_expert = segment / world_size;
            const int32_t source_rank = segment - local_expert * world_size;
            __gm__ int32_t* prefix = reinterpret_cast<__gm__ int32_t*>(
                reinterpret_cast<__gm__ uint8_t*>(rank_prefix) + source_rank * rank_prefix_stride);
            const int32_t segment_begin = prefix[local_expert];
            count = static_cast<uint64_t>(prefix[local_expert + 1] - segment_begin);
            recv_weight_segment_begin[segment] = segment_begin;
        }
        scan_scratch[thread] = count;
        asc_syncthreads();

        for (int32_t offset = 1; offset < kScanThreads; offset <<= 1) {
            const uint64_t addend = thread >= offset ? scan_scratch[thread - offset] : 0U;
            asc_syncthreads();
            if (thread >= offset) {
                scan_scratch[thread] += addend;
            }
            asc_syncthreads();
        }

        const uint64_t tile_base = scan_scratch[kScanThreads];
        if (segment < num_segments) {
            const uint64_t segment_begin = tile_base + scan_scratch[thread] - count;
            recv_segment_offset[segment] = static_cast<int32_t>(
                segment_begin < static_cast<uint64_t>(recv_route_capacity) ? segment_begin : recv_route_capacity);
        }
        asc_syncthreads();

        if (thread == 0) {
            const int32_t valid_segments =
                num_segments - tile_begin < kScanThreads ? num_segments - tile_begin : kScanThreads;
            const uint64_t next_total = tile_base + scan_scratch[valid_segments - 1];
            if (next_total > static_cast<uint64_t>(recv_route_capacity)) {
                scan_scratch[kScanThreads] = static_cast<uint64_t>(recv_route_capacity);
                notify_status[0] = 1;
            } else {
                scan_scratch[kScanThreads] = next_total;
            }
        }
        asc_syncthreads();
    }

    if (thread == 0) {
        recv_segment_offset[num_segments] = static_cast<int32_t>(scan_scratch[kScanThreads]);
    }
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(kNotifySimtMaxThreads) inline void BuildSendChunkTokenEndSimt(
    __gm__ int32_t* send_token_count, __gm__ int32_t* send_token_ids, __gm__ int32_t* send_chunk_token_end,
    int32_t num_servers, int32_t num_tokens, int32_t token_capacity, int32_t num_tokens_per_chunk,
    int32_t num_token_chunks, int32_t chunk_capacity, int32_t thread_count)
{
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t total_chunks = num_servers * chunk_capacity;
    for (int32_t linear_chunk = thread; linear_chunk < total_chunks; linear_chunk += thread_count) {
        const int32_t destination_server = linear_chunk / chunk_capacity;
        const int32_t chunk_id = linear_chunk - destination_server * chunk_capacity;
        const int32_t token_count = send_token_count[destination_server];
        const int32_t compact_begin = chunk_id * num_tokens_per_chunk;
        int32_t compact_end = compact_begin + num_tokens_per_chunk;
        compact_end = compact_end < token_count ? compact_end : token_count;

        // 与 Dispatch 的完成信号保持完全一致：中间 chunk 使用最后一个原始 token_id 的开区间上界；
        // 空 chunk、最后一个有效 chunk 和容量尾部统一发布 N，表示该 source 已经没有后续 token。
        int32_t token_end = num_tokens;
        if (chunk_id < num_token_chunks && compact_begin < compact_end && chunk_id + 1 < num_token_chunks) {
            const int32_t token_id = send_token_ids[destination_server * token_capacity + compact_end - 1];
            token_end = token_id + 1;
        }
        send_chunk_token_end[linear_chunk] = token_end;
    }
    asc_threadfence();
}

__aicore__ inline void InitializeNotifySignals(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t core_idx,
    uint32_t core_num, AscendC::LocalTensor<uint64_t> signal_ub, uint32_t enable_timer,
    uint64_t& profile_local_clear_done)
{
    for (uint32_t source_server = core_idx; source_server < tiling.num_servers; source_server += core_num) {
        StoreNotifySignal(GatewayReadySignal(symmetric_buffer, tiling, source_server), signal_ub, 0U);
    }
    for (uint32_t source_rank = core_idx; source_rank < tiling.world_size; source_rank += core_num) {
        StoreNotifySignal(RankReadySignal(symmetric_buffer, tiling, source_rank), signal_ub, 0U);
    }

    // 必须先让所有 rank 完成旧 flag 清零，随后才允许新的远端 ready 写入。
    AscendC::SyncAll<true>();
    if (enable_timer != 0U && core_idx == 0U) {
        profile_local_clear_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }
    aclshmemi_sync_npu_v4<true>(0);
}

__aicore__ inline uint32_t ServerRouteBase(
    AscendC::GlobalTensor<int32_t> send_route_count_gm, uint32_t destination_server)
{
    uint32_t route_base = 0U;
    for (uint32_t server = 0; server < destination_server; ++server) {
        route_base += static_cast<uint32_t>(send_route_count_gm.GetValue(server));
    }
    return route_base;
}

__aicore__ inline void SendNotifyToGateway(
    GM_ADDR send_route_count, GM_ADDR send_expert_prefix, GM_ADDR packed_route_position, GM_ADDR packed_gateway_lane,
    __gm__ int32_t* send_chunk_token_end, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t destination_server,
    __gm__ uint64_t* source_signal, __ubuf__ uint8_t* udma_wqe_buffer, AscendC::LocalTensor<uint8_t> staging_ub,
    AscendC::LocalTensor<uint64_t> signal_ub)
{
    const uint32_t source_server = tiling.rank / tiling.ranks_per_server;
    const uint32_t gateway_lane = tiling.rank % tiling.ranks_per_server;
    const uint32_t destination_rank = destination_server * tiling.ranks_per_server + gateway_lane;
    const uint32_t prefix_elements = tiling.experts_per_server + 1U;
    const uint32_t prefix_bytes = prefix_elements * sizeof(int32_t);
    const uint32_t chunk_end_bytes = tiling.num_token_chunks * sizeof(int32_t);
    const uint32_t chunk_end_stride = tiling.chunk_capacity * sizeof(int32_t);

    AscendC::GlobalTensor<int32_t> route_count_gm;
    route_count_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(send_route_count));
    const uint32_t route_count = static_cast<uint32_t>(route_count_gm.GetValue(destination_server));
    const uint32_t route_base = ServerRouteBase(route_count_gm, destination_server);
    __gm__ uint8_t* source_prefix = reinterpret_cast<__gm__ uint8_t*>(send_expert_prefix) +
                                    static_cast<uint64_t>(destination_server) * prefix_elements * sizeof(int32_t);
    __gm__ uint8_t* source_routes =
        reinterpret_cast<__gm__ uint8_t*>(packed_route_position) + static_cast<uint64_t>(route_base) * sizeof(int32_t);
    __gm__ uint8_t* source_route_lanes = reinterpret_cast<__gm__ uint8_t*>(packed_gateway_lane) + route_base;
    __gm__ uint8_t* source_chunk_end = reinterpret_cast<__gm__ uint8_t*>(send_chunk_token_end) +
                                       static_cast<uint64_t>(destination_server) * chunk_end_stride;
    __gm__ uint8_t* destination_prefix = GatewayPrefixRow(symmetric_buffer, tiling, source_server);
    __gm__ uint8_t* destination_routes = GatewayRouteRow(symmetric_buffer, tiling, source_server);
    __gm__ uint8_t* destination_route_lanes = GatewayRouteLaneRow(symmetric_buffer, tiling, source_server);
    __gm__ uint8_t* destination_chunk_end = GatewayChunkEndRow(symmetric_buffer, tiling, source_server);
    __gm__ uint64_t* destination_ready = GatewayReadySignal(symmetric_buffer, tiling, source_server);

    // 同一 source rank 在每个目标 server 只写同号 gateway 的 source_server 行。
    if (destination_rank == tiling.rank) {
        uint32_t staging_buffer_idx = BeginNotifyLocalCopyPipeline();
        staging_buffer_idx =
            EnqueueNotifyBytesLocal(destination_prefix, source_prefix, prefix_bytes, staging_ub, staging_buffer_idx);
        staging_buffer_idx = EnqueueNotifyBytesLocal(
            destination_chunk_end, source_chunk_end, chunk_end_bytes, staging_ub, staging_buffer_idx);
        staging_buffer_idx = EnqueueNotifyBytesLocal(
            destination_routes, source_routes, route_count * sizeof(int32_t), staging_ub, staging_buffer_idx);
        staging_buffer_idx = EnqueueNotifyBytesLocal(
            destination_route_lanes, source_route_lanes, route_count, staging_ub, staging_buffer_idx);
        EndNotifyLocalCopyPipeline();
        StoreNotifySignal(destination_ready, signal_ub, 1U);
        return;
    }

    // 第一跳只发送 prefix、chunk 上界和 route_position；topk_weight 留到 Dispatch 再发送。
    PutNotifyPayload(
        destination_prefix, source_prefix, prefix_bytes, destination_chunk_end, source_chunk_end, chunk_end_bytes,
        destination_routes, source_routes, route_count * sizeof(int32_t), destination_route_lanes, source_route_lanes,
        route_count, destination_rank, 0U, udma_wqe_buffer);
    PublishNotifyReady(destination_ready, source_signal, destination_rank, 0U, udma_wqe_buffer);
}

__aicore__ inline void ForwardGatewayNotifyRow(
    GM_ADDR gateway_route_count, GM_ADDR gateway_slot_mask, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t source_server, uint32_t destination_lane,
    uint32_t qp_slot, bool build_gateway_outputs, __gm__ uint64_t* source_signal, __ubuf__ uint8_t* udma_wqe_buffer,
    AscendC::LocalTensor<uint8_t> staging_ub, AscendC::LocalTensor<uint64_t> signal_ub)
{
    // ready 可见即表示第一跳 prefix、chunk 上界和 route 数据可见。
    aclshmem_uint64_wait_until(GatewayReadySignal(symmetric_buffer, tiling, source_server), ACLSHMEM_CMP_NE, 0U);

    __gm__ uint8_t* gateway_prefix = GatewayPrefixRow(symmetric_buffer, tiling, source_server);
    __gm__ uint8_t* gateway_routes = GatewayRouteRow(symmetric_buffer, tiling, source_server);
    __gm__ uint8_t* gateway_route_lanes = GatewayRouteLaneRow(symmetric_buffer, tiling, source_server);
    AscendC::GlobalTensor<int32_t> prefix_gm;
    prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gateway_prefix));
    const uint32_t route_count = static_cast<uint32_t>(prefix_gm.GetValue(tiling.experts_per_server));
    if (build_gateway_outputs) {
        const uint32_t mask_threads = NotifySimtThreads(MaxU32(1U, route_count));
        asc_vf_call<BuildGatewaySlotMaskSimt>(
            dim3(mask_threads), reinterpret_cast<__gm__ int32_t*>(gateway_routes),
            reinterpret_cast<__gm__ int32_t*>(gateway_route_count),
            reinterpret_cast<__gm__ int32_t*>(gateway_slot_mask), static_cast<int32_t>(source_server),
            static_cast<int32_t>(route_count), static_cast<int32_t>(tiling.token_capacity),
            static_cast<int32_t>(tiling.num_topk), static_cast<int32_t>(tiling.num_topk_shift),
            static_cast<int32_t>(mask_threads));
    }

    const uint32_t gateway_lane = tiling.rank % tiling.ranks_per_server;
    const uint32_t destination_server = tiling.rank / tiling.ranks_per_server;
    const uint32_t source_rank = source_server * tiling.ranks_per_server + gateway_lane;
    const uint32_t rank_prefix_bytes = (tiling.num_local_experts + 1U) * sizeof(int32_t);
    const uint32_t chunk_end_bytes = tiling.num_token_chunks * sizeof(int32_t);
    const uint32_t expert_begin = destination_lane * tiling.num_local_experts;
    const uint32_t route_begin = static_cast<uint32_t>(prefix_gm.GetValue(expert_begin));
    const uint32_t route_end = static_cast<uint32_t>(prefix_gm.GetValue(expert_begin + tiling.num_local_experts));
    const uint32_t destination_rank = destination_server * tiling.ranks_per_server + destination_lane;
    __gm__ uint8_t* destination_prefix = RankPrefixRow(symmetric_buffer, tiling, source_rank);
    __gm__ uint8_t* destination_routes = RankRouteRow(symmetric_buffer, tiling, source_rank);
    __gm__ uint8_t* destination_route_lanes = RankRouteLaneRow(symmetric_buffer, tiling, source_rank);
    __gm__ uint8_t* destination_chunk_end = RankChunkEndRow(symmetric_buffer, tiling, source_rank);
    __gm__ uint64_t* destination_ready = RankReadySignal(symmetric_buffer, tiling, source_rank);
    __gm__ uint8_t* source_prefix = gateway_prefix + expert_begin * sizeof(int32_t);
    __gm__ uint8_t* source_routes = gateway_routes + route_begin * sizeof(int32_t);
    __gm__ uint8_t* source_route_lanes = gateway_route_lanes + route_begin;
    __gm__ uint8_t* source_chunk_end = GatewayChunkEndRow(symmetric_buffer, tiling, source_server);
    const uint32_t route_bytes = (route_end - route_begin) * sizeof(int32_t);

    if (destination_rank == tiling.rank) {
        uint32_t staging_buffer_idx = BeginNotifyLocalCopyPipeline();
        staging_buffer_idx = EnqueueNotifyBytesLocal(
            destination_prefix, source_prefix, rank_prefix_bytes, staging_ub, staging_buffer_idx);
        staging_buffer_idx = EnqueueNotifyBytesLocal(
            destination_chunk_end, source_chunk_end, chunk_end_bytes, staging_ub, staging_buffer_idx);
        staging_buffer_idx =
            EnqueueNotifyBytesLocal(destination_routes, source_routes, route_bytes, staging_ub, staging_buffer_idx);
        staging_buffer_idx = EnqueueNotifyBytesLocal(
            destination_route_lanes, source_route_lanes, route_end - route_begin, staging_ub, staging_buffer_idx);
        EndNotifyLocalCopyPipeline();
        StoreNotifySignal(destination_ready, signal_ub, 1U);
    } else {
        PutNotifyPayload(
            destination_prefix, source_prefix, rank_prefix_bytes, destination_chunk_end, source_chunk_end,
            chunk_end_bytes, destination_routes, source_routes, route_bytes, destination_route_lanes,
            source_route_lanes, route_end - route_begin, destination_rank, qp_slot, udma_wqe_buffer);
        PublishNotifyReady(destination_ready, source_signal, destination_rank, qp_slot, udma_wqe_buffer);
    }
}

__aicore__ inline void ExchangeNotifyMetadata(
    GM_ADDR send_route_count, GM_ADDR send_expert_prefix, GM_ADDR packed_route_position, GM_ADDR packed_gateway_lane,
    __gm__ int32_t* send_chunk_token_end, GM_ADDR gateway_route_count, GM_ADDR gateway_slot_mask,
    GM_ADDR symmetric_buffer, __gm__ uint64_t* source_signal,
    const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t core_idx,
    AscendC::LocalTensor<uint8_t> staging_ub, AscendC::LocalTensor<uint64_t> signal_ub,
    __ubuf__ uint8_t* udma_wqe_buffer)
{
    if (core_idx < tiling.notify_producer_workers) {
        for (uint32_t destination_server = core_idx; destination_server < tiling.num_servers;
             destination_server += tiling.notify_producer_workers) {
            SendNotifyToGateway(
                send_route_count, send_expert_prefix, packed_route_position, packed_gateway_lane, send_chunk_token_end,
                symmetric_buffer, tiling, destination_server, source_signal, udma_wqe_buffer, staging_ub, signal_ub);
        }
        const uint32_t gateway_lane = tiling.rank % tiling.ranks_per_server;
        for (uint32_t destination_server = core_idx; destination_server < tiling.num_servers;
             destination_server += tiling.notify_producer_workers) {
            const uint32_t destination_rank = destination_server * tiling.ranks_per_server + gateway_lane;
            if (destination_rank != tiling.rank) {
                aclshmemx_udma_qp_quiet(static_cast<int32_t>(destination_rank), 0U);
            }
        }
    }

    const uint32_t gateway_worker_begin = tiling.notify_producer_workers;
    const uint32_t gateway_worker_end = gateway_worker_begin + tiling.notify_gateway_workers;
    if (core_idx >= gateway_worker_begin && core_idx < gateway_worker_end) {
        const uint32_t gateway_worker = core_idx - gateway_worker_begin;
        const uint32_t destination_lane = gateway_worker % tiling.ranks_per_server;
        const uint32_t lane_worker = gateway_worker / tiling.ranks_per_server;
        const uint32_t workers_for_lane =
            (tiling.notify_gateway_workers - 1U - destination_lane) / tiling.ranks_per_server + 1U;
        const uint32_t gateway_lane = tiling.rank % tiling.ranks_per_server;
        const uint32_t destination_server = tiling.rank / tiling.ranks_per_server;
        const uint32_t destination_rank = destination_server * tiling.ranks_per_server + destination_lane;

        // 每个 worker 固定一个目标 lane 和一个 QP，只处理自己的 source_server 子序列。
        for (uint32_t source_server = lane_worker; source_server < tiling.num_servers;
             source_server += workers_for_lane) {
            ForwardGatewayNotifyRow(
                gateway_route_count, gateway_slot_mask, symmetric_buffer, tiling, source_server, destination_lane,
                lane_worker, destination_lane == gateway_lane, source_signal, udma_wqe_buffer, staging_ub, signal_ub);
        }
        if (destination_rank != tiling.rank) {
            aclshmemx_udma_qp_quiet(static_cast<int32_t>(destination_rank), lane_worker);
        }
        if (destination_lane == gateway_lane) {
            // slot mask 的各行互不重叠，允许多次 SIMT 调用排队，最后统一等待写回完成。
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
    }
}

__aicore__ inline void FinalizeRecvMetadata(
    GM_ADDR recv_route_position, GM_ADDR recv_gateway_lane, GM_ADDR recv_segment_offset,
    GM_ADDR recv_weight_segment_begin, GM_ADDR recv_chunk_token_end, GM_ADDR notify_status, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t core_idx, uint32_t core_num,
    AscendC::LocalTensor<uint8_t> staging_ub, uint32_t enable_timer, uint64_t& profile_segment_done,
    uint64_t& profile_chunk_done)
{
    if (core_idx == 0U) {
        asc_vf_call<BuildRecvSegmentsSimt>(
            dim3(32U), reinterpret_cast<__gm__ int32_t*>(NotifyBufferAt(symmetric_buffer, tiling.rank_prefix_offset)),
            reinterpret_cast<__gm__ int32_t*>(recv_segment_offset),
            reinterpret_cast<__gm__ int32_t*>(recv_weight_segment_begin),
            reinterpret_cast<__gm__ int32_t*>(notify_status),
            reinterpret_cast<__ubuf__ uint64_t*>(staging_ub.GetPhyAddr()), static_cast<int32_t>(tiling.world_size),
            static_cast<int32_t>(tiling.num_local_experts), static_cast<int32_t>(tiling.rank_prefix_stride),
            static_cast<int32_t>(tiling.recv_route_capacity));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        if (enable_timer != 0U) {
            profile_segment_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
        }
    }

    // 每个 source_rank 行只由一个 AIV 搬运。Gateway 卡本身也执行该步骤，因此后续 server Combine
    // 可以直接复用 recv_chunk_token_end，不需要再保留一份 gateway_chunk_token_end。
    const uint32_t chunk_end_bytes = tiling.num_token_chunks * sizeof(int32_t);
    const uint32_t chunk_end_stride = tiling.chunk_capacity * sizeof(int32_t);
    uint32_t staging_buffer_idx = BeginNotifyLocalCopyPipeline();
    for (uint32_t source_rank = core_idx; source_rank < tiling.world_size; source_rank += core_num) {
        __gm__ uint8_t* source = RankChunkEndRow(symmetric_buffer, tiling, source_rank);
        __gm__ uint8_t* destination = reinterpret_cast<__gm__ uint8_t*>(recv_chunk_token_end) +
                                      static_cast<uint64_t>(source_rank) * chunk_end_stride;
        staging_buffer_idx =
            EnqueueNotifyBytesLocal(destination, source, chunk_end_bytes, staging_ub, staging_buffer_idx);
    }
    EndNotifyLocalCopyPipeline();

    // chunk 表不依赖 segment prefix，其他 AIV 可在 core 0 做 scan 时先行搬运；route 重排前再统一同步。
    AscendC::SyncAll<true>();
    if (enable_timer != 0U && core_idx == 0U) {
        profile_chunk_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    AscendC::GlobalTensor<int32_t> recv_segment_gm;
    AscendC::GlobalTensor<int32_t> recv_weight_segment_gm;
    recv_segment_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recv_segment_offset));
    recv_weight_segment_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recv_weight_segment_begin));
    staging_buffer_idx = BeginNotifyLocalCopyPipeline();
    for (uint32_t segment = core_idx; segment < tiling.num_local_experts * tiling.world_size; segment += core_num) {
        const uint32_t output_begin = static_cast<uint32_t>(recv_segment_gm.GetValue(segment));
        const uint32_t output_end = static_cast<uint32_t>(recv_segment_gm.GetValue(segment + 1U));
        if (output_begin == output_end) {
            continue;
        }

        const uint32_t source_rank = segment % tiling.world_size;
        const uint32_t row_begin = static_cast<uint32_t>(recv_weight_segment_gm.GetValue(source_rank));
        const uint32_t expert_begin = static_cast<uint32_t>(recv_weight_segment_gm.GetValue(segment));
        __gm__ uint8_t* source = RankRouteRow(symmetric_buffer, tiling, source_rank) +
                                 static_cast<uint64_t>(expert_begin - row_begin) * sizeof(int32_t);
        __gm__ uint8_t* destination = reinterpret_cast<__gm__ uint8_t*>(recv_route_position) +
                                      static_cast<uint64_t>(output_begin) * sizeof(int32_t);
        staging_buffer_idx = EnqueueNotifyBytesLocal(
            destination, source, (output_end - output_begin) * sizeof(int32_t), staging_ub, staging_buffer_idx);
        source = RankRouteLaneRow(symmetric_buffer, tiling, source_rank) + expert_begin - row_begin;
        destination = reinterpret_cast<__gm__ uint8_t*>(recv_gateway_lane) + output_begin;
        staging_buffer_idx =
            EnqueueNotifyBytesLocal(destination, source, output_end - output_begin, staging_ub, staging_buffer_idx);
    }
    EndNotifyLocalCopyPipeline();
    AscendC::SyncAll<true>();
}

__aicore__ inline void WaitForRankNotifyRows(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint32_t core_idx,
    uint32_t core_num)
{
    // 第二跳的每个 source_rank 行都由唯一 gateway 写入；AIV 完成自身发送后立即等待分配到的行。
    for (uint32_t source_rank = core_idx; source_rank < tiling.world_size; source_rank += core_num) {
        aclshmem_uint64_wait_until(RankReadySignal(symmetric_buffer, tiling, source_rank), ACLSHMEM_CMP_NE, 0U);
    }
    AscendC::SyncAll<true>();
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dup_server_notify_reset_kernel(
    GM_ADDR symmetric_buffer, ascend_deepep::DupServerNotifyPrepareTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    const uint32_t core_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    const uint32_t core_num = get_block_num() * get_subblockdim();
    ascend_deepep::UbWorkspace resource;
    AscendC::LocalTensor<uint64_t> signal_ub = resource.template GetBufferByByte<uint64_t>(0U);

    uint64_t profile_start = 0U;
    uint64_t profile_local_clear_done = 0U;
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_start = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    InitializeNotifySignals(
        symmetric_buffer, tiling, core_idx, core_num, signal_ub, tiling.enable_timer, profile_local_clear_done);

    if (tiling.enable_timer != 0U && core_idx == 0U && aclshmem_my_pe() == 0) {
        const uint64_t profile_end = static_cast<uint64_t>(AscendC::GetSystemCycle());
        const int32_t local_clear_cycles = static_cast<int32_t>(profile_local_clear_done - profile_start);
        const int32_t rank_sync_cycles = static_cast<int32_t>(profile_end - profile_local_clear_done);
        const int32_t total_cycles = static_cast<int32_t>(profile_end - profile_start);
        AscendC::printf("[ascend_deepep][dup_notify_reset_timer] cycles_per_us=1000\n");
        AscendC::printf(
            "[ascend_deepep][dup_notify_reset_timer] stage=signal_local_clear cycles=%d\n", local_clear_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_reset_timer] stage=signal_rank_sync cycles=%d\n", rank_sync_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_reset_timer] stage=total cycles=%d\n", total_cycles);
    }

    // Timer 模式下 rank 0 的 printf 会造成下一 kernel 的入场偏斜；额外同步仅用于保持打点可比。
    if (tiling.enable_timer != 0U) {
        aclshmemi_sync_npu_v4<true>(0);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dup_server_notify_prepare_kernel(
    GM_ADDR topk_idx, GM_ADDR topk_weights, GM_ADDR send_token_count, GM_ADDR send_token_ids, GM_ADDR send_gateway_lane,
    GM_ADDR send_route_count, GM_ADDR send_expert_prefix, GM_ADDR packed_route_position, GM_ADDR packed_gateway_lane,
    GM_ADDR source_server_mask, GM_ADDR packed_topk_weights, GM_ADDR recv_route_position, GM_ADDR recv_gateway_lane,
    GM_ADDR recv_segment_offset, GM_ADDR recv_weight_segment_begin, GM_ADDR recv_chunk_token_end,
    GM_ADDR gateway_route_count, GM_ADDR gateway_slot_mask, GM_ADDR notify_status, GM_ADDR scratch_buffer,
    GM_ADDR symmetric_buffer, ascend_deepep::DupServerNotifyPrepareTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    const uint32_t core_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    const uint32_t core_num = get_block_num() * get_subblockdim();
    const uint32_t tokens_per_core = tiling.max_tokens_per_core;
    const uint32_t token_begin = MinU32(core_idx * tokens_per_core, tiling.num_tokens);
    const uint32_t token_end = MinU32(token_begin + tokens_per_core, tiling.num_tokens);
    const uint32_t local_tokens = token_end - token_begin;
    const uint32_t state_elems = tiling.expert_stride + 2U * tiling.server_stride;
    const uint32_t topk_slot_bytes =
        tiling.inputs_in_ub != 0U ?
            AlignUpU32(
                tiling.max_tokens_per_core * tiling.num_topk * static_cast<uint32_t>(sizeof(ascend_deepep::topk_idx_t)),
                32U) :
            0U;
    const uint32_t source_server_mask_offset =
        (state_elems + tiling.pack_workers * tiling.scratch_stride) * static_cast<uint32_t>(sizeof(int32_t));
    const uint32_t source_server_mask_bytes =
        AlignUpU32(tiling.max_tokens_per_core * static_cast<uint32_t>(sizeof(uint64_t)), 32U);
    const NotifyUbLayout ub_layout{
        tiling.pack_workers * tiling.scratch_stride, source_server_mask_offset, topk_slot_bytes,
        source_server_mask_offset + source_server_mask_bytes};
    const NotifyRuntime runtime{
        token_begin, token_end, NotifySimtThreads(MaxU32(tiling.num_experts, tiling.pack_workers))};

    ascend_deepep::UbWorkspace resource;
    __ubuf__ uint8_t* udma_wqe_buffer =
        reinterpret_cast<__ubuf__ uint8_t*>(resource.template GetBufferByByte<uint8_t>(0).GetPhyAddr());
    AscendC::LocalTensor<uint64_t> signal_ub = resource.template GetBufferByByte<uint64_t>(kNotifySignalUbOffset);
    AscendC::LocalTensor<uint8_t> communication_staging_ub =
        resource.template GetBufferByByte<uint8_t>(kNotifyCopyUbOffset);
    AscendC::LocalTensor<int32_t> expert_ub = resource.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> server_ub =
        resource.template GetBufferByByte<int32_t>(tiling.expert_stride * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> server_route_base_ub = resource.template GetBufferByByte<int32_t>(
        (tiling.expert_stride + tiling.server_stride) * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> histogram_ub =
        resource.template GetBufferByByte<int32_t>(state_elems * sizeof(int32_t));
    AscendC::LocalTensor<uint64_t> local_server_mask_ub =
        resource.template GetBufferByByte<uint64_t>(ub_layout.source_server_mask_offset);
    const uint32_t topk_ub_offset = tiling.inputs_in_ub != 0U ? ub_layout.stage_base_bytes : 0U;
    const uint32_t weight_ub_offset = tiling.inputs_in_ub != 0U && tiling.has_topk_weights != 0U ?
                                          ub_layout.stage_base_bytes + ub_layout.topk_slot_bytes :
                                          0U;
    AscendC::LocalTensor<ascend_deepep::topk_idx_t> topk_ub =
        resource.template GetBufferByByte<ascend_deepep::topk_idx_t>(topk_ub_offset);
    AscendC::LocalTensor<float> weight_ub = resource.template GetBufferByByte<float>(weight_ub_offset);

    AscendC::GlobalTensor<ascend_deepep::topk_idx_t> topk_idx_gm;
    AscendC::GlobalTensor<float> topk_weights_gm;
    AscendC::GlobalTensor<int32_t> send_route_count_gm;
    AscendC::GlobalTensor<int32_t> scratch_gm;
    topk_idx_gm.SetGlobalBuffer(reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(topk_idx));
    if (tiling.has_topk_weights != 0U) {
        topk_weights_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topk_weights));
    }
    send_route_count_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(send_route_count));
    scratch_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch_buffer));

    uint64_t profile_start = 0U;
    uint64_t profile_input_ready = 0U;
    uint64_t profile_count_done = 0U;
    uint64_t profile_scratch_sync_done = 0U;
    uint64_t profile_prefix_done = 0U;
    uint64_t profile_prefix_sync_done = 0U;
    uint64_t profile_pack_done = 0U;
    uint64_t profile_chunk_done = 0U;
    uint64_t profile_exchange_publish_done = 0U;
    uint64_t profile_exchange_done = 0U;
    uint64_t profile_finalize_segment_done = 0U;
    uint64_t profile_finalize_chunk_done = 0U;
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_start = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    if (tiling.inputs_in_ub != 0U) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        LoadNotifyInputs(
            topk_idx_gm, topk_weights_gm, topk_ub, weight_ub, static_cast<uint64_t>(token_begin) * tiling.num_topk,
            local_tokens * tiling.num_topk, tiling.has_topk_weights);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }

    // 阶段 1：统计本卡 route，并保留每个 SIMT worker 的私有直方图供阶段 3 复用。
    AscendC::Duplicate(expert_ub, static_cast<int32_t>(0), tiling.scratch_stride);
    AscendC::Duplicate(histogram_ub, static_cast<int32_t>(0), ub_layout.histogram_elems);
    AscendC::PipeBarrier<PIPE_V>();
    if (tiling.inputs_in_ub != 0U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }
    if (tiling.enable_timer != 0U) {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
        if (core_idx == 0U) {
            profile_input_ready = static_cast<uint64_t>(AscendC::GetSystemCycle());
        }
    }
    if (local_tokens > 0U) {
        LaunchCountNotifyRoutes(
            topk_ub, histogram_ub, expert_ub, server_ub, local_server_mask_ub, tiling, runtime, topk_idx,
            source_server_mask);
    }

    const uint32_t scratch_row = core_idx * tiling.scratch_stride;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_count_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }
    CopyUbToGm(scratch_gm[scratch_row], expert_ub, tiling.scratch_stride);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::SyncAll<true>();
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_scratch_sync_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    // 阶段 2：每个 AIV 负责一个或多个 server，将计数转换为无写冲突的确定性偏移。
    const uint32_t prefix_threads = NotifySimtThreads(tiling.experts_per_server);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
    asc_vf_call<BuildNotifyPrefixesSimt>(
        dim3(prefix_threads), reinterpret_cast<__gm__ int32_t*>(scratch_buffer),
        reinterpret_cast<__gm__ int32_t*>(send_token_count), reinterpret_cast<__gm__ int32_t*>(send_route_count),
        reinterpret_cast<__gm__ int32_t*>(send_expert_prefix),
        reinterpret_cast<__ubuf__ int32_t*>(expert_ub.GetPhyAddr()), static_cast<int32_t>(core_idx),
        static_cast<int32_t>(core_num), static_cast<int32_t>(tiling.num_servers),
        static_cast<int32_t>(tiling.experts_per_server), static_cast<int32_t>(tiling.expert_stride),
        static_cast<int32_t>(tiling.scratch_stride), static_cast<int32_t>(prefix_threads));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_prefix_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }
    // prefix 由 SIMT 直接写 GM；其他 AIV 从 scratch 回读偏移前需要先完成本卡同步。
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    AscendC::SyncAll<true>();
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_prefix_sync_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    // 阶段 3：回读本核偏移，按 server/expert 稳定打包 route 和去重后的 token_id。
    CopyGmToUb(expert_ub, scratch_gm[scratch_row], tiling.scratch_stride);
    CopyGmToUb(server_route_base_ub, send_route_count_gm, tiling.num_servers);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
    asc_vf_call<BuildServerRouteBaseSimt>(
        dim3(32U), reinterpret_cast<__ubuf__ int32_t*>(server_route_base_ub.GetPhyAddr()),
        static_cast<int32_t>(tiling.num_servers));
    AscendC::PipeBarrier<PIPE_V>();
    if (local_tokens > 0U) {
        LaunchPackNotifyRoutes(
            topk_ub, weight_ub, histogram_ub, expert_ub, server_ub, server_route_base_ub, local_server_mask_ub, tiling,
            runtime, topk_idx, topk_weights, send_token_ids, packed_route_position, packed_topk_weights);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID4);
    // 本地打包由 SIMT 直接写 GM；通信 AIV 使用这些结果前必须完成一次本卡同步。
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    AscendC::SyncAll<true>();
    const uint32_t lane_threads = NotifySimtThreads(tiling.send_route_capacity);
    asc_vf_call<BuildGatewayLaneTablesSimt>(
        dim3(lane_threads), reinterpret_cast<__gm__ int32_t*>(send_token_count),
        reinterpret_cast<__gm__ uint8_t*>(send_gateway_lane), reinterpret_cast<__gm__ int32_t*>(packed_route_position),
        reinterpret_cast<__gm__ uint8_t*>(packed_gateway_lane), static_cast<int32_t>(tiling.num_servers),
        static_cast<int32_t>(tiling.token_capacity), static_cast<int32_t>(tiling.send_route_capacity),
        static_cast<int32_t>(tiling.rank % tiling.ranks_per_server), static_cast<int32_t>(core_idx),
        static_cast<int32_t>(core_num), static_cast<int32_t>(lane_threads));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID4);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    AscendC::SyncAll<true>();
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_pack_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    // send_token_ids 已经按目标 server 紧凑且升序排列。只需读取每个紧凑 chunk 的最后一项，
    // 生成与 Dispatch completion 完全一致的原始 token_id 开区间上界。
    __gm__ int32_t* send_chunk_token_end = reinterpret_cast<__gm__ int32_t*>(scratch_buffer) +
                                           static_cast<uint64_t>(tiling.block_dim) * tiling.scratch_stride;
    if (core_idx == 0U) {
        const uint32_t chunk_threads = NotifySimtThreads(tiling.num_servers * tiling.chunk_capacity);
        asc_vf_call<BuildSendChunkTokenEndSimt>(
            dim3(chunk_threads), reinterpret_cast<__gm__ int32_t*>(send_token_count),
            reinterpret_cast<__gm__ int32_t*>(send_token_ids), send_chunk_token_end,
            static_cast<int32_t>(tiling.num_servers), static_cast<int32_t>(tiling.num_tokens),
            static_cast<int32_t>(tiling.token_capacity), static_cast<int32_t>(tiling.num_tokens_per_chunk),
            static_cast<int32_t>(tiling.num_token_chunks), static_cast<int32_t>(tiling.chunk_capacity),
            static_cast<int32_t>(chunk_threads));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID4);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    AscendC::SyncAll<true>();
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_chunk_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    // 远端 ready 的发送源必须与本卡接收 ready 区分开，否则会提前点亮本卡的接收 flag。
    __gm__ uint64_t* source_signal = reinterpret_cast<__gm__ uint64_t*>(
        reinterpret_cast<__gm__ uint8_t*>(scratch_buffer) +
        static_cast<uint64_t>(core_idx) * tiling.scratch_stride * sizeof(int32_t));
    StoreNotifySignal(source_signal, signal_ub, 1U);

    // Producer 与 gateway worker 使用不相交的 AIV，第一跳和第二跳可以自然形成流水。
    ExchangeNotifyMetadata(
        send_route_count, send_expert_prefix, packed_route_position, packed_gateway_lane, send_chunk_token_end,
        gateway_route_count, gateway_slot_mask, symmetric_buffer, source_signal, tiling, core_idx,
        communication_staging_ub, signal_ub, udma_wqe_buffer);
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_exchange_publish_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }
    WaitForRankNotifyRows(symmetric_buffer, tiling, core_idx, core_num);
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        profile_exchange_done = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    // 目的 rank 将 source-major 临时行压成 Dispatch 使用的 (local_expert, src_rank) 顺序。
    FinalizeRecvMetadata(
        recv_route_position, recv_gateway_lane, recv_segment_offset, recv_weight_segment_begin, recv_chunk_token_end,
        notify_status, symmetric_buffer, tiling, core_idx, core_num, communication_staging_ub, tiling.enable_timer,
        profile_finalize_segment_done, profile_finalize_chunk_done);
    if (tiling.enable_timer != 0U && core_idx == 0U) {
        const uint64_t profile_end = static_cast<uint64_t>(AscendC::GetSystemCycle());
        if (aclshmem_my_pe() != 0) {
            return;
        }
        const int32_t input_init_cycles = static_cast<int32_t>(profile_input_ready - profile_start);
        const int32_t count_cycles = static_cast<int32_t>(profile_count_done - profile_input_ready);
        const int32_t scratch_sync_cycles = static_cast<int32_t>(profile_scratch_sync_done - profile_count_done);
        const int32_t prefix_cycles = static_cast<int32_t>(profile_prefix_done - profile_scratch_sync_done);
        const int32_t prefix_sync_cycles = static_cast<int32_t>(profile_prefix_sync_done - profile_prefix_done);
        const int32_t pack_cycles = static_cast<int32_t>(profile_pack_done - profile_prefix_sync_done);
        const int32_t chunk_cycles = static_cast<int32_t>(profile_chunk_done - profile_pack_done);
        const int32_t exchange_cycles = static_cast<int32_t>(profile_exchange_done - profile_chunk_done);
        const int32_t exchange_publish_cycles =
            static_cast<int32_t>(profile_exchange_publish_done - profile_chunk_done);
        const int32_t exchange_wait_cycles =
            static_cast<int32_t>(profile_exchange_done - profile_exchange_publish_done);
        const int32_t finalize_cycles = static_cast<int32_t>(profile_end - profile_exchange_done);
        const int32_t finalize_segment_cycles =
            static_cast<int32_t>(profile_finalize_segment_done - profile_exchange_done);
        const int32_t finalize_chunk_cycles =
            static_cast<int32_t>(profile_finalize_chunk_done - profile_finalize_segment_done);
        const int32_t finalize_route_cycles = static_cast<int32_t>(profile_end - profile_finalize_chunk_done);
        const int32_t total_cycles = static_cast<int32_t>(profile_end - profile_start);
        AscendC::printf(
            "[ascend_deepep][dup_notify_timer] inputs_in_ub=%d weights=%d cycles_per_us=1000\n",
            static_cast<int32_t>(tiling.inputs_in_ub), static_cast<int32_t>(tiling.has_topk_weights));
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=input_init cycles=%d\n", input_init_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=count cycles=%d\n", count_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=scratch_sync cycles=%d\n", scratch_sync_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=prefix cycles=%d\n", prefix_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=prefix_sync cycles=%d\n", prefix_sync_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=pack cycles=%d\n", pack_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=chunk cycles=%d\n", chunk_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=exchange cycles=%d\n", exchange_cycles);
        AscendC::printf(
            "[ascend_deepep][dup_notify_timer] stage=exchange_core0_work cycles=%d\n", exchange_publish_cycles);
        AscendC::printf(
            "[ascend_deepep][dup_notify_timer] stage=exchange_receive_wait cycles=%d\n", exchange_wait_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=finalize cycles=%d\n", finalize_cycles);
        AscendC::printf(
            "[ascend_deepep][dup_notify_timer] stage=finalize_segment cycles=%d\n", finalize_segment_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=finalize_chunk cycles=%d\n", finalize_chunk_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=finalize_route cycles=%d\n", finalize_route_cycles);
        AscendC::printf("[ascend_deepep][dup_notify_timer] stage=total cycles=%d\n", total_cycles);
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void dup_server_notify_prepare_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* topk_idx, uint8_t* topk_weights, uint8_t* send_token_count,
    uint8_t* send_token_ids, uint8_t* send_gateway_lane, uint8_t* send_route_count, uint8_t* send_expert_prefix,
    uint8_t* packed_route_position, uint8_t* packed_gateway_lane, uint8_t* source_server_mask,
    uint8_t* packed_topk_weights, uint8_t* recv_route_position, uint8_t* recv_gateway_lane,
    uint8_t* recv_segment_offset, uint8_t* recv_weight_segment_begin, uint8_t* recv_chunk_token_end,
    uint8_t* gateway_route_count, uint8_t* gateway_slot_mask, uint8_t* notify_status, uint8_t* scratch_buffer,
    uint8_t* symmetric_buffer, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* tiling = reinterpret_cast<ascend_deepep::DupServerNotifyPrepareTilingData*>(tiling_host);
    // 同一 stream 保证旧 flag 清零并完成全 rank 同步后，再执行 metadata 计算和交换。
    dup_server_notify_reset_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kNotifyResetUbBytes), stream>>>(
        symmetric_buffer, *tiling, ffts_addr);
    dup_server_notify_prepare_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kNotifyUbBytes), stream>>>(
        topk_idx, topk_weights, send_token_count, send_token_ids, send_gateway_lane, send_route_count,
        send_expert_prefix, packed_route_position, packed_gateway_lane, source_server_mask, packed_topk_weights,
        recv_route_position, recv_gateway_lane, recv_segment_offset, recv_weight_segment_begin, recv_chunk_token_end,
        gateway_route_count, gateway_slot_mask, notify_status, scratch_buffer, symmetric_buffer, *tiling, ffts_addr);
}
