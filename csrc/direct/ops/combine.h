#pragma once

#include <cstddef>
#include <functional>
#include <torch/custom_class.h>
#include <tuple>

#include <torch/extension.h>

class ShmemInstance;
namespace ascend_deepep::runtime {
class BufferRuntime;
}

namespace ascend_deepep::ops {

using CombineResult = std::tuple<torch::Tensor, c10::optional<torch::Tensor>>;

// Internal one-shot continuation, consumed before ElasticBuffer.combine returns.
// Retains host launch arguments and the Buffer owner, not a new device buffer.
class CombineEpilogue : public torch::CustomClassHolder {
public:
    explicit CombineEpilogue(std::function<CombineResult()> launch);
    void SetOwner(c10::intrusive_ptr<runtime::BufferRuntime> owner);
    CombineResult Run();

private:
    std::function<CombineResult()> launch_;
    c10::intrusive_ptr<runtime::BufferRuntime> owner_;
    int64_t stream_id_;
    int device_index_;
    bool consumed_{false};
};

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> V1Combine(
    torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> bias_0,
    c10::optional<torch::Tensor> bias_1, torch::Tensor src_idx, torch::Tensor rank_prefix_matrix,
    torch::Tensor channel_prefix_matrix, torch::Tensor send_head, int64_t num_aiv,
    int64_t num_max_nvl_chunked_send_tokens, int64_t num_max_nvl_chunked_recv_tokens, int rank, int world_size,
    std::size_t symmetric_buffer_bytes, std::size_t notify_workspace_bytes, bool use_udma, ShmemInstance& shmem,
    void* symmetric_buffer = nullptr, c10::optional<int64_t> allocation_stream_id = c10::nullopt,
    c10::optional<int64_t> num_max_tokens_per_rank = c10::nullopt, int64_t udma_peers_per_worker = 4,
    int64_t udma_wqes_per_visit = 8, c10::intrusive_ptr<CombineEpilogue>* epilogue = nullptr,
    c10::optional<torch::Tensor> expanded_x = c10::nullopt,
    c10::optional<torch::Tensor> single_source_slots = c10::nullopt, std::function<void()> aggregate = {},
    c10::optional<torch::Tensor> pipeline_mapping = c10::nullopt,
    c10::optional<torch::Tensor> pipeline_values = c10::nullopt, int64_t pipeline_chunk_tokens = 0,
    int64_t reduce_num_aiv = 0);

}  // namespace ascend_deepep::ops
