#include <cstdint>
#include <string>

#include <torch/custom_class.h>
#include <torch/script.h>

#include "runtime/buffer_runtime.h"
#include "runtime/shmem_instance.h"

using ascend_deepep::runtime::BufferRuntime;

static auto shmem_instance_registry =
    torch::jit::class_<ShmemInstance>("AscendDeepEP", "ShmemInstance")
        .def(torch::jit::init<int64_t, int64_t, std::string>())
        .def("init", static_cast<void (ShmemInstance::*)(int64_t)>(&ShmemInstance::Init))
        .def("barrier", &ShmemInstance::Barrier)
        .def("destroy", &ShmemInstance::Destroy)
        .def("is_available", &ShmemInstance::IsAvailable)
        .def("rank", &ShmemInstance::Rank)
        .def("world_size", &ShmemInstance::WorldSize);

static auto combine_epilogue_registry =
    torch::jit::class_<ascend_deepep::ops::CombineEpilogue>("AscendDeepEP", "_CombineEpilogue")
        .def("run", &ascend_deepep::ops::CombineEpilogue::Run);

static auto buffer_runtime_registry =
    torch::jit::class_<BufferRuntime>("AscendDeepEP", "BufferRuntime")
        .def(torch::jit::init<int64_t, int64_t, int64_t, std::string, bool, std::string, bool, std::vector<int64_t>,
                              std::string, bool, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(),
             "",
             {torch::arg("rank"), torch::arg("num_ranks"), torch::arg("num_bytes"), torch::arg("transport"),
              torch::arg("low_latency_mode"), torch::arg("ip_port"), torch::arg("use_1g_huge_pages"),
              torch::arg("unique_id"), torch::arg("bootstrap_key"), torch::arg("v1_shared_jetty"),
              torch::arg("v1_token_schedule") = false, torch::arg("udma_dispatch_peer_order") = 0,
              torch::arg("udma_combine_peer_order") = 0, torch::arg("udma_ranks_per_group") = 0,
              torch::arg("udma_peers_per_group_per_worker") = 0, torch::arg("udma_workers_per_peer") = 1,
              torch::arg("udma_qps_per_worker") = 1})
        .def("is_initialized", &BufferRuntime::IsInitialized)
        .def("_configure_tuning", &BufferRuntime::ConfigureTuning)
        .def("barrier", &BufferRuntime::Barrier)
        .def("destroy", &BufferRuntime::Destroy)
        .def("abort_initialization", &BufferRuntime::AbortInitialization)
        .def("mte_mapping_errors", &BufferRuntime::MteMappingErrors)
        .def("get_comm_meta_ptr", &BufferRuntime::CommMetaPtr)
        .def("get_workspace_ptr", &BufferRuntime::WorkspacePtr)
        .def("rank", &BufferRuntime::Rank)
        .def("num_ranks", &BufferRuntime::NumRanks)
        .def("num_bytes", &BufferRuntime::NumBytes)
        .def("local_mem_size", &BufferRuntime::LocalMemSize)
        .def("num_device_aiv", &BufferRuntime::NumDeviceAiv)
        .def("transport", &BufferRuntime::Transport)
        .def("data_op_engine_type", &BufferRuntime::DataOpEngineType)
        .def("low_latency_mode", &BufferRuntime::LowLatencyMode)
        .def("use_1g_huge_pages", &BufferRuntime::Use1gHugePages)
        .def("get_dispatch_layout", &BufferRuntime::GetDispatchLayout, "",
             {torch::arg("topk_idx"), torch::arg("num_experts"), torch::arg("allocation_stream_id") = c10::IValue()})
        .def("intranode_dispatch", &BufferRuntime::IntranodeDispatch, "",
             {torch::arg("x"),
              torch::arg("x_scales"),
              torch::arg("topk_idx"),
              torch::arg("topk_weights"),
              torch::arg("num_tokens_per_rank"),
              torch::arg("is_token_in_rank"),
              torch::arg("num_tokens_per_expert"),
              torch::arg("cached_num_recv_tokens"),
              torch::arg("cached_rank_prefix_matrix"),
              torch::arg("cached_channel_prefix_matrix"),
              torch::arg("expert_alignment"),
              torch::arg("num_worst_tokens"),
              torch::arg("num_aiv"),
              torch::arg("num_max_nvl_chunked_send_tokens"),
              torch::arg("num_max_nvl_chunked_recv_tokens"),
              torch::arg("zero_copy"),
              torch::arg("allocation_stream_id") = c10::IValue(),
              torch::arg("num_max_tokens_per_rank") = c10::IValue(),
              torch::arg("udma_peers_per_worker") = 4,
              torch::arg("udma_wqes_per_visit") = 8})
        .def("intranode_dispatch_with_plan", &BufferRuntime::IntranodeDispatchWithPlan, "",
             {torch::arg("x"),
              torch::arg("x_scales"),
              torch::arg("topk_idx"),
              torch::arg("topk_weights"),
              torch::arg("num_tokens_per_rank"),
              torch::arg("is_token_in_rank"),
              torch::arg("num_tokens_per_expert"),
              torch::arg("cached_num_recv_tokens"),
              torch::arg("cached_rank_prefix_matrix"),
              torch::arg("cached_channel_prefix_matrix"),
              torch::arg("expert_alignment"),
              torch::arg("num_worst_tokens"),
              torch::arg("num_aiv"),
              torch::arg("num_max_nvl_chunked_send_tokens"),
              torch::arg("num_max_nvl_chunked_recv_tokens"),
              torch::arg("zero_copy"),
              torch::arg("allocation_stream_id") = c10::IValue(),
              torch::arg("num_max_tokens_per_rank") = c10::IValue(),
              torch::arg("udma_peers_per_worker") = 4,
              torch::arg("udma_wqes_per_visit") = 8,
              torch::arg("cached_dispatch_plan") = c10::IValue()})
        .def("intranode_dispatch_with_handle", &BufferRuntime::IntranodeDispatchWithHandle, "",
             {torch::arg("x"),
              torch::arg("x_scales"),
              torch::arg("topk_idx"),
              torch::arg("topk_weights"),
              torch::arg("num_tokens_per_rank"),
              torch::arg("is_token_in_rank"),
              torch::arg("num_tokens_per_expert"),
              torch::arg("cached_num_recv_tokens"),
              torch::arg("cached_rank_prefix_matrix"),
              torch::arg("cached_channel_prefix_matrix"),
              torch::arg("expert_alignment"),
              torch::arg("num_worst_tokens"),
              torch::arg("num_aiv"),
              torch::arg("num_max_nvl_chunked_send_tokens"),
              torch::arg("num_max_nvl_chunked_recv_tokens"),
              torch::arg("zero_copy"),
              torch::arg("allocation_stream_id") = c10::IValue(),
              torch::arg("num_max_tokens_per_rank") = c10::IValue(),
              torch::arg("udma_peers_per_worker") = 4,
              torch::arg("udma_wqes_per_visit") = 8,
              torch::arg("cached_dispatch_plan") = c10::IValue(),
              torch::arg("defer_public_expert_prefix") = false})
        .def("intranode_combine", &BufferRuntime::IntranodeCombine, "",
             {torch::arg("x"), torch::arg("topk_weights"), torch::arg("bias_0"), torch::arg("bias_1"),
              torch::arg("src_idx"), torch::arg("rank_prefix_matrix"), torch::arg("channel_prefix_matrix"),
              torch::arg("send_head"), torch::arg("num_aiv"), torch::arg("num_max_nvl_chunked_send_tokens"),
              torch::arg("num_max_nvl_chunked_recv_tokens"), torch::arg("allocation_stream_id") = c10::IValue(),
              torch::arg("num_max_tokens_per_rank") = c10::IValue(), torch::arg("udma_peers_per_worker") = 4,
              torch::arg("udma_wqes_per_visit") = 8})
        .def("_intranode_combine_begin", &BufferRuntime::BeginCombine, "",
             {torch::arg("x"),
              torch::arg("topk_weights"),
              torch::arg("bias_0"),
              torch::arg("bias_1"),
              torch::arg("src_idx"),
              torch::arg("rank_prefix_matrix"),
              torch::arg("channel_prefix_matrix"),
              torch::arg("send_head"),
              torch::arg("num_aiv"),
              torch::arg("num_max_nvl_chunked_send_tokens"),
              torch::arg("num_max_nvl_chunked_recv_tokens"),
              torch::arg("allocation_stream_id") = c10::IValue(),
              torch::arg("num_max_tokens_per_rank") = c10::IValue(),
              torch::arg("udma_peers_per_worker") = 4,
              torch::arg("udma_wqes_per_visit") = 8,
              torch::arg("expanded_x") = c10::IValue(),
              torch::arg("single_source_slots") = c10::IValue(),
              torch::arg("early_aggregate_map") = c10::IValue(),
              torch::arg("pipeline_chunk_tokens") = int64_t(0),
              torch::arg("reduce_num_aiv") = int64_t(0)})
        .def("expand", &BufferRuntime::Expand, "",
             {torch::arg("x"), torch::arg("x_scales"), torch::arg("topk_idx"), torch::arg("topk_weights"),
              torch::arg("num_tokens_per_expert"), torch::arg("expert_alignment"),
              torch::arg("cached_num_expanded_tokens"), torch::arg("cached_expert_offsets"),
              torch::arg("cached_actual_expert_counts"), torch::arg("cached_recv_to_expanded_idx"),
              torch::arg("num_aiv"), torch::arg("allocation_stream_id") = c10::IValue(),
              torch::arg("public_expert_prefix") = c10::IValue(), torch::arg("public_src_metadata") = c10::IValue(),
              torch::arg("single_source_slots") = c10::IValue()})
        .def("aggregate", &BufferRuntime::Aggregate, "",
             {torch::arg("x"), torch::arg("topk_weights"), torch::arg("expanded_values"),
              torch::arg("recv_to_expanded_idx"), torch::arg("num_aiv"),
              torch::arg("allocation_stream_id") = c10::IValue(), torch::arg("single_source_slots") = c10::IValue(),
              torch::arg("rank_prefix_matrix") = c10::IValue()})
        .def("elastic_barrier", &BufferRuntime::ElasticBarrier);
