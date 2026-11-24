// Copyright (c) 2026, Lu Lu

#include "ops/low_latency.hpp"
#include <pybind11/stl.h>
#include <climits>
#include <stdexcept>
#include <string>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/layout/ep/diagnostics.hpp>

#include <vector>
#include "runtime/session.hpp"
#include "runtime/ll_core_config.hpp"
#include "ubx/v2.hpp"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "../kernels/combine/low_latency/combine_layout.h"
#include "../kernels/combine/low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "kernels/ep/dispatch.hpp"
#include "deep_ep/comm/comm_args.hpp"
#include "../kernels/combine/low_latency/launch.hpp"
#include "../kernels/runtime/clock_alignment.h"
#include "../kernels/runtime/clock_wait.hpp"

namespace py = pybind11;
namespace deepep {
// Declared below; the reference-profile helper runs before it.
extern "C" uint32_t deepep_ll_reference_profile_enabled();
namespace {
using Shape = std::vector<int64_t>;
constexpr uint64_t kRankBaseBiasBytes = 1024U;
uint32_t pin_session_cores(Session& session)
{
    session.workspace();
    const auto config = query_ll_core_config();
    if (config.device < 0) throw std::runtime_error("LL launch requires a current device for core verification");
    return session.pin_low_latency_cores(config.active);
}

uint8_t* peer_buffer_base(Session& session, int peer)
{
    if (peer < 0 || peer >= session.world()) throw std::invalid_argument("Invalid LL peer");
    if (session.ll_rdma() && !deep_ep::LlSameHost(session.rank(), peer)) return nullptr;
    auto* original = static_cast<uint8_t*>(peer == session.rank() ? session.workspace() : aclshmem_ptr(session.workspace(), peer));
    if (original == nullptr) throw std::invalid_argument("Invalid LL peer mapping");
    if (session.capacity() != (8ULL << 30)) return original;
    // Translate the original symmetric pointer before applying the destination-rank
    // bias. The shared peer table keeps local receive and remote send views equal.
    return original + uint64_t(peer) * kRankBaseBiasBytes;
}
uint8_t* tensor(py::handle value, const Shape& shape, const char* dtype)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(value).cast<bool>()) {
        throw std::invalid_argument("LL requires tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(value);
    if (!obj.attr("is_contiguous")().cast<bool>() || obj.attr("shape").cast<Shape>() != shape ||
        !obj.attr("dtype").is(torch.attr(dtype))) {
        throw std::invalid_argument("LL tensor shape, dtype or contiguity mismatch");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}
void require_npu(py::handle value, const char* what)
{
    auto obj = py::reinterpret_borrow<py::object>(value);
    if (obj.attr("device").attr("type").cast<std::string>() != "npu")
        throw std::invalid_argument(std::string("LL clock wait requires ") + what + " on the NPU");
}
uint8_t* reference_profile(py::dict tensors, uint32_t cores)
{
    if (!tensors.contains("ll_reference_profile") || tensors["ll_reference_profile"].is_none()) return nullptr;
    if (!deepep_ll_reference_profile_enabled())
        throw std::invalid_argument("Rebuild with DEEPEP_LL_REFERENCE_PROFILE=ON");
    auto profile = py::reinterpret_borrow<py::object>(tensors["ll_reference_profile"]);
    auto device = profile.attr("device");
    if (device.attr("type").cast<std::string>() != "npu" ||
        !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
        throw std::invalid_argument("LL reference profile must be on the input NPU");
    auto* pointer = tensor(profile, {cores, profile::kSlots, profile::kFields}, "int64");
    if (reinterpret_cast<uintptr_t>(pointer) % 512U != 0U)
        throw std::invalid_argument("LL reference profile requires 512-byte alignment");
    return pointer;
}
deep_ep::EpMemoryServerDedupLayout layout(int p, int b, int h, int k, int e, bool fp8)
{
    const uint32_t coreNum = query_ll_core_config().active;
    const auto mode = fp8 ? deep_ep::EpServerDedupInputMode::PrequantizedFp8Packs
                         : deep_ep::EpServerDedupInputMode::Plain16;
    if (b <= 0 || b > 65536) {
        throw std::invalid_argument("LL dispatch exceeds topology, shape, UB or workspace limits");
    }
    auto result = deep_ep::get_dispatch_buffer_size(uint32_t(p), uint32_t(b), uint32_t(h), uint32_t(k),
                                                    uint32_t(e), mode, coreNum);
    if (result.capacityRows > INT32_MAX) {
        throw std::invalid_argument("LL dispatch exceeds topology, shape, UB or workspace limits");
    }
    return result;
}
template <class T>
py::bytes bytes(const T& data)
{
    return py::bytes(reinterpret_cast<const char*>(&data), sizeof(data));
}
void check_dispatch_dtype(const std::string& dtype)
{
    if (dtype != "bfloat16" && dtype != "float8_e4m3fn")
        throw std::invalid_argument("LL dispatch supports BF16 or packed FP8 E4M3FN");
}
// Combine tiling dtype codes live with the Combine kernels; Dispatch has no
// dtype enum and only distinguishes "scales present" from plain BF16.
uint8_t combine_dtype_code(const std::string& dtype)
{
    if (dtype == "bfloat16") return CombineDedup::kEpServerDedupDtypeBf16;
    if (dtype == "float16") return CombineDedup::kEpServerDedupDtypeFp16;
    throw std::invalid_argument("LL combine supports BF16 or FP16 expert output");
}
}  // namespace

// LL capability exports now live in the Host module: dispatch is compiled
// through DeepJIT, so no dispatch device translation unit exists any more.
// Combine/clock still load the device library for their own exports.
extern "C" uint32_t deepep_ll_core_protocol_version()
{
    return deep_ep::kLlCoreProtocolVersion;
}
extern "C" uint32_t deepep_ll_rdma_enabled()
{
#ifdef DEEPEP_LL_RDMA
    return 1U;
#else
    return 0U;
#endif
}
extern "C" uint32_t deepep_ll_ranks_per_group(uint32_t world)
{
    return deep_ep::LlGroupForWorld(world);
}
extern "C" uint32_t deepep_ll_reference_profile_enabled()
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    return 1U;
#else
    return 0U;
#endif
}
extern "C" uint32_t deepep_ll_reference_profile_schema()
{
    return profile::kSchema;
}

void bind_low_latency(py::module_& module)
{
    if (deepep_ll_core_protocol_version() != deep_ep::kLlCoreProtocolVersion)
        throw std::runtime_error("LL Host/device core protocol mismatch; rebuild and reinstall the complete wheel");
    module.attr("low_latency_core_protocol") = deep_ep::kLlCoreProtocolVersion;
    module.attr("low_latency_roce_qps_per_peer") = 0; // actual active physical cores
    module.attr("low_latency_implementation") = "ndr-hns1825";
#ifdef DEEPEP_LL_RDMA
    if (!deepep_ll_rdma_enabled()) throw std::runtime_error("LL Host/device RDMA build mismatch; rebuild complete wheel");
    module.attr("low_latency_rdma_enabled") = true;
#else
    if (deepep_ll_rdma_enabled()) throw std::runtime_error("LL Host/device RDMA build mismatch; rebuild complete wheel");
    module.attr("low_latency_rdma_enabled") = false;
#endif
    const bool reference_enabled = bool(deepep_ll_reference_profile_enabled());
    if (deepep_ll_reference_profile_schema() != profile::kSchema)
        throw std::runtime_error("LL Host/device reference profile mismatch; rebuild and reinstall the complete wheel");
    module.attr("low_latency_reference_profile_enabled") = reference_enabled;
    module.def("low_latency_reference_profile_layout", []() {
        py::dict result;
        result["schema"] = profile::kSchema;
        result["slots"] = profile::kSlots;
        result["fields"] = profile::kFields;
        std::vector<std::string> names;
        for (const auto* name : profile::kNames) names.emplace_back(name);
        result["words"] = names;
        return result;
    });
    module.def("low_latency_core_config", []() {
        const auto cfg = query_ll_core_config();
        py::dict result;
        result["platform_aiv"] = cfg.platform; result["available_aiv"] = cfg.available;
        result["active_aiv"] = cfg.active; result["send_cores"] = cfg.active / 2U;
        result["recv_cores"] = cfg.active / 2U; result["device"] = cfg.device;
        result["runtime_device_queried"] = cfg.device >= 0;
        result["control_capacity_aiv"] = cfg.device >= 0 ? deep_ep::jit_num_vec_cores() : 0;
        return result;
    });
    for (uint32_t world : {16U, 32U, 64U, 128U, 256U})
        if (deepep_ll_ranks_per_group(world) != deep_ep::LlGroupForWorld(world))
            throw std::runtime_error("LL Host/device tiling-key policy mismatch; rebuild the complete wheel");
    module.def("low_latency_ranks_per_group", [](uint32_t world) { return deep_ep::LlGroupForWorld(world); });
    module.attr("low_latency_clock_alignment_version") = clock_alignment::kVersion;
    module.attr("low_latency_clock_cycles_per_second") = clock_alignment::kCyclesPerSecond;
    module.attr("low_latency_clock_wait_record_words") = deep_ep::kClockWaitRecordWords;
    module.attr("low_latency_clock_wait_record_fields") = std::vector<std::string>{
        "enter", "release", "batch", "core"};
    module.def("low_latency_clock_sync",
               [](Session& session, py::object comm_args, py::object samples, uint32_t sample_count,
                  uint32_t warmup_sample_count, uint64_t magic, uint64_t timeout_cycles,
                  uintptr_t stream_address) {
                   if (!session.ll_rdma() || !stream_address || !magic || !timeout_cycles ||
                       sample_count == 0U || sample_count > clock_alignment::kMaxSamples ||
                       warmup_sample_count > clock_alignment::kMaxWarmupSamples) {
                       throw std::invalid_argument("Invalid LL clock synchronization configuration");
                   }
                   auto* comm = tensor(
                       comm_args, {sizeof(deep_ep::CommArgs)}, "uint8");
                   auto* output = tensor(
                       samples, {sample_count, clock_alignment::kSampleWords}, "int64");
                   session.workspace();
                   py::gil_scoped_release release;
                   deepep_clock_sync_kernel_do(
                       reinterpret_cast<void*>(stream_address), comm, output, sample_count,
                       warmup_sample_count, magic, timeout_cycles);
               });
    module.def("low_latency_clock_sleep_at",
               [](Session& session, uint64_t local_deadline, py::object release_cycle,
                  uintptr_t stream_address) {
                   if (!session.ll_rdma() || !stream_address) {
                       throw std::invalid_argument("Invalid LL clock sleep-at configuration");
                   }
                   uint8_t* output = nullptr;
                   if (!release_cycle.is_none()) {
                       output = tensor(release_cycle, {1}, "int64");
                   }
                   session.workspace();
                   py::gil_scoped_release release;
                   deepep_clock_sleep_at_kernel_do(
                       reinterpret_cast<void*>(stream_address), local_deadline, output);
               });
    // In-graph wait shared by every captured call: the deadline lives in a GM
    // table and the call index is a runtime argument, so a replay only needs to
    // refresh the table contents. Cores must match the Dispatch launch exactly
    // because the record stride is a function of the participating core count.
    module.def("low_latency_clock_wait_table",
               [](Session& session, py::object deadlines, int64_t num_iters, int64_t index,
                  py::object records, int64_t batch, bool diagnostic, uintptr_t stream_address) {
                   if (!session.ll_rdma() || !stream_address || num_iters <= 0 || num_iters > INT32_MAX ||
                       index < 0 || index >= num_iters) {
                       throw std::invalid_argument("Invalid LL clock wait-table configuration");
                   }
                   const uint32_t cores = pin_session_cores(session);
                   if (cores == 0U) throw std::invalid_argument("LL clock wait-table has no active cores");
                   auto* table = tensor(deadlines, {num_iters}, "int64");
                   require_npu(deadlines, "deadlines");
                   uint8_t* output = nullptr;
                   if (diagnostic) {
                       output = tensor(records, {num_iters, int64_t(cores),
                                                 int64_t(deep_ep::kClockWaitRecordWords)}, "int64");
                       require_npu(records, "records");
                       if (reinterpret_cast<uintptr_t>(output) % 32U != 0U) {
                           throw std::invalid_argument("LL clock wait records require 32-byte alignment");
                       }
                   }
                   py::gil_scoped_release release;
                   deep_ep::launch_clock_wait_table(cores, reinterpret_cast<void*>(stream_address),
                                                    table, index, output, batch, diagnostic);
               },
               "Wait until deadlines[index] (device cycles) on every active AIV core; "
               "records, when requested, are int64[num_iters][cores][4].");
    module.def("low_latency_layout", [](int p, int b, int h, int k, int e, const std::string& dtype) {
        // A rejected shape is a Host-only capability result, not a launch failure.
        check_dispatch_dtype(dtype);
        try {
            auto plan = layout(p, b, h, k, e, dtype == "float8_e4m3fn");
            py::dict result;
            result["capacity_rows"] = plan.capacityRows;
            result["scale_packs"] = (h + 127) / 128;
            return py::object(result);
        } catch (const std::invalid_argument&) {
            return py::object(py::none());
        }
    });
    module.def("low_latency_comm_args", [](Session& session) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        if (session.capacity() != deep_ep::kEpServerDedupPeerBytes ||
            session.world() > deep_ep::DISPATCH_DEDUP_MAX_RANK_SIZE ||
            session.world() < 16 || session.world() % deep_ep::LlGroupForWorld(session.world()) != 0) {
            throw std::invalid_argument("LL requires exactly 8 GiB of symmetric workspace");
        }
        pin_session_cores(session);
        deep_ep::CommArgs args{};
        args.rank = session.rank();
        args.rankSize = session.world();
        args.localRank = session.rank() % deep_ep::LlGroupForWorld(session.world());
        args.localRankSize = deep_ep::LlGroupForWorld(session.world());
        args.reserved0 = deep_ep::kNumRanksPerHost;
        args.peerMemBytes = session.capacity();
        args.fftsVal = shmemx_get_ffts_config();
        if (session.ll_rdma()) {
            args.reservedAddresses[0] = static_cast<uint8_t*>(session.workspace());
            args.reservedAddresses[1] = static_cast<uint8_t*>(session.ll_rdma_stage());
            args.reservedConfig[0] = deep_ep::rdma::kBackend;
            args.reservedConfig[1] = session.ll_rdma_qps();
        }
        for (int peer = 0; peer < session.world(); ++peer) {
            args.peerMems[peer] = peer_buffer_base(session, peer);
            if (session.ll_rdma() && !deep_ep::LlSameHost(session.rank(), peer)) continue;
            if (args.peerMems[peer] == nullptr || reinterpret_cast<uintptr_t>(args.peerMems[peer]) % 32) {
                throw std::invalid_argument("LL requires aligned MTE mappings for every peer");
            }
        }
        return bytes(args);
    });
    module.def("low_latency_combine_tiling",
               [](int p, int b, int h, int k, int e, const std::string& dtype, int num_max_tokens_per_rank, bool rdma) {
                   if (rdma) {
                       return bytes(ubx::v2_tiling(p, b, h, k, e, combine_dtype_code(dtype), num_max_tokens_per_rank,
                                                 query_ll_core_config().active, deep_ep::kNumRanksPerHost));
                   }
                   CombineDedup::Layout plan{};
                   if ((dtype != "bfloat16" && dtype != "float16") ||
                       !CombineDedup::BuildLayout(p, b, h, k, e, plan, query_ll_core_config().active)) {
                       throw std::invalid_argument("LL combine exceeds dtype, topology, UB or workspace limits");
                   }
                   CombineDedup::EpServerDedupCombineTilingData data{};
                   data.bs = b;
                   data.h = h;
                   data.numMaxTokensPerRank = num_max_tokens_per_rank;
                   data.topK = k;
                   data.moeExpertNum = e;
                   data.dtype = combine_dtype_code(dtype);
                   return bytes(data);
               },
               py::arg("world"), py::arg("tokens"), py::arg("hidden"), py::arg("topk"), py::arg("experts"),
               py::arg("dtype"), py::arg("capacity"), py::arg("rdma") = false);
    module.def("dispatch_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e, int rows,
                                          int capacity, const std::string& dtype, uintptr_t stream_address) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        check_dispatch_dtype(dtype);
        // DeepEP-Ascend carries no dtype enum: FP8 is exactly "scales present",
        // everything else is plain BF16 on the same kernel.
        const bool fp8 = dtype == "float8_e4m3fn";
        const uint32_t cores = pin_session_cores(session);
        const auto mode = fp8 ? deep_ep::EpServerDedupInputMode::PrequantizedFp8Packs
                              : deep_ep::EpServerDedupInputMode::Plain16;
        const auto plan = deep_ep::get_dispatch_buffer_size(
            uint32_t(session.world()), uint32_t(capacity), uint32_t(h), uint32_t(k), uint32_t(e), mode, cores);
        const uint32_t group = deep_ep::LlGroupForWorld(uint32_t(session.world()));
        const int64_t c = plan.capacityRows;
        if (!stream_address || rows < c || b <= 0 || b > int(capacity) ||
            session.capacity() != deep_ep::kEpServerDedupPeerBytes) {
            throw std::invalid_argument("Invalid LL token count, output capacity, workspace or stream");
        }
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        const bool dedup = deep_ep::EpServerDedupEnabled(uint32_t(capacity), uint32_t(session.world()));

        void* comm_args = get("comm_args", {sizeof(deep_ep::comm::CommArgs)}, "uint8");
        void* peer_ptrs = static_cast<uint8_t*>(comm_args) + offsetof(deep_ep::comm::CommArgs, peerMems);
        void* x = get("input", {b, h}, dtype.c_str());
        void* expert_ids = get("indices", {b, k}, "int32");
        void* expert_scales = get("weights", {b, k}, "float32");
        void* expand_x_out = get("output", {rows, h}, dtype.c_str());
        // Both formats occupy four bytes per 128 hidden elements. The kernel
        // transports opaque bytes: FP32 per-128 or packed E8M0 per-32.
        const char* scale_dtype = "int32";
        if (fp8) {
            auto torch = py::module_::import("torch");
            if (tensors["scales"].attr("dtype").is(torch.attr("float32"))) {
                scale_dtype = "float32";
            }
        }
        void* dynamic_scales_out = fp8 ? get("output_scales", {rows, (h + 127) / 128}, scale_dtype) : nullptr;
        void* expert_token_nums_out = get("expert_recv_counts", {e / session.world()}, "int64");
        void* send_counts_out = get("send_counts", {e}, "int32");
        void* token_type_out = get("token_type", {c}, "int32");
        void* destination_index_out = get("destination_index", {c, 3}, "int32");
        void* relay_read_index_out = get("relay_read_index", {c, k, 2}, "int32");
        // Per-rank send metadata is sized by the admitted capacity M, while the
        // kernel only fills the first B rows.
        void* source_mask_out = get("source_mask", {capacity}, "int32");
        void* source_weights_out = get("source_weights", {capacity, k}, "float32");
        void* input_scales = fp8 ? get("scales", {b, (h + 127) / 128}, scale_dtype) : nullptr;
        void* profile_out = reference_profile(tensors, cores);
        void* rdma_remote_base = session.ll_rdma() ? session.workspace() : nullptr;
        void* rdma_stage_base = session.ll_rdma() ? session.ll_rdma_stage() : nullptr;
        session.workspace();  // Validate the owning context before launch.
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        deep_ep::launch_dispatch(
            x, expert_ids, expert_scales, expand_x_out, dynamic_scales_out,
            expert_token_nums_out, send_counts_out, token_type_out, destination_index_out,
            relay_read_index_out, source_mask_out, source_weights_out, input_scales,
            peer_ptrs, rdma_remote_base, rdma_stage_base, profile_out,
            session.rank(), b, deep_ep::kNumRanksPerHost,
            session.ll_rdma() ? int(deep_ep::rdma::kBackend) : 0,
            session.ll_rdma() ? int(session.ll_rdma_qps()) : 0,
            int64_t(session.capacity()), shmemx_get_ffts_config(),
            session.world(), capacity, fp8 ? h : h * 2, k, e, int(cores),
            int(deep_ep::jit_num_ubuf_bytes_per_vec_core()),
            int(group), fp8, dedup, session.ll_rdma(),
            reinterpret_cast<void*>(stream_address));
    });
    module.def("combine_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e, int capacity,
                                         const std::string& dtype, uintptr_t stream_address) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        CombineDedup::Layout plan{};
        if (!stream_address || session.capacity() != deep_ep::kEpServerDedupPeerBytes ||
            (dtype != "bfloat16" && dtype != "float16") || b <= 0 || capacity < b) {
            throw std::invalid_argument("Invalid LL combine dtype, shape, workspace or stream");
        }
        uint32_t cores = 0;
        if (session.ll_rdma()) {
            cores = pin_session_cores(session);
            plan.capacityRows =
                ubx::v2_capacity(session.world(), b, h, k, e, cores, deep_ep::kNumRanksPerHost);
        } else if (!CombineDedup::BuildLayout(session.world(), b, h, k, e, plan, query_ll_core_config().active)) {
            throw std::invalid_argument("Invalid LL combine shape");
        }
        // Row metadata was sized by the Dispatch capacity M, while the kernel
        // still works row-by-row over the actual B tokens.
        const uint64_t world = uint64_t(session.world());
        const uint64_t perRankExperts = uint64_t(e) / world;
        const int64_t c = int64_t(world * uint64_t(capacity) *
                                  (uint64_t(k) < perRankExperts ? uint64_t(k) : perRankExperts));
        int32_t device = -1;
        if (session.ll_rdma() && aclrtGetDevice(&device) != 0)
            throw std::runtime_error("LL Combine requires a current NPU");
        auto get = [&](const char* name, Shape shape, const char* type) {
            auto value = py::reinterpret_borrow<py::object>(tensors[name]);
            auto* pointer = tensor(value, shape, type);
            if (session.ll_rdma() && (value.attr("device").attr("type").cast<std::string>() != "npu" ||
                                     value.attr("device").attr("index").cast<int>() != device))
                throw std::invalid_argument("LL Combine tensors must be on the current NPU");
            return pointer;
        };
        deepep::EpServerDedupCombineKernelArgs args{};
        args.commArgs = get("comm_args", {sizeof(deep_ep::CommArgs)}, "uint8");
        // Python validates rows against the dispatch handle. The device selects
        // rows below the actual receive count; C_B sizes metadata, not expert storage.
        const auto input_shape = tensors["input"].attr("shape").cast<Shape>();
        if (input_shape.size() != 2 || input_shape[0] < 0 || input_shape[1] != h) {
            throw std::invalid_argument("LL combine input must have shape [rows, H]");
        }
        args.expertOut = get("input", input_shape, dtype.c_str());
        args.destinationIndex = get("destination_index", {c, 3}, "int32");
        args.sendCounts = get("send_counts", {e}, "int32");
        args.expertScales = get("source_weights", {capacity, k}, "float32");
        args.tokenType = get("token_type", {c}, "int32");
        args.relayReadIndex = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMask = get("source_mask", {capacity}, "int32");
        args.yOut = get("output", {b, h}, dtype.c_str());
        args.tilingData = get("tiling", {sizeof(CombineDedup::EpServerDedupCombineTilingData)}, "uint8");
        session.workspace();
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        if (session.ll_rdma()) {
            const ubx::CombineArgs rdma_args{args.commArgs,     args.expertOut, args.destinationIndex, args.sendCounts,
                                             args.expertScales, args.tokenType, args.relayReadIndex,   args.sourceMask,
                                             args.yOut,         args.tilingData, cores};
            ubx_combine_launch(reinterpret_cast<void*>(stream_address), &rdma_args);
        } else {
            combine_server_dedup_kernel_do(reinterpret_cast<void*>(stream_address), &args);
        }
    });
}
}  // namespace deepep
