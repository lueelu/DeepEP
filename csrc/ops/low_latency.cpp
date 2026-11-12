// Copyright (c) 2026, Lu Lu

#include "ops/low_latency.hpp"
#include <pybind11/stl.h>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>
#include "runtime/session.hpp"
#include "runtime/ll_core_config.hpp"
#include "runtime/ll_jit.hpp"
#include "ubx/v2.hpp"
#include "../kernels/dispatch/low_latency/common/ll_rdma_layout.h"
#include "../kernels/dispatch/low_latency/common/ll_stage_timer_layout.h"
#include "../kernels/dispatch/low_latency/common/ll_reference_profile_layout.h"
#include "../kernels/dispatch/low_latency/common/ll_expert_timer_layout.h"
#include "../kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "../kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "../kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h"
#include "../kernels/combine/low_latency/combine_layout.h"
#include "../kernels/combine/low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "../kernels/dispatch/low_latency/launch.hpp"
#include "../kernels/combine/low_latency/launch.hpp"

namespace py = pybind11;
namespace deepep {
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
    if (session.ll_rdma() && !DispatchDedup::LlSameHost(session.rank(), peer)) return nullptr;
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
uint8_t* stage_timer(py::dict tensors, uint32_t cores, uint32_t& mode, bool allow_reference = false)
{
    mode = 1U;
    if (tensors.contains("ll_reference_profile") && !tensors["ll_reference_profile"].is_none()) {
        if (!allow_reference) throw std::invalid_argument("LL reference profiling supports dispatch only");
        if ((tensors.contains("ll_stage_timer") && !tensors["ll_stage_timer"].is_none()) ||
            (tensors.contains("ll_expert_timer") && !tensors["ll_expert_timer"].is_none()))
            throw std::invalid_argument("LL reference profiling cannot be combined with stage/expert tracing");
        if (!deepep_ll_reference_profile_enabled() || deepep_ll_stage_timer_enabled())
            throw std::invalid_argument("Rebuild with DEEPEP_LL_REFERENCE_PROFILE=ON and DEEPEP_LL_STAGE_TIMER=OFF");
        auto profile = py::reinterpret_borrow<py::object>(tensors["ll_reference_profile"]);
        auto device = profile.attr("device");
        if (device.attr("type").cast<std::string>() != "npu" ||
            !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
            throw std::invalid_argument("LL reference profile must be on the input NPU");
        auto* pointer = tensor(profile, {cores, LlReferenceProfile::kSlots, LlReferenceProfile::kFields}, "int64");
        if (reinterpret_cast<uintptr_t>(pointer) % 512U != 0U)
            throw std::invalid_argument("LL reference profile requires 512-byte alignment");
        mode = 3U;  // Private launch selector; the reference record's boundary mode remains 1.
        return pointer;
    }
    if (!tensors.contains("ll_stage_timer") || tensors["ll_stage_timer"].is_none()) return nullptr;
    if (!deepep_ll_stage_timer_enabled()) throw std::invalid_argument("Legacy LL stage tracing was removed; use LL reference profiling");
    if (tensors.contains("ll_stage_timer_mode")) mode = tensors["ll_stage_timer_mode"].cast<uint32_t>();
    if (mode != 1U && mode != 2U) throw std::invalid_argument("Invalid LL timer mode");
    auto timer = py::reinterpret_borrow<py::object>(tensors["ll_stage_timer"]);
    auto device = timer.attr("device");
    if (device.attr("type").cast<std::string>() != "npu" ||
        !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
        throw std::invalid_argument("LL timer must be on the input NPU");
    auto* pointer = tensor(timer, {cores, LlStageTimer::kSlots, LlStageTimer::kFields}, "int64");
    if (reinterpret_cast<uintptr_t>(pointer) % 512U != 0U)
        throw std::invalid_argument("LL timer requires 512-byte alignment for independent core writeback");
    return pointer;
}
uint8_t* expert_timer(py::dict tensors, uint32_t cores, uint32_t experts, uint8_t* stage)
{
    if (!tensors.contains("ll_expert_timer") || tensors["ll_expert_timer"].is_none()) return nullptr;
    if (!stage || !deepep_ll_stage_timer_enabled())
        throw std::invalid_argument("Expert diagnostics require stage capture and timer ON");
    auto value=py::reinterpret_borrow<py::object>(tensors["ll_expert_timer"]);
    auto device=value.attr("device");
    if (device.attr("type").cast<std::string>() != "npu" ||
        !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
        throw std::invalid_argument("Expert timer must be on the input NPU");
    auto* pointer=tensor(value, {cores, LlExpertTimer::kHeaderRows + experts*LlExpertTimer::StageCount,
                                LlExpertTimer::kFields}, "int64");
    if (reinterpret_cast<uintptr_t>(pointer)%512U)
        throw std::invalid_argument("Expert timer requires 512-byte alignment");
    return pointer;
}
DispatchDedup::EpMemoryServerDedupLayout layout(int p, int b, int h, int k, int e, bool fp8)
{
    const uint32_t coreNum = query_ll_core_config().active;
    DispatchDedup::EpMemoryServerDedupLayout result{};
    auto mode = fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs
                    : DispatchDedup::EpServerDedupInputMode::Plain16;
    if (b <= 0 || b > 65536 ||
        DispatchDedup::BuildDispatchServerDedupLayout(p, b, h, k, e, &result, mode, coreNum) !=
            DispatchDedup::DISPATCH_DEDUP_SUCCESS ||
        !DispatchDedup::DispatchDedupPacketUbFits(result, b, k, e, p, coreNum) || result.capacityRows > INT32_MAX) {
        throw std::invalid_argument("LL dispatch exceeds topology, shape, UB or workspace limits");
    }
    return result;
}
template <class T>
py::bytes bytes(const T& data)
{
    return py::bytes(reinterpret_cast<const char*>(&data), sizeof(data));
}
uint8_t dtype_code(const std::string& dtype)
{
    if (dtype == "bfloat16") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16;
    if (dtype == "float16") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
    if (dtype == "float8_e4m3fn") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3;
    throw std::invalid_argument("LL dispatch supports BF16/FP16 or packed FP8 E4M3FN");
}
static_assert(DispatchDedup::kLlDispatchFp16 == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16 &&
              DispatchDedup::kLlDispatchBf16 == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16 &&
              DispatchDedup::kLlDispatchE4M3 == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 &&
              DispatchDedup::kLlDispatchE5M2 == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E5M2,
              "LL dispatch variant dtype values must match tiling");
static_assert(DispatchDedup::kLlDispatchPlain16 == uint8_t(DispatchDedup::EpServerDedupInputMode::Plain16) &&
              DispatchDedup::kLlDispatchQuantize16 == uint8_t(DispatchDedup::EpServerDedupInputMode::Quantize16) &&
              DispatchDedup::kLlDispatchPrequantized == uint8_t(DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs),
              "LL dispatch variant input mode values must match tiling");

// Both device tiling construction and Host launch selection use these fields.
// Selection needs no device readback and does not synchronize the stream.
DispatchDedup::EpServerDedupTilingData dispatch_format_tiling(int world, const std::string& dtype)
{
    DispatchDedup::EpServerDedupTilingData data{};
    const bool fp8 = dtype == "float8_e4m3fn";
    data.tilingKey = DispatchDedup::LlTilingKeyForWorld(world);
    data.dtype = dtype_code(dtype);
    data.quantMode = fp8 ? 4 : 0;
    data.inputMode = static_cast<uint8_t>(fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs
                                            : DispatchDedup::EpServerDedupInputMode::Plain16);
    data.expandXOutDtype = data.dtype;
    return data;
}

// The JIT identity comes from the same Host description as the tiling builder.
// `capacity` is the per-rank token bound Python selected for this handle; the
// dedup decision must match the value the device reads from that tiling.
LlDispatchJitSpec dispatch_jit_spec(int world, int b, int h, int k, int e, const std::string& dtype, int capacity,
                                   uint32_t cores)
{
    const auto format = dispatch_format_tiling(world, dtype);
    LlDispatchJitSpec spec;
    spec.bs = uint32_t(b);
    spec.h = uint32_t(h);
    spec.topK = uint32_t(k);
    spec.experts = uint32_t(e);
    spec.cores = cores;
    spec.world = uint32_t(world);
    spec.inputClass = DispatchDedup::LlDispatchInputClass(format.dtype, format.quantMode, format.expandXOutDtype,
                                                          format.inputMode);
    if (spec.inputClass == DispatchDedup::kLlDispatchInvalidVariant)
        throw std::invalid_argument("Unsupported LL dispatch format");
    spec.group = format.tilingKey;
    spec.inputMode = format.inputMode;
    spec.e4m3 = dtype == "float8_e4m3fn";
    spec.dedup = DispatchDedup::EpServerDedupEnabled(uint32_t(capacity), uint32_t(b), uint32_t(world));
    return spec;
}

LlCombineJitSpec combine_jit_spec(int world, int b, int h, int k, int e, const std::string& dtype, int capacity,
                                 uint32_t cores)
{
    LlCombineJitSpec spec;
    spec.bs = uint32_t(b);
    spec.h = uint32_t(h);
    spec.topK = uint32_t(k);
    spec.experts = uint32_t(e);
    spec.cores = cores;
    spec.world = uint32_t(world);
    spec.group = DispatchDedup::LlTilingKeyForWorld(uint32_t(world));
    spec.bf16 = dtype == "bfloat16";
    spec.dedup = DispatchDedup::EpServerDedupEnabled(uint32_t(capacity), uint32_t(b), uint32_t(world));
    return spec;
}
}  // namespace

void bind_low_latency(py::module_& module)
{
    if (deepep_ll_core_protocol_version() != DispatchDedup::kLlCoreProtocolVersion)
        throw std::runtime_error("LL Host/device core protocol mismatch; rebuild and reinstall the complete wheel");
    module.attr("low_latency_core_protocol") = DispatchDedup::kLlCoreProtocolVersion;
    module.attr("low_latency_roce_qps_per_peer") = 0; // actual active physical cores
    module.attr("low_latency_implementation") = "ndr-hns1825";
#ifdef DEEPEP_LL_JIT
    module.attr("low_latency_jit_enabled") = true;
#else
    module.attr("low_latency_jit_enabled") = false;
#endif
    module.def("low_latency_jit_available", &deepep::ll_jit_available);
    module.def("low_latency_jit_prepare_dispatch",
               [](Session& session, int b, int h, int k, int e, const std::string& dtype, int capacity) {
                   dtype_code(dtype);
                   layout(session.world(), b, h, k, e, dtype == "float8_e4m3fn");
                   const uint32_t cores = pin_session_cores(session);
                   ll_jit_prepare_dispatch(dispatch_jit_spec(session.world(), b, h, k, e, dtype, capacity, cores));
               },
               py::arg("session"), py::arg("tokens"), py::arg("hidden"), py::arg("topk"), py::arg("experts"),
               py::arg("dtype"), py::arg("capacity"));
    module.def("low_latency_jit_prepare_combine",
               [](Session& session, int b, int h, int k, int e, const std::string& dtype, int capacity, bool rdma) {
                   if (dtype != "bfloat16" && dtype != "float16")
                       throw std::invalid_argument("LL combine supports BF16/FP16");
                   if (rdma != session.ll_rdma())
                       throw std::invalid_argument("LL combine JIT backend differs from the Session transport");
                   uint32_t cores = 0;
                   if (rdma) {
                       cores = pin_session_cores(session);
                   } else {
                       CombineDedup::Layout plan{};
                       if (!CombineDedup::BuildLayout(session.world(), b, h, k, e, plan))
                           throw std::invalid_argument("Invalid LL combine shape");
                       cores = CombineDedup::kEpServerDedupCombineCoreNum;
                   }
                   ll_jit_prepare_combine(combine_jit_spec(session.world(), b, h, k, e, dtype, capacity, cores), rdma);
               },
               py::arg("session"), py::arg("tokens"), py::arg("hidden"), py::arg("topk"), py::arg("experts"),
               py::arg("dtype"), py::arg("capacity"), py::arg("rdma"));
#ifdef DEEPEP_LL_RDMA
    if (!deepep_ll_rdma_enabled()) throw std::runtime_error("LL Host/device RDMA build mismatch; rebuild complete wheel");
    module.attr("low_latency_rdma_enabled") = true;
#else
    if (deepep_ll_rdma_enabled()) throw std::runtime_error("LL Host/device RDMA build mismatch; rebuild complete wheel");
    module.attr("low_latency_rdma_enabled") = false;
#endif
    module.attr("low_latency_stage_timer_enabled") = bool(deepep_ll_stage_timer_enabled());
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    constexpr uint32_t reference_enabled = 1U;
#else
    constexpr uint32_t reference_enabled = 0U;
#endif
    if (deepep_ll_reference_profile_enabled() != reference_enabled ||
        deepep_ll_reference_profile_schema() != LlReferenceProfile::kSchema ||
        (reference_enabled && deepep_ll_stage_timer_enabled()))
        throw std::runtime_error("LL Host/device reference profile mismatch; rebuild and reinstall the complete wheel");
    module.attr("low_latency_reference_profile_enabled") = bool(reference_enabled);
    module.def("low_latency_reference_profile_layout", []() {
        py::dict result;
        result["schema"] = LlReferenceProfile::kSchema;
        result["slots"] = LlReferenceProfile::kSlots;
        result["fields"] = LlReferenceProfile::kFields;
        std::vector<std::string> names;
        for (const auto* name : LlReferenceProfile::kNames) names.emplace_back(name);
        result["words"] = names;
        return result;
    });
    module.def("low_latency_expert_timer_layout", []() {
        py::dict result;
        result["schema"]=LlExpertTimer::kSchema;
        result["header_rows"]=LlExpertTimer::kHeaderRows;
        result["fields"]=LlExpertTimer::kFields;
        std::vector<std::string> names;
        for (const auto* name : LlExpertTimer::kNames) names.emplace_back(name);
        result["stages"]=names;
        return result;
    });
    module.def("low_latency_stage_timer_layout", []() {
        py::dict result;
        result["schema"] = LlStageTimer::kSchema;
        result["slots"] = LlStageTimer::kSlots;
        result["fields"] = LlStageTimer::kFields;
        std::vector<std::string> names;
        for (const auto* name : LlStageTimer::kNames) names.emplace_back(name);
        result["stages"] = names;
        return result;
    });
    module.def("low_latency_core_config", []() {
        const auto cfg = query_ll_core_config();
        py::dict result;
        result["platform_aiv"] = cfg.platform; result["available_aiv"] = cfg.available;
        result["active_aiv"] = cfg.active; result["send_cores"] = cfg.active / 2U;
        result["recv_cores"] = cfg.active / 2U; result["device"] = cfg.device;
        result["runtime_device_queried"] = cfg.device >= 0;
        result["control_capacity_aiv"] = DispatchDedup::kLlMaxCoreNum;
        return result;
    });
    for (uint32_t world : {16U, 32U, 64U, 128U, 256U})
        if (deepep_ll_ranks_per_group(world) != DispatchDedup::LlGroupForWorld(world))
            throw std::runtime_error("LL Host/device tiling-key policy mismatch; rebuild the complete wheel");
    module.def("low_latency_ranks_per_group", [](uint32_t world) { return DispatchDedup::LlGroupForWorld(world); });
    module.def("low_latency_layout", [](int p, int b, int h, int k, int e, const std::string& dtype) {
        // A rejected shape is a Host-only capability result, not a launch failure.
        dtype_code(dtype);
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
        if (session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            session.world() > DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE ||
            session.world() < 16 || session.world() % DispatchDedup::LlGroupForWorld(session.world()) != 0) {
            throw std::invalid_argument("LL requires exactly 8 GiB of symmetric workspace");
        }
        pin_session_cores(session);
        DispatchDedup::CommArgs args{};
        args.rank = session.rank();
        args.rankSize = session.world();
        args.localRank = session.rank() % DispatchDedup::LlGroupForWorld(session.world());
        args.localRankSize = DispatchDedup::LlGroupForWorld(session.world());
        args.reserved0 = DispatchDedup::kLlPhysicalRanksPerHost;
        args.peerMemBytes = session.capacity();
        args.fftsVal = shmemx_get_ffts_config();
        if (session.ll_rdma()) {
            args.reservedAddresses[0] = static_cast<uint8_t*>(session.workspace());
            args.reservedAddresses[1] = static_cast<uint8_t*>(session.ll_rdma_stage());
            args.reservedConfig[0] = LlRdma::kBackend;
            args.reservedConfig[1] = session.ll_rdma_qps();
        }
        for (int peer = 0; peer < session.world(); ++peer) {
            args.peerMems[peer] = peer_buffer_base(session, peer);
            if (session.ll_rdma() && !DispatchDedup::LlSameHost(session.rank(), peer)) continue;
            if (args.peerMems[peer] == nullptr || reinterpret_cast<uintptr_t>(args.peerMems[peer]) % 32) {
                throw std::invalid_argument("LL requires aligned MTE mappings for every peer");
            }
        }
        return bytes(args);
    });
    module.def("low_latency_dispatch_tiling", [](int p, int b, int h, int k, int e, const std::string& dtype,
                                                 int num_max_tokens_per_rank) {
        const bool fp8 = dtype == "float8_e4m3fn";
        layout(p, b, h, k, e, fp8);
        auto data = dispatch_format_tiling(p, dtype);
        data.bs = b;
        data.h = h;
        data.numMaxTokensPerRank = num_max_tokens_per_rank;
        data.topK = k;
        data.moeExpertNum = e;
        data.expertTokenNumsType = DispatchDedup::kLlExpertTokenNumsType;
        return bytes(data);
    });
    module.def("low_latency_combine_tiling",
               [](int p, int b, int h, int k, int e, const std::string& dtype, int num_max_tokens_per_rank, bool rdma) {
                   if (rdma) {
                       return bytes(ubx::v2_tiling(p, b, h, k, e, dtype_code(dtype), num_max_tokens_per_rank,
                                                 query_ll_core_config().active, DispatchDedup::kLlPhysicalRanksPerHost));
                   }
                   CombineDedup::Layout plan{};
                   if ((dtype != "bfloat16" && dtype != "float16") || !CombineDedup::BuildLayout(p, b, h, k, e, plan)) {
                       throw std::invalid_argument("LL combine exceeds dtype, topology, UB or workspace limits");
                   }
                   CombineDedup::EpServerDedupCombineTilingData data{};
                   data.bs = b;
                   data.h = h;
                   data.numMaxTokensPerRank = num_max_tokens_per_rank;
                   data.topK = k;
                   data.moeExpertNum = e;
                   data.dtype = dtype_code(dtype);
                   return bytes(data);
               },
               py::arg("world"), py::arg("tokens"), py::arg("hidden"), py::arg("topk"), py::arg("experts"),
               py::arg("dtype"), py::arg("capacity"), py::arg("rdma") = false);
    module.def("dispatch_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e, int rows,
                                          const std::string& dtype, uintptr_t stream_address, bool jit,
                                          int capacity) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        dtype_code(dtype);
        const bool fp8 = dtype == "float8_e4m3fn";
        const auto plan = layout(session.world(), b, h, k, e, fp8);
        const auto format = dispatch_format_tiling(session.world(), dtype);
        if (DispatchDedup::LlDispatchInputClass(format.dtype, format.quantMode, format.expandXOutDtype,
                                                format.inputMode) == DispatchDedup::kLlDispatchInvalidVariant)
            throw std::invalid_argument("Unsupported LL dispatch format");
        const int64_t c = plan.capacityRows;
        if (!stream_address || rows < c || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes) {
            throw std::invalid_argument("Invalid LL output capacity, workspace or stream");
        }
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupKernelArgs args{};
        args.coreNum = pin_session_cores(session);
        args.stageTimer = stage_timer(tensors, args.coreNum, args.timerMode, true);
        args.expertTimer = expert_timer(tensors, args.coreNum, e, args.stageTimer);
        args.commArgs = get("comm_args", {sizeof(DispatchDedup::CommArgs)}, "uint8");
        args.x = get("input", {b, h}, dtype.c_str());
        args.expertIds = get("indices", {b, k}, "int32");
        args.expertScales = get("weights", {b, k}, "float32");
        args.expandXOut = get("output", {rows, h}, dtype.c_str());
        // Both formats occupy four bytes per 128 hidden elements. The kernel
        // transports opaque bytes: FP32 per-128 or packed E8M0 per-32.
        const char* scale_dtype = "int32";
        if (fp8) {
            auto torch = py::module_::import("torch");
            if (tensors["scales"].attr("dtype").is(torch.attr("float32"))) {
                scale_dtype = "float32";
            }
        }
        args.dynamicScalesOut = fp8 ? get("output_scales", {rows, (h + 127) / 128}, scale_dtype) : nullptr;
        args.expertTokenNumsOut = get("expert_recv_counts", {e / session.world()}, "int64");
        args.sendCountsOut = get("send_counts", {e}, "int32");
        args.tokenTypeOut = get("token_type", {c}, "int32");
        args.destinationIndexOut = get("destination_index", {c, 3}, "int32");
        args.relayReadIndexOut = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMaskOut = get("source_mask", {b}, "int32");
        args.expertScalesOut = get("source_weights", {b, k}, "float32");
        args.tilingData = get("tiling", {sizeof(DispatchDedup::EpServerDedupTilingData)}, "uint8");
        args.inputScales = fp8 ? get("scales", {b, (h + 127) / 128}, scale_dtype) : nullptr;
        session.workspace();  // Validate the owning context before launch.
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        if (jit) {
            if (capacity <= 0) throw std::invalid_argument("LL JIT dispatch requires the per-rank capacity");
            ll_jit_launch_dispatch(reinterpret_cast<void*>(stream_address), args,
                                   dispatch_jit_spec(session.world(), b, h, k, e, dtype, capacity, args.coreNum));
        } else {
            dispatch_server_dedup_kernel_do(reinterpret_cast<void*>(stream_address), &args);
        }
    },
               py::arg("session"), py::arg("tensors"), py::arg("b"), py::arg("h"), py::arg("k"), py::arg("e"),
               py::arg("rows"), py::arg("dtype"), py::arg("stream"), py::arg("jit") = false,
               py::arg("capacity") = 0);
    module.def("combine_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e,
                                         const std::string& dtype, uintptr_t stream_address, bool jit, int capacity) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        CombineDedup::Layout plan{};
        if (!stream_address || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            (dtype != "bfloat16" && dtype != "float16")) {
            throw std::invalid_argument("Invalid LL combine dtype, shape, workspace or stream");
        }
        uint32_t cores = 0;
        if (session.ll_rdma()) {
            cores = pin_session_cores(session);
            plan.capacityRows =
                ubx::v2_capacity(session.world(), b, h, k, e, cores, DispatchDedup::kLlPhysicalRanksPerHost);
        } else if (!CombineDedup::BuildLayout(session.world(), b, h, k, e, plan)) {
            throw std::invalid_argument("Invalid LL combine shape");
        }
        const int64_t c = plan.capacityRows;
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
        args.commArgs = get("comm_args", {sizeof(DispatchDedup::CommArgs)}, "uint8");
        // Python validates rows against the dispatch handle. The device selects
        // rows below the actual receive count; C_B sizes metadata, not expert storage.
        const auto input_shape = tensors["input"].attr("shape").cast<Shape>();
        if (input_shape.size() != 2 || input_shape[0] < 0 || input_shape[1] != h) {
            throw std::invalid_argument("LL combine input must have shape [rows, H]");
        }
        args.expertOut = get("input", input_shape, dtype.c_str());
        args.destinationIndex = get("destination_index", {c, 3}, "int32");
        args.sendCounts = get("send_counts", {e}, "int32");
        args.expertScales = get("source_weights", {b, k}, "float32");
        args.tokenType = get("token_type", {c}, "int32");
        args.relayReadIndex = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMask = get("source_mask", {b}, "int32");
        args.yOut = get("output", {b, h}, dtype.c_str());
        args.tilingData = get("tiling", {sizeof(CombineDedup::EpServerDedupCombineTilingData)}, "uint8");
        session.workspace();
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        if (jit && capacity <= 0) throw std::invalid_argument("LL JIT combine requires the per-rank capacity");
        if (session.ll_rdma()) {
            const ubx::CombineArgs rdma_args{args.commArgs,     args.expertOut, args.destinationIndex, args.sendCounts,
                                             args.expertScales, args.tokenType, args.relayReadIndex,   args.sourceMask,
                                             args.yOut,         args.tilingData, cores};
            if (jit) {
                ll_jit_launch_combine_ubx(reinterpret_cast<void*>(stream_address), rdma_args,
                                          combine_jit_spec(session.world(), b, h, k, e, dtype, capacity, cores));
            } else {
                ubx_combine_launch(reinterpret_cast<void*>(stream_address), &rdma_args);
            }
        } else if (jit) {
            ll_jit_launch_combine(reinterpret_cast<void*>(stream_address), args,
                                  combine_jit_spec(session.world(), b, h, k, e, dtype, capacity,
                                                   CombineDedup::kEpServerDedupCombineCoreNum));
        } else {
            combine_server_dedup_kernel_do(reinterpret_cast<void*>(stream_address), &args);
        }
    },
               py::arg("session"), py::arg("tensors"), py::arg("b"), py::arg("h"), py::arg("k"), py::arg("e"),
               py::arg("dtype"), py::arg("stream"), py::arg("jit") = false, py::arg("capacity") = 0);
}
}  // namespace deepep
