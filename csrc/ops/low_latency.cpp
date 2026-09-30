// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#include "ops/low_latency.hpp"
#include <pybind11/stl.h>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>
#include "runtime/session.hpp"
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
DispatchDedup::EpMemoryServerDedupLayout layout(int p, int b, int h, int k, int e, bool fp8)
{
    DispatchDedup::EpMemoryServerDedupLayout result{};
    auto mode = fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs
                    : DispatchDedup::EpServerDedupInputMode::Plain16;
    if (b <= 0 || b > 65536 ||
        DispatchDedup::BuildDispatchServerDedupLayout(p, b, h, k, e, &result, mode) !=
            DispatchDedup::DISPATCH_DEDUP_SUCCESS ||
        !DispatchDedup::DispatchDedupPacketUbFits(result, b, k, e, p) || result.capacityRows > INT32_MAX) {
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
}  // namespace

void bind_low_latency(py::module_& module)
{
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
        if (session.capacity() != DispatchDedup::kEpServerDedupPeerBytes || session.world() > 128) {
            throw std::invalid_argument("LL requires exactly 8 GiB of symmetric workspace");
        }
        DispatchDedup::CommArgs args{};
        args.rank = session.rank();
        args.rankSize = session.world();
        args.localRank = session.rank() % 8;
        args.localRankSize = 8;
        args.peerMemBytes = session.capacity();
        args.fftsVal = shmemx_get_ffts_config();
        for (int peer = 0; peer < session.world(); ++peer) {
            args.peerMems[peer] = static_cast<uint8_t*>(aclshmem_ptr(session.workspace(), peer));
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
        DispatchDedup::EpServerDedupTilingData data{};
        data.bs = b;
        data.h = h;
        data.numMaxTokensPerRank = num_max_tokens_per_rank;
        data.topK = k;
        data.moeExpertNum = e;
        data.expertTokenNumsType = 1;
        data.dtype = dtype_code(dtype);
        data.quantMode = fp8 ? 4 : 0;
        data.inputMode = static_cast<uint8_t>(fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs
                                                  : DispatchDedup::EpServerDedupInputMode::Plain16);
        data.expandXOutDtype = data.dtype;
        return bytes(data);
    });
    module.def("low_latency_combine_tiling",
               [](int p, int b, int h, int k, int e, const std::string& dtype, int num_max_tokens_per_rank) {
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
               });
    module.def("dispatch_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e, int rows,
                                          const std::string& dtype, uintptr_t stream_address) {
        dtype_code(dtype);
        const bool fp8 = dtype == "float8_e4m3fn";
        const auto plan = layout(session.world(), b, h, k, e, fp8);
        const int64_t c = plan.capacityRows;
        if (!stream_address || rows < c || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes) {
            throw std::invalid_argument("Invalid LL output capacity, workspace or stream");
        }
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupKernelArgs args{};
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
        dispatch_server_dedup_kernel_do(reinterpret_cast<void*>(stream_address), &args);
    });
    module.def("combine_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e,
                                         const std::string& dtype, uintptr_t stream_address) {
        CombineDedup::Layout plan{};
        if (!stream_address || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            (dtype != "bfloat16" && dtype != "float16") ||
            !CombineDedup::BuildLayout(session.world(), b, h, k, e, plan)) {
            throw std::invalid_argument("Invalid LL combine dtype, shape, workspace or stream");
        }
        const int64_t c = plan.capacityRows;
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
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
        combine_server_dedup_kernel_do(reinterpret_cast<void*>(stream_address), &args);
    });
}
}  // namespace deepep
