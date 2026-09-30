// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#include "ops/device_phase_timing.hpp"

#include "ops/notify.hpp"
#include <pybind11/stl.h>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernels/notify/launch.hpp"
#include "kernels/notify/tiling.hpp"
#include "kernels/dispatch/tiling.hpp"
#include "runtime/session.hpp"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"

namespace py = pybind11;
namespace deepep {
namespace {
ascend_deepep::NotifyTiling tiling(int rank, int world, int tokens, int topk, int experts)
{
    if (world < 2 || world > 256 || (world & (world - 1)) || rank < 0 || rank >= world || tokens < 1 ||
        tokens > 10240 || topk < 1 || topk > 16 || experts < world || experts > 1024 || experts % world ||
        uint64_t(world) * world * tokens * topk >= (1ULL << 31)) {
        throw std::invalid_argument(
            "Notify requires EP2..256 powers of two, T1..10240, K1..16, "
            "E<=1024 divisible by EP, and EP*EP*T*K < 2**31");
    }
    return ascend_deepep::MakeNotifyTiling(rank, world, tokens, topk, experts, 256, 7168, 32);
}
using Shape = std::vector<int64_t>;
using Specs = std::map<std::string, std::pair<Shape, std::string>>;
Specs specs(const ascend_deepep::NotifyTiling& t)
{
    const int64_t p = t.world, n = t.tokens, k = t.topk, e = t.experts;
    const int64_t f = t.forward_capacity, b = t.backward_capacity, c = t.chunks, l = std::min(p, int64_t(8));
    Specs result = {{"meta", {{n, k, 6}, "int32"}},
                    {"hist", {{((n + 15) / 16) * ((e + p + 15) / 16 * 16)}, "int32"}},
                    {"bases", {{e}, "int32"}},
                    {"prefix", {{p + 1}, "int32"}},
                    {"counts", {{e}, "int32"}},
                    {"gateways", {{p}, "int32"}},
                    {"local_dst", {{n, k}, "int32"}},
                    {"server_mask", {{n}, "int64"}},
                    {"dst", {{n, k}, "int32"}},
                    {"forward_list", {{p, f, 6}, "int32"}},
                    {"forward_counts", {{p}, "int32"}},
                    {"backward_list", {{l, b, 3}, "int64"}},
                    {"backward_counts", {{l}, "int32"}},
                    {"chunk_ranges", {{p, c, 2}, "int32"}},
                    {"chunk_masks", {{p / std::min(p, int64_t(64)), c}, "int32"}},
                    {"rank_rows", {{p}, "int32"}},
                    {"gather_rows", {{p}, "int32"}},
                    {"expert_totals", {{e}, "int32"}},
                    // R is only known after Notify. Every route can land on one rank.
                    {"weight_return_meta", {{b}, "int32"}}};
    if (t.world <= 256U)
        result["_peer_plan"] = {{int64_t(ascend_deepep::DispatchPlanBytes(t.world, t.tokens, t.topk))}, "uint8"};
    return result;
}
uint8_t* tensor_pointer(py::handle tensor, const Shape& shape, const std::string& dtype, int device)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(tensor).cast<bool>()) {
        throw std::invalid_argument("Notify arguments must be framework tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(tensor);
    if (obj.attr("device").attr("type").cast<std::string>() != "npu" ||
        obj.attr("device").attr("index").cast<int>() != device || !obj.attr("is_contiguous")().cast<bool>() ||
        obj.attr("shape").cast<Shape>() != shape || !obj.attr("dtype").is(torch.attr(dtype.c_str()))) {
        throw std::invalid_argument("Notify tensor shape, dtype, contiguity or device mismatch");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}
}  // namespace

void bind_notify(py::module_& module)
{
    module.def("notify_layout", [](int rank, int world, int tokens, int topk, int experts) {
        const auto t = tiling(rank, world, tokens, topk, experts);
        py::dict result;
        result["workspace_bytes"] = t.workspace_bytes;
        result["tensors"] = py::cast(specs(t));
        return result;
    });
    module.def("notify", [](Session& session, py::object indices, py::dict outputs, int tokens, int topk, int experts,
                            uintptr_t stream_address) {
        auto t = tiling(session.rank(), session.world(), tokens, topk, experts);
        auto* workspace = static_cast<uint8_t*>(session.workspace());
        if (t.workspace_bytes > session.capacity() / 2U || stream_address == 0) {
            throw std::invalid_argument("Notify workspace too small or invalid communication stream");
        }
        int32_t device = -1;
        if (aclrtGetDevice(&device) != 0) {
            throw std::runtime_error("Cannot obtain current ACL device");
        }
        auto* input = tensor_pointer(indices, {tokens, topk}, "int64", device);
        std::map<std::string, uint8_t*> ptr;
        for (const auto& entry : specs(t)) {
            ptr[entry.first] =
                tensor_pointer(outputs[py::str(entry.first)], entry.second.first, entry.second.second, device);
        }
        // Mapping validation happens on every rank before device collectives (Python preflight).
        auto stream = reinterpret_cast<aclrtStream>(stream_address);
        const bool measure = outputs.contains("kernel_timing") && outputs["kernel_timing"].cast<bool>();
        py::gil_scoped_release release;
        DevicePhaseTiming timing(measure), plan_timing(measure);
        timing.mark(0, stream);
        session.invalidate_dispatch_controls();
        notify_full_kernel_do(stream, input, workspace, ptr.at("meta"), ptr.at("hist"), ptr.at("bases"),
                              ptr.at("prefix"), ptr.at("counts"), ptr.at("gateways"), ptr.at("local_dst"),
                              ptr.at("server_mask"), ptr.at("dst"), ptr.at("forward_list"), ptr.at("forward_counts"),
                              ptr.at("backward_list"), ptr.at("backward_counts"), ptr.at("chunk_ranges"),
                              ptr.at("chunk_masks"), ptr.at("rank_rows"), ptr.at("gather_rows"),
                              ptr.at("weight_return_meta"), reinterpret_cast<uint8_t*>(&t));
        timing.mark(1, stream);
        notify_dispatch_barrier_kernel_do(stream, t.num_cores);
        // The post-Notify barrier retires count exchange. Its first two 512B
        // slots can now hold admission status without enlarging the workspace.
        auto plan_t = ascend_deepep::MakeDispatchTiling(t.rank, t.world, t.tokens, t.topk, t.experts, 1U, true);
        plan_t.sync = t.count_send;
        plan_timing.mark(0, stream);
        if (t.world <= 256U) {
            const auto plan_bytes = ascend_deepep::DispatchPlanBytes(t.world, t.tokens, t.topk);
            if (aclrtMemsetAsync(ptr.at("_peer_plan"), plan_bytes, 0, plan_bytes, stream) != 0)
                throw std::runtime_error("Cannot initialize Notify peer-table mailbox");
        }
        plan_timing.mark(1, stream);
        if (t.world <= 256U)
            notify_dispatch_peer_plan_kernel_do(stream, workspace, ptr.at("dst"), ptr.at("forward_list"),
                                                ptr.at("forward_counts"), ptr.at("_peer_plan"),
                                                reinterpret_cast<uint8_t*>(&plan_t), t.address_stride);
        plan_timing.mark(2, stream);
        plan_timing.mark(3, stream);
        timing.mark(2, stream);
        const auto bytes = uint64_t(experts) * sizeof(int32_t);
        if (aclrtMemcpyAsync(ptr.at("expert_totals"), bytes, workspace + t.expert_totals, bytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream) != 0) {
            throw std::runtime_error("Cannot preserve notify expert totals");
        }
        timing.mark(3, stream);
        if (t.world <= 256U) {
            if (aclrtSynchronizeStream(stream) != 0) throw std::runtime_error("Notify peer-table preparation failed");
            uint32_t status = 0;
            if (aclrtMemcpy(&status, sizeof(status), workspace + plan_t.sync + 512U, sizeof(status),
                            ACL_MEMCPY_DEVICE_TO_HOST) != 0)
                throw std::runtime_error("Cannot read Notify peer-table admission status");
            if (status) throw std::invalid_argument("Notify peer-table admission failed: SQ capacity or invalid plan");
        }
        return std::make_pair(timing.finish(stream), plan_timing.finish(stream));
    });
    module.def("notify_check_mapping", [](Session& session) {
        void* workspace = session.workspace();
        for (int peer = 0; peer < session.world(); ++peer) {
            if (aclshmem_ptr(workspace, peer) == nullptr) {
                throw std::runtime_error("Notify requires MTE-mapped SHMEM workspace for every peer");
            }
        }
    });
}
}  // namespace deepep
