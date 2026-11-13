// Copyright (c) 2026, Lu Lu

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
#include "kernels/notify/profile.hpp"
#include "kernels/dispatch/tiling.hpp"
#include "runtime/session.hpp"
#include "runtime/jit_runtime.hpp"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"

#ifndef DEEPEP_JIT_PI_ON_CHIP
#define DEEPEP_JIT_PI_ON_CHIP 0
#endif

namespace py = pybind11;
namespace deepep {
namespace {
ascend_deepep::NotifyTiling tiling(int rank, int world, int tokens, int topk, int experts, int chunk_tokens,
                                   uint32_t num_cores)
{
    if (world < 2 || world > 512 || (world & (world - 1)) || rank < 0 || rank >= world || tokens < 1 ||
        tokens > 16384 || (chunk_tokens != 256 && chunk_tokens != 512 && chunk_tokens != 1024) || topk < 1 ||
        topk > 16 || experts < world || experts > 1024 || experts % world ||
        uint64_t(world) * tokens * topk >= (1ULL << 31)) {
        throw std::invalid_argument(
            "Notify requires EP2..512 powers of two, T1..16384, K1..16, "
            "E<=1024 divisible by EP, and EP*T*K < 2**31");
    }
    if (num_cores != 32U && num_cores != 64U) {
        throw std::invalid_argument("Notify requires 32 or 64 AIVs");
    }
    return ascend_deepep::MakeNotifyTiling(rank, world, tokens, topk, experts, chunk_tokens, 7168, num_cores);
}
using Shape = std::vector<int64_t>;
using Specs = std::map<std::string, std::pair<Shape, std::string>>;
Specs specs(const ascend_deepep::NotifyTiling& t, bool dispatch_plan)
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
                    {"dst", {{n, k}, "int64"}},
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
    if (dispatch_plan)
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

std::vector<std::string> jit_flags()
{
    return {
        "-std=c++17",
        "-DCATLASS_ARCH=3510",
        "-DUSE_SIMT",
        "-DL2_CACHE_HINT",
        "-DDEEPEP_LL_RDMA=1",
        "-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1",
        std::string("-DACLSHMEMI_HNS_1825_PI_ON_CHIP=") + std::to_string(DEEPEP_JIT_PI_ON_CHIP),
        "--cce-auto-infer-kernel-type=false",
    };
}

std::string notify_source(const char* file)
{
    return std::string("#include <kernels/notify/") + file + ">\n";
}

void launch_notify_full(void* stream, uint8_t* input, uint8_t* workspace, uint8_t* meta, uint8_t* hist,
                        uint8_t* bases, uint8_t* prefix, uint8_t* counts, uint8_t* gateways, uint8_t* local_dst,
                        uint8_t* mask, uint8_t* dst, uint8_t* forward, uint8_t* forward_counts, uint8_t* backward,
                        uint8_t* backward_counts, uint8_t* ranges, uint8_t* masks, uint8_t* rank_rows,
                        uint8_t* gather_rows, uint8_t* weight_return_meta, const ascend_deepep::NotifyTiling& t)
{
    const auto kernel = deepep::jit_compile("notify_full", notify_source("notify_full_kernel.cpp"), jit_flags());
    const int ub = ascend_deepep::kNotifyDynamicUbBytes +
                   (t.dispatch_plan.address ? ascend_deepep::kNotifyDispatchPlanUbBytes : 0);
    deepep::jit_launch(kernel, stream, static_cast<int>(t.num_cores), ub, input, workspace, meta, hist, bases, prefix, counts,
               gateways, local_dst, mask, dst, forward, forward_counts, backward, backward_counts, ranges, masks,
               rank_rows, gather_rows, weight_return_meta, t);
}

void launch_notify_profile(void* stream, uint8_t* input, uint8_t* workspace, uint8_t* meta, uint8_t* hist,
                           uint8_t* bases, uint8_t* prefix, uint8_t* counts, uint8_t* gateways, uint8_t* local_dst,
                           uint8_t* mask, uint8_t* dst, uint8_t* forward, uint8_t* forward_counts, uint8_t* backward,
                           uint8_t* backward_counts, uint8_t* ranges, uint8_t* masks, uint8_t* rank_rows,
                           uint8_t* gather_rows, uint8_t* weight_return_meta, uint8_t* profile, uint64_t sample,
                           const ascend_deepep::NotifyTiling& t)
{
    const auto kernel = deepep::jit_compile("notify_profile", notify_source("notify_profile_kernel.cpp"), jit_flags());
    const int ub = ascend_deepep::kNotifyDynamicUbBytes + ascend_deepep::notify_diagnostics::kUbBytes +
                   (t.dispatch_plan.address ? ascend_deepep::kNotifyDispatchPlanUbBytes : 0);
    deepep::jit_launch(kernel, stream, static_cast<int>(t.num_cores), ub, input, workspace, meta, hist, bases, prefix, counts,
               gateways, local_dst, mask, dst, forward, forward_counts, backward, backward_counts, ranges, masks,
               rank_rows, gather_rows, weight_return_meta, profile, sample, t);
}
}  // namespace

void launch_notify_barrier(void* stream, uint32_t num_cores)
{
    const auto kernel = deepep::jit_compile("notify_barrier", notify_source("notify_dispatch_barrier_kernel.cpp"), jit_flags());
    deepep::jit_launch(kernel, stream, static_cast<int>(num_cores), 0);
}

void bind_notify(py::module_& module){
    module.def(
        "notify_profile_layout",
        [](uint32_t num_cores) {
            if (num_cores != 32U && num_cores != 64U) {
                throw std::invalid_argument("Notify requires 32 or 64 AIVs");
            }
            return py::make_tuple(ascend_deepep::notify_diagnostics::kVersion, num_cores,
                                  ascend_deepep::notify_diagnostics::kWords);
        },
        py::arg("num_cores") = ascend_deepep::kNotifyCores);
    module.def(
        "notify_layout",
        [](int rank, int world, int tokens, int topk, int experts, int chunk_tokens, bool combine_only,
           uint32_t num_cores) {
            const auto t = tiling(rank, world, tokens, topk, experts, chunk_tokens, num_cores);
            py::dict result;
            result["workspace_bytes"] = t.workspace_bytes;
            result["tensors"] = py::cast(specs(t, !combine_only));
            return result;
        },
        py::arg("rank"), py::arg("world"), py::arg("tokens"), py::arg("topk"), py::arg("experts"),
        py::arg("chunk_tokens") = 256, py::arg("combine_only") = false,
        py::arg("num_cores") = ascend_deepep::kNotifyCores);
    module.def(
        "notify",
        [](Session& session, py::object indices, py::dict outputs, int tokens, int topk, int experts,
           uintptr_t stream_address, int chunk_tokens, uint32_t num_cores, py::object rank_dedup) {
            auto t = tiling(session.rank(), session.world(), tokens, topk, experts, chunk_tokens, num_cores);
            if (!rank_dedup.is_none()) {
                if (!py::isinstance<py::bool_>(rank_dedup)) throw py::type_error("rank_dedup must be bool or None");
                t.rank_dedup = rank_dedup.cast<bool>();
            }
            if (t.rank_dedup && !session.combine_only() && t.world != 8U && t.world != 16U)
                throw std::invalid_argument("Rank-deduplicated Dispatch supports EP8/16");
            auto* workspace = static_cast<uint8_t*>(session.workspace());
            if (t.workspace_bytes > session.capacity() / 2U || stream_address == 0) {
                throw std::invalid_argument("Notify workspace too small or invalid communication stream");
            }
            int32_t device = -1;
            if (aclrtGetDevice(&device) != 0) {
                throw std::runtime_error("Cannot obtain current ACL device");
            }
            auto* input = tensor_pointer(indices, {tokens, topk}, "int64", device);
            // Notify-only callers may use the public Combine-compatible layout
            // on a Dispatch-capable session. Only a supplied private plan
            // requests Dispatch construction; Combine-only always skips it.
            const bool dispatch_plan = !session.combine_only() && outputs.contains("_peer_plan");
            std::map<std::string, uint8_t*> ptr;
            for (const auto& entry : specs(t, dispatch_plan)) {
                ptr[entry.first] =
                    tensor_pointer(outputs[py::str(entry.first)], entry.second.first, entry.second.second, device);
            }
            t.expert_totals_output = reinterpret_cast<uint64_t>(ptr.at("expert_totals"));
            if (dispatch_plan) {
                const auto p =
                    ascend_deepep::MakeDispatchTiling(t.rank, t.world, t.tokens, t.topk, t.experts, 1U, true);
                t.dispatch_plan = {reinterpret_cast<uint64_t>(ptr.at("_peer_plan")), p.plan_phase_bytes,
                                   p.plan_rows_offset, p.plan_scratch_offset, p.plan_mailbox_offset};
            }
            // Mapping validation happens on every rank before device collectives (Python preflight).
            auto stream = reinterpret_cast<aclrtStream>(stream_address);
            const bool measure = outputs.contains("kernel_timing") && outputs["kernel_timing"].cast<bool>();
            uint8_t* profile = nullptr;
            uint64_t sample = 0;
            if (outputs.contains("profile")) {
                profile = tensor_pointer(outputs["profile"], {t.num_cores, ascend_deepep::notify_diagnostics::kWords},
                                         "int64", device);
                const auto id = outputs["profile_sample"].cast<int64_t>();
                if (id <= 0) throw std::invalid_argument("Notify profile_sample must be positive int64");
                sample = uint64_t(id);
            }
            py::gil_scoped_release release;
            DevicePhaseTiming timing(measure);
            timing.mark(0, stream);
            session.invalidate_dispatch_controls();
            if (profile) {
                launch_notify_profile(stream, input, workspace, ptr.at("meta"), ptr.at("hist"), ptr.at("bases"),
                                      ptr.at("prefix"), ptr.at("counts"), ptr.at("gateways"), ptr.at("local_dst"),
                                      ptr.at("server_mask"), ptr.at("dst"), ptr.at("forward_list"),
                                      ptr.at("forward_counts"), ptr.at("backward_list"), ptr.at("backward_counts"),
                                      ptr.at("chunk_ranges"), ptr.at("chunk_masks"), ptr.at("rank_rows"),
                                      ptr.at("gather_rows"), ptr.at("weight_return_meta"), profile, sample, t);
            } else {
                launch_notify_full(
                    stream, input, workspace, ptr.at("meta"), ptr.at("hist"), ptr.at("bases"), ptr.at("prefix"),
                    ptr.at("counts"), ptr.at("gateways"), ptr.at("local_dst"), ptr.at("server_mask"), ptr.at("dst"),
                    ptr.at("forward_list"), ptr.at("forward_counts"), ptr.at("backward_list"),
                    ptr.at("backward_counts"), ptr.at("chunk_ranges"), ptr.at("chunk_masks"), ptr.at("rank_rows"),
                    ptr.at("gather_rows"), ptr.at("weight_return_meta"), t);
            }
            timing.mark(1, stream);
            // Publication, Dispatch directory construction and totals export
            // finish in Notify itself. No payload/table-processing task follows.
            timing.mark(2, stream);
            timing.mark(3, stream);
            // Keep the diagnostic tuple ABI; in-kernel clocks report plan time.
            return std::make_pair(timing.finish(stream), measure ? std::vector<float>(4, 0.0F) : std::vector<float>{});
        },
        py::arg("session"), py::arg("indices"), py::arg("outputs"), py::arg("tokens"), py::arg("topk"),
        py::arg("experts"), py::arg("stream_address"), py::arg("chunk_tokens") = 256,
        py::arg("num_cores") = ascend_deepep::kNotifyCores, py::arg("rank_dedup") = py::none());
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
