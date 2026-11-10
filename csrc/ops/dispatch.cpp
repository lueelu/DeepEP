// Copyright (c) 2026, Lu Lu

#include "ops/device_phase_timing.hpp"

#include "ops/dispatch.hpp"
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>
#include <array>
#include "kernels/dispatch/tiling.hpp"
#include "kernels/dispatch/launch.hpp"
#include "kernels/notify/launch.hpp"
#include "runtime/session.hpp"

namespace py = pybind11;
namespace deepep {
namespace {
ascend_deepep::DispatchTiling layout(int rank, int world, int tokens, int topk, int experts, int rows, bool fp8,
                                     bool weights = false, int hidden = 7168)
{
    if (world < 2 || world > 512 || (world & (world - 1)) || rank < 0 || rank >= world || tokens < 1 ||
        tokens > 16384 || topk < 1 || topk > 16 || experts < world || experts > 1024 || experts % world || rows < 1 ||
        uint64_t(rows) > uint64_t(world) * tokens * topk || uint64_t(world) * tokens * topk >= (1ULL << 31) ||
        hidden <= 0 || hidden > 7168 || hidden % (fp8 ? 128 : 16)) {
        throw std::invalid_argument(
            "Invalid dispatch shape (EP2..512, H<=7168 aligned to BF16:16/FP8:128, T1..16384, K1..16, E<=1024)");
    }
    auto t = ascend_deepep::MakeDispatchTiling(rank, world, tokens, topk, experts, rows, fp8, weights, hidden);
    if (uint64_t(world) * t.address_stride >= (1ULL << 63) || t.workspace_bytes > (1ULL << 63)) {
        throw std::invalid_argument("Dispatch address encoding or workspace capacity overflow");
    }
    return t;
}

using Shape = std::vector<int64_t>;
uint8_t* pointer(py::handle tensor, const Shape& shape, const char* dtype, int device)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(tensor).cast<bool>()) {
        throw std::invalid_argument("Dispatch requires framework tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(tensor);
    if (obj.attr("device").attr("type").cast<std::string>() != "npu" ||
        obj.attr("device").attr("index").cast<int>() != device || !obj.attr("is_contiguous")().cast<bool>() ||
        obj.attr("shape").cast<Shape>() != shape || !obj.attr("dtype").is(torch.attr(dtype))) {
        throw std::invalid_argument("Dispatch tensor shape, dtype, device or contiguity mismatch");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}
}  // namespace

void bind_dispatch(py::module_& module)
{
    module.def(
        "dispatch_layout",
        [](int rank, int world, int tokens, int topk, int experts, int rows, bool fp8, bool weights, int hidden) {
            const auto t = layout(rank, world, tokens, topk, experts, rows, fp8, weights, hidden);
            py::dict result;
            result["workspace_bytes"] = t.workspace_bytes;
            result["address_stride"] = t.address_stride;
            result["sync"] = t.sync;
            result["book"] = t.book;
            result["input"] = t.input;
            result["output"] = t.output;
            result["weights"] = t.weights;
            result["scales"] = t.scales;
            result["inbox"] = t.inbox;
            result["inbox_stride"] = t.inbox_stride;
            result["peer_book"] = t.peer_book;
            result["done"] = t.done;
            result["weight_inbox"] = t.weight_inbox;
            result["weight_inbox_stride"] = t.weight_inbox_stride;
            result["weight_capacity"] = t.weight_capacity;
            result["aggregate_protocol"] = 1;
            result["aggregate_packet_bytes"] = 512;
            result["scale_batch_rows"] = fp8 ? ascend_deepep::scale_pipeline::Make(world, 224U).batch : 0U;
            return result;
        },
        py::arg("rank"), py::arg("world"), py::arg("tokens"), py::arg("topk"), py::arg("experts"), py::arg("rows"),
        py::arg("fp8"), py::arg("weights") = false, py::arg("hidden") = 7168);
    module.def("dispatch_plan_bytes", &ascend_deepep::DispatchPlanBytes);
    module.def("dispatch", [](Session& session, py::dict tensors, int tokens, int topk, int experts, int rows,
                              int local_rows, bool fp8, uintptr_t stream_address) {
        if (session.combine_only()) throw std::invalid_argument("Dispatch requires combine_only=False");
        const int hidden = tensors["input"].attr("size")(1).cast<int>();
        const auto t = layout(session.rank(), session.world(), tokens, topk, experts, rows, fp8,
                              !tensors["weights"].is_none(), hidden);
        auto* workspace = static_cast<uint8_t*>(session.workspace());
        if (!session.udma() || t.workspace_bytes > session.capacity() / 2U || stream_address == 0 || local_rows < 0 ||
            local_rows > rows) {
            throw std::invalid_argument(
                "Dispatch needs a MTE+URMA session, sufficient workspace and valid stream/rows");
        }
        int32_t device = -1;
        if (aclrtGetDevice(&device) != 0) {
            throw std::runtime_error("Cannot obtain current ACL device");
        }
        const char* dtype = fp8 ? "float8_e4m3fn" : "bfloat16";
        auto* input = pointer(tensors["input"], {tokens, hidden}, dtype, device);
        auto* output = pointer(tensors["output"], {local_rows, hidden}, dtype, device);
        auto* destinations = pointer(tensors["dst"], {tokens, topk}, "int64", device);
        auto* forward = pointer(tensors["forward_list"], {session.world(), int64_t(tokens) * topk, 6}, "int32", device);
        auto* counts = pointer(tensors["forward_counts"], {session.world()}, "int32", device);
        uint8_t *weights = nullptr, *out_weights = nullptr, *scales = nullptr, *out_scales = nullptr;
        if (!tensors["weights"].is_none()) {
            weights = pointer(tensors["weights"], {tokens, topk}, "float32", device);
            out_weights = pointer(tensors["output_weights"], {local_rows}, "float32", device);
        }
        if (fp8) {
            scales = pointer(tensors["scales"], {tokens, hidden / 128}, "float32", device);
            out_scales = pointer(tensors["output_scales"], {local_rows, hidden / 128}, "float32", device);
        }
        auto stream = reinterpret_cast<aclrtStream>(stream_address);
        auto config = t;
        if (tensors.contains("dedup_group_size")) {
            config.dedup_group_size = tensors["dedup_group_size"].cast<uint32_t>();
            if (config.dedup_group_size != 1U && config.dedup_group_size != 8U)
                throw std::invalid_argument("Dispatch dedup_group_size must be 1 or 8");
        }
        config.input_address = reinterpret_cast<uint64_t>(input);
        config.output_address = reinterpret_cast<uint64_t>(output);
        config.output_weights_address = reinterpret_cast<uint64_t>(out_weights);
        config.output_scales_address = reinterpret_cast<uint64_t>(out_scales);
        if (tensors.contains("rotation_wqes") && tensors["rotation_wqes"].cast<uint32_t>() != 4U)
            throw std::invalid_argument("Dispatch publishes four WQEs per doorbell");
        if (tensors.contains("detail_profile") && !tensors["detail_profile"].is_none()) {
            config.detail_profile =
                reinterpret_cast<uint64_t>(pointer(tensors["detail_profile"], {64, 32}, "int64", device));
            config.detail_mode = tensors["detail_mode"].cast<uint32_t>();
            if (config.detail_mode != 1U && config.detail_mode != 2U)
                throw std::invalid_argument("Invalid detail mode");
        }
        config.output_rows = local_rows;
        if (tensors.contains("weight_profile") && !tensors["weight_profile"].is_none()) {
            config.weight_profile =
                reinterpret_cast<uint64_t>(pointer(tensors["weight_profile"], {32, 16}, "int64", device));
        }
        if (tensors.contains("profile") && !tensors["profile"].is_none()) {
            config.profile = reinterpret_cast<uint64_t>(pointer(tensors["profile"], {64, 32}, "int64", device));
        }
        if (config.weight_profile && (!config.profile || !weights || !session.peer_table())) {
            throw std::invalid_argument("Weight diagnostics require a main profile and peer-table weights");
        }
        if (config.detail_profile && !config.profile)
            throw std::invalid_argument("Detail diagnostics require main profile");
        const bool measure = tensors.contains("kernel_timing") && tensors["kernel_timing"].cast<bool>();
        // Explicit benchmark alignment only. Timing/profiling does not change
        // the production synchronization contract by itself.
        const bool benchmark_barriers =
            tensors.contains("benchmark_barriers") && tensors["benchmark_barriers"].cast<bool>();
        const bool benchmark_entry_barrier = benchmark_barriers || (tensors.contains("benchmark_entry_barrier") &&
                                                                    tensors["benchmark_entry_barrier"].cast<bool>());
        if (tensors.contains("plan") && !tensors["plan"].is_none()) {
            if (!session.peer_table()) throw std::invalid_argument("peer-table session required");
            config.peer_table = reinterpret_cast<uint64_t>(
                pointer(tensors["plan"], {int64_t(ascend_deepep::DispatchPlanBytes(session.world(), tokens, topk))},
                        "uint8", device));
            config.generation = session.next_generation();
        }
        if (!config.peer_table) throw std::invalid_argument("Dispatch requires a prepared peer table");
        py::gil_scoped_release release;
        DevicePhaseTiming timing(measure);
        const Session::DispatchControlLayout controls{t.sync,
                                                      t.inbox,
                                                      t.inbox_stride,
                                                      t.peer_book,
                                                      t.done,
                                                      t.weight_inbox,
                                                      t.weight_inbox_stride,
                                                      t.world,
                                                      t.fp8,
                                                      uint64_t(weights != nullptr)};
        if (!session.dispatch_controls_match(controls)) {
            // Cold start / shape switch / Notify or LL reuse. No sender can run
            // until every receiver has completed this initialization. Initialize
            // all packet flags, not only headers: untouched heap bytes must never
            // masquerade as this Session's first generation. Cached calls retain
            // older generations and skip this untimed full-inbox clear.
            // This is lazy initialization, not per-call synchronization.
            if (aclrtSynchronizeStream(stream) != 0) throw std::runtime_error("Dispatch reuse sync failed");
            session.barrier();
            session.invalidate_dispatch_controls();
            auto clear = [&](uint64_t offset, uint64_t bytes) {
                if (offset > session.capacity() / 2U || bytes > session.capacity() / 2U - offset)
                    throw std::invalid_argument("Dispatch controls exceed workspace");
                if (aclrtMemsetAsync(workspace + offset, bytes, 0, bytes, stream) != 0)
                    throw std::runtime_error("Cannot initialize dispatch controls");
            };
            clear(t.sync, uint64_t(t.world) * 49152U + 65536U);
            clear(t.peer_book, uint64_t(t.world) * 512U);
            clear(t.done, uint64_t(t.world) * ascend_deepep::kDispatchCompletionLanes * 512U);
            for (uint32_t peer = 0; peer < t.world; ++peer) {
                if (t.fp8) clear(t.inbox + uint64_t(peer) * t.inbox_stride, t.inbox_stride);
                if (weights) clear(t.weight_inbox + uint64_t(peer) * t.weight_inbox_stride, t.weight_inbox_stride);
            }
            if (aclrtSynchronizeStream(stream) != 0) throw std::runtime_error("Dispatch control initialization failed");
            session.barrier();
            session.remember_dispatch_controls(controls);
        }
        timing.mark(0, stream);
        if (benchmark_entry_barrier) notify_dispatch_barrier_kernel_do(stream, 64U);
        timing.mark(1, stream);
        dispatch_kernel_do(stream, workspace, destinations, forward, counts, weights, scales,
                           reinterpret_cast<uint8_t*>(&config));
        timing.mark(2, stream);
        // The kernel joins incoming completion flags before its stream event.
        if (benchmark_barriers) notify_dispatch_barrier_kernel_do(stream, 64U);
        timing.mark(3, stream);
        return timing.finish(stream);
    });
}
}  // namespace deepep
