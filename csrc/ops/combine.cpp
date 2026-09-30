// Copyright (c) 2026, Lu Lu
// Modified by lishaoxun 2026

#include "ops/combine.hpp"
#include <pybind11/stl.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "kernels/combine/tiling.hpp"
#include "kernels/combine/launch.hpp"
#include "runtime/session.hpp"

namespace py = pybind11;
namespace deepep {
namespace {
ascend_deepep::PutCombineTiling layout(int rank, int world, int tokens, int topk, int rows, int gathered, int chunk)
{
    if (world < 2 || world > 256 || (world & (world - 1)) || rank < 0 || rank >= world || tokens < 1 ||
        tokens > 10240 || topk < 1 || topk > 16 || chunk < 1 || chunk > 10240 || rows < 1 || gathered < 1 ||
        uint64_t(rows) > uint64_t(world) * tokens * topk || uint64_t(gathered) > uint64_t(world) * tokens * topk ||
        uint64_t(world) * world * tokens * topk >= (1ULL << 31)) {
        throw std::invalid_argument("Invalid expanded Combine shape");
    }
    const auto t = ascend_deepep::MakeCombineTiling(rank, world, tokens, topk, rows, gathered, chunk);
    const uint64_t range_bytes = uint64_t(world) * t.chunks_per_group * 8U;
    const uint64_t mask_bytes = uint64_t(t.total_chunk_num) * 4U;
    if ((range_bytes + 31U) / 32U * 32U + (mask_bytes + 31U) / 32U * 32U >
        ascend_deepep::combine_config::kGatewayControlUbBytes) {
        throw std::invalid_argument("Combine chunk ranges/masks exceed the resident 44 KiB UB capacity");
    }
    if (ascend_deepep::CombineWorkspaceBytes(t) > (16ULL << 30)) {
        throw std::invalid_argument("Combine exceeds its half of the 32 GiB session limit");
    }
    return t;
}
using Shape = std::vector<int64_t>;
uint8_t* pointer(py::handle tensor, const Shape& shape, const char* dtype, int device)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(tensor).cast<bool>()) {
        throw std::invalid_argument("Combine requires framework tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(tensor);
    if (obj.attr("device").attr("type").cast<std::string>() != "npu" ||
        obj.attr("device").attr("index").cast<int>() != device || !obj.attr("is_contiguous")().cast<bool>() ||
        obj.attr("requires_grad").cast<bool>() || obj.attr("shape").cast<Shape>() != shape ||
        !obj.attr("dtype").is(torch.attr(dtype))) {
        throw std::invalid_argument("Combine tensor shape, dtype, device or contiguity mismatch");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}
void copy(void* dst, const void* src, uint64_t bytes, aclrtStream stream)
{
    if (bytes && aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) != 0) {
        throw std::runtime_error("Combine workspace copy failed; restart the distributed job");
    }
}
}  // namespace
void bind_combine(py::module_& module)
{
    module.def("combine_layout", [](int rank, int world, int tokens, int topk, int rows, int gathered, int chunk) {
        const auto t = layout(rank, world, tokens, topk, rows, gathered, chunk);
        py::dict result;
        result["control_slot_num"] = t.control_slot_num;
        result["expert_input"] = t.expert_input_offset_bytes;
        result["gather_input"] = t.gather_input_offset_bytes;
        result["server_partial"] = t.server_partial_offset_bytes;
        result["returned_partial"] = t.returned_partial_offset_bytes;
        result["output"] = t.output_offset_bytes;
        result["weight_recv"] = t.weight_recv_offset_bytes;
        result["weight_local_done"] = t.weight_local_done_offset_bytes;
        result["weight_done"] = t.weight_done_offset_bytes;
        result["workspace_bytes"] = ascend_deepep::CombineWorkspaceBytes(t);
        return result;
    });
    module.def("combine", [](Session& session, py::dict tensors, int tokens, int topk, int rows, int gathered,
                             int local_rows, int chunk, uintptr_t stream_address) {
        auto t = layout(session.rank(), session.world(), tokens, topk, rows, gathered, chunk);
        const uint64_t half = session.capacity() / 2U;
        auto* workspace = static_cast<uint8_t*>(session.workspace()) + half;
        if (!session.udma() || ascend_deepep::CombineWorkspaceBytes(t) > half || !stream_address || local_rows < 0 ||
            local_rows > rows) {
            throw std::invalid_argument("Combine needs MTE+URMA, sufficient workspace and a valid stream/rows");
        }
        int32_t device = -1;
        if (aclrtGetDevice(&device) != 0) {
            throw std::runtime_error("Cannot obtain current ACL device");
        }
        auto* input = pointer(tensors["input"], {local_rows, 7168}, "bfloat16", device);
        auto* output = pointer(tensors["output"], {tokens, 7168}, "bfloat16", device);
        auto* forward = pointer(tensors["forward_list"], {session.world(), int64_t(tokens) * topk, 6}, "int32", device);
        auto* counts = pointer(tensors["forward_counts"], {session.world()}, "int32", device);
        auto* backward =
            pointer(tensors["backward_list"],
                    {std::min(session.world(), 8), int64_t(session.world()) * tokens * topk, 3}, "int64", device);
        auto* backward_counts = pointer(tensors["backward_counts"], {std::min(session.world(), 8)}, "int32", device);
        auto* mask = pointer(tensors["server_mask"], {tokens}, "int64", device);
        auto* ranges = pointer(tensors["chunk_ranges"], {session.world(), t.chunks_per_group, 2}, "int32", device);
        auto* masks =
            pointer(tensors["chunk_masks"], {session.world() / t.group_size, t.chunks_per_group}, "int32", device);
        uint8_t *weights = nullptr, *weight_meta = nullptr, *weight_output = nullptr;
        t.with_weights = tensors.contains("weights");
        t.weight_rows = uint32_t(local_rows);
        if (t.with_weights) {
            weights = pointer(tensors["weights"], {local_rows}, "float32", device);
            weight_meta = pointer(tensors["weight_return_meta"], {local_rows}, "int32", device);
            weight_output = pointer(tensors["weight_output"], {tokens, topk}, "float32", device);
        }
        auto stream = reinterpret_cast<aclrtStream>(stream_address);
        py::gil_scoped_release release;
        // Before overwriting even input storage, all peers have finished the
        // previous operator and its output copy on this ordered stream.
        notify_dispatch_barrier_kernel_do(stream, 64U);
        copy(workspace + t.expert_input_offset_bytes, input, uint64_t(local_rows) * 14336U, stream);
        combine_prepare_kernel_do(stream, workspace, reinterpret_cast<uint8_t*>(&t));
        combine_kernel_do(stream, workspace, forward, counts, backward, backward_counts, mask, ranges, masks, weights,
                          weight_meta, reinterpret_cast<uint8_t*>(&t));
        notify_dispatch_barrier_kernel_do(stream, 64U);
        copy(output, workspace + t.output_offset_bytes, uint64_t(tokens) * 14336U, stream);
        if (t.with_weights) {
            copy(weight_output, workspace + t.weight_recv_offset_bytes, uint64_t(tokens) * topk * sizeof(float),
                 stream);
        }
    });
}
}  // namespace deepep
