// Copyright (c) 2026, Lu Lu

#include "ops/combine.hpp"
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <mutex>
#include <sys/types.h>
#include <stdexcept>
#include <vector>
#include "kernels/combine/tiling.hpp"
#include "kernels/combine/launch.hpp"
#include "kernels/combine/profile.hpp"
#include "kernels/combine/chunk_profile.hpp"
#include "kernels/combine/peer_profile.hpp"
#include "runtime/session.hpp"

namespace py = pybind11;
namespace {
constexpr size_t gm_page_bytes = 1ULL << 30;
struct HugeGmBlock {
    void* address = nullptr;
    aclrtDrvMemHandle handle = nullptr;
    size_t requested = 0;
    size_t allocated = 0;
    int device = -1;
    bool mapped = false;

    void release()
    {
        auto check = [](aclError code) {
            if (code != ACL_SUCCESS) throw std::runtime_error("Huge GM release failed: " + std::to_string(code));
        };
        if (mapped) {
            check(aclrtUnmapMem(address));
            mapped = false;
        }
        if (address) {
            check(aclrtReleaseMemAddress(address));
            address = nullptr;
        }
        if (handle) {
            check(aclrtFreePhysical(handle));
            handle = nullptr;
        }
    }
};
std::mutex gm_mutex;
std::map<uintptr_t, HugeGmBlock> gm_blocks;
}  // namespace

extern "C" __attribute__((visibility("default"))) void* deepep_gm_huge_alloc(ssize_t bytes, int device, aclrtStream)
{
    HugeGmBlock block;
    try {
        auto check = [](aclError code, const char* operation) {
            if (code != ACL_SUCCESS) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
        };
        if (bytes <= 0 || static_cast<uint64_t>(bytes) > std::numeric_limits<int64_t>::max() - (gm_page_bytes - 1))
            throw std::invalid_argument("Invalid huge GM allocation size");
        int32_t current = -1;
        check(aclrtGetDevice(&current), "aclrtGetDevice");
        if (current != device) throw std::invalid_argument("Huge GM allocator device mismatch");
        aclrtPhysicalMemProp property{};
        property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        property.memAttr = ACL_HBM_MEM_HUGE1G;
        property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        property.location.id = device;
        size_t granularity = 0;
        check(aclrtMemGetAllocationGranularity(&property, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity),
              "aclrtMemGetAllocationGranularity");
        if (granularity != gm_page_bytes) throw std::runtime_error("Huge GM requires 1 GiB physical pages");
        block.requested = static_cast<size_t>(bytes);
        block.allocated = (block.requested + gm_page_bytes - 1) / gm_page_bytes * gm_page_bytes;
        block.device = device;
        check(aclrtMallocPhysical(&block.handle, block.allocated, &property, 0), "aclrtMallocPhysical");
        check(aclrtReserveMemAddress(&block.address, block.allocated, 0, nullptr, 0), "aclrtReserveMemAddress");
        check(aclrtMapMem(block.address, block.allocated, 0, block.handle, 0), "aclrtMapMem");
        block.mapped = true;
        aclrtMemAccessDesc access{};
        access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        access.location = property.location;
        check(aclrtMemSetAccess(block.address, block.allocated, &access, 1), "aclrtMemSetAccess");
        std::lock_guard<std::mutex> lock(gm_mutex);
        gm_blocks.emplace(reinterpret_cast<uintptr_t>(block.address), block);
        return block.address;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Huge GM allocation failed: %s\n", error.what());
        try {
            block.release();
        } catch (const std::exception& cleanup) {
            std::fprintf(stderr, "%s\n", cleanup.what());
        }
        return nullptr;
    }
}

extern "C" __attribute__((visibility("default"))) void deepep_gm_huge_free(void* address, size_t, aclrtStream)
{
    try {
        std::lock_guard<std::mutex> lock(gm_mutex);
        auto found = gm_blocks.find(reinterpret_cast<uintptr_t>(address));
        if (found == gm_blocks.end()) throw std::invalid_argument("Unknown huge GM allocation");
        found->second.release();
        gm_blocks.erase(found);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Huge GM free failed: %s\n", error.what());
    }
}

namespace deepep {
namespace {
ascend_deepep::PutCombineTiling layout(int rank, int world, int tokens, int topk, int rows, int gathered, int chunk,
                                       int hidden)
{
    if (world < 2 || world > 512 || (world & (world - 1)) || rank < 0 || rank >= world || tokens < 1 ||
        tokens > 16384 || topk < 1 || topk > 16 || chunk < 1 || chunk > 16384 || rows < 1 || gathered < 1 ||
        uint64_t(rows) > uint64_t(world) * tokens * topk || uint64_t(gathered) > uint64_t(world) * tokens * topk ||
        uint64_t(world) * tokens * topk >= (1ULL << 31) || hidden <= 0 || hidden > 7168 || hidden % 16) {
        throw std::invalid_argument("Invalid expanded Combine shape");
    }
    const auto t = ascend_deepep::MakeCombineTiling(rank, world, tokens, topk, rows, gathered, chunk, hidden);
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
    module.def("gm_huge_allocator_version", [] { return 1; });
    module.def("gm_huge_allocation_info", [](uintptr_t address, size_t bytes) {
        std::lock_guard<std::mutex> lock(gm_mutex);
        auto found = gm_blocks.upper_bound(address);
        if (found == gm_blocks.begin()) throw std::invalid_argument("Tensor is not allocated from huge GM");
        --found;
        const auto& block = found->second;
        const auto offset = address - found->first;
        if (!block.mapped || offset > block.requested || bytes > block.requested - offset)
            throw std::invalid_argument("Tensor exceeds its huge GM allocation");
        py::dict result;
        result["page_bytes"] = gm_page_bytes;
        result["allocation_bytes"] = block.allocated;
        result["requested_bytes"] = block.requested;
        result["tensor_bytes"] = bytes;
        result["offset_bytes"] = offset;
        result["device"] = block.device;
        return result;
    });
    module.def("combine_input_staging_version", [] { return 1; });
    module.def("combine_reuse_version", [] { return 1; });
    module.def("combine_ep8_version", [] { return 2; });
    // Compatibility with older Python clients; both names select the EP8 pipeline.
    module.def("combine_ep8_mte_version", [] { return 2; });
    module.def("combine_profile_version", [] { return ascend_deepep::combine_diagnostics::kVersion; });
    module.def("combine_chunk_profile_version", [] { return ascend_deepep::combine_chunks::kVersion; });
    module.def("combine_peer_profile_version", [] { return ascend_deepep::combine_peers::kVersion; });
    module.def(
        "combine_layout",
        [](int rank, int world, int tokens, int topk, int rows, int gathered, int chunk, int hidden) {
            const auto t = layout(rank, world, tokens, topk, rows, gathered, chunk, hidden);
            py::dict result;
            result["control_slot_num"] = t.control_slot_num;
            result["gather_input"] = t.gather_input_offset_bytes;
            result["server_partial"] = t.server_partial_offset_bytes;
            result["returned_partial"] = t.returned_partial_offset_bytes;
            result["output"] = t.output_offset_bytes;
            result["weight_recv"] = t.weight_recv_offset_bytes;
            result["weight_local_done"] = t.weight_local_done_offset_bytes;
            result["weight_done"] = t.weight_done_offset_bytes;
            result["workspace_bytes"] = ascend_deepep::CombineWorkspaceBytes(t);
            return result;
        },
        py::arg("rank"), py::arg("world"), py::arg("tokens"), py::arg("topk"), py::arg("rows"), py::arg("gathered"),
        py::arg("chunk"), py::arg("hidden") = 7168);
    module.def("combine", [](Session& session, py::dict tensors, int tokens, int topk, int rows, int gathered,
                             int local_rows, int chunk, uintptr_t stream_address) {
        const int hidden = tensors["input"].attr("size")(1).cast<int>();
        auto t = layout(session.rank(), session.world(), tokens, topk, rows, gathered, chunk, hidden);
        const uint32_t reuse = tensors.contains("reuse_tokens") ? tensors["reuse_tokens"].cast<uint32_t>() : 0U;
        if (reuse) {
            if (reuse > uint32_t(tokens)) throw std::invalid_argument("Invalid Combine reuse window");
            t = layout(session.rank(), session.world(), tokens, topk, reuse, reuse * topk, chunk, hidden);
            t.expert_row_capacity = rows;
        }
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
        auto* input = pointer(tensors["input"], {local_rows, hidden}, "bfloat16", device);
        auto* output = pointer(tensors["output"], {tokens, hidden}, "bfloat16", device);
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
        uint8_t *profile = nullptr, *chunk_profile = nullptr, *peer_profile = nullptr;
        uint64_t sample = 0;
        if (tensors.contains("profile") && !tensors["profile"].is_none()) {
            profile = pointer(tensors["profile"], {64, ascend_deepep::combine_diagnostics::kWords}, "int64", device);
            chunk_profile = pointer(
                tensors["chunk_profile"],
                {64,
                 int64_t(ascend_deepep::combine_chunks::RecordsPerCore(t.rank_num / t.group_size, t.chunks_per_group)),
                 ascend_deepep::combine_chunks::kWords},
                "int64", device);
            peer_profile = pointer(tensors["peer_profile"],
                                   {2, session.world(), t.chunks_per_group, 2, ascend_deepep::combine_peers::kWords},
                                   "int64", device);
            sample = tensors["profile_sample"].cast<uint64_t>();
            if (!sample || sample >= (1ULL << 63)) throw std::invalid_argument("Invalid Combine profile sample");
        }
        const bool ep8_pipeline = tensors.contains("ep8_rank_pipeline")
                                      ? tensors["ep8_rank_pipeline"].cast<bool>()
                                      : tensors.contains("ep8_rank_mte") && tensors["ep8_rank_mte"].cast<bool>();
        const bool stage_input = tensors.contains("stage_input") && tensors["stage_input"].cast<bool>();
        uint8_t *reuse_backward = nullptr, *reuse_offsets = nullptr;
        if (reuse) {
            if (stage_input || ep8_pipeline || profile) {
                throw std::invalid_argument("Combine reuse cannot use staging, rank-return, or legacy diagnostics");
            }
            const int64_t plan_rows = tensors["reuse_backward"].attr("size")(0).cast<int64_t>();
            if (plan_rows < 0 || plan_rows > int64_t(std::min(session.world(), 8)) * t.backward_row_capacity) {
                throw std::invalid_argument("Invalid Combine reuse plan size");
            }
            reuse_backward = pointer(tensors["reuse_backward"], {plan_rows, 3}, "int64", device);
            reuse_offsets = pointer(tensors["reuse_offsets"], {std::min(session.world(), 8) + 1}, "int64", device);
        }
        const uint64_t staging_offset = (ascend_deepep::CombineWorkspaceBytes(t) + 511U) / 512U * 512U;
        const uint64_t staging_bytes = uint64_t(rows) * hidden * sizeof(uint16_t);
        if (stage_input && (staging_offset > half || staging_bytes > half - staging_offset)) {
            throw std::invalid_argument("Combine input staging exceeds its reserved SHMEM half");
        }
        if (ep8_pipeline && t.rank_num != 8U) {
            throw std::invalid_argument("EP8 return pipeline requires EP8");
        }
        auto stream = reinterpret_cast<aclrtStream>(stream_address);
        py::gil_scoped_release release;
        // All peers must finish the previous operator before reusing workspace.
        notify_dispatch_barrier_kernel_do(stream, 64U);
        if (reuse) {
            if (t.with_weights &&
                aclrtMemsetAsync(workspace + t.weight_recv_offset_bytes, uint64_t(tokens) * topk * sizeof(float), 0,
                                 uint64_t(tokens) * topk * sizeof(float), stream) != 0) {
                throw std::runtime_error("Cannot reset Combine weight return buffer");
            }
            combine_reuse_kernel_do(stream, workspace, input, forward, counts, reuse_backward, reuse_offsets, mask,
                                    weights, weight_meta, reinterpret_cast<uint8_t*>(&t), reuse);
            notify_dispatch_barrier_kernel_do(stream, 64U);
            copy(output, workspace + t.output_offset_bytes, uint64_t(tokens) * hidden * 2U, stream);
            if (t.with_weights) {
                copy(weight_output, workspace + t.weight_recv_offset_bytes, uint64_t(tokens) * topk * sizeof(float),
                     stream);
            }
            return;
        }
        if (stage_input) {
            auto* staged = workspace + staging_offset;
            copy(staged, input, uint64_t(local_rows) * hidden * sizeof(uint16_t), stream);
            input = staged;
        }
        combine_prepare_kernel_do(stream, workspace, reinterpret_cast<uint8_t*>(&t));
        if (ep8_pipeline && profile) {
            combine_ep8_profile_kernel_do(stream, workspace, input, forward, counts, backward, backward_counts, mask,
                                          ranges, masks, weights, weight_meta, profile, sample, chunk_profile,
                                          peer_profile, reinterpret_cast<uint8_t*>(&t));
        } else if (ep8_pipeline) {
            combine_ep8_kernel_do(stream, workspace, input, forward, counts, backward, backward_counts, mask, ranges,
                                  masks, weights, weight_meta, reinterpret_cast<uint8_t*>(&t));
        } else if (profile) {
            combine_profile_kernel_do(stream, workspace, input, forward, counts, backward, backward_counts, mask,
                                      ranges, masks, weights, weight_meta, profile, sample, chunk_profile, peer_profile,
                                      reinterpret_cast<uint8_t*>(&t));
        } else {
            combine_kernel_do(stream, workspace, input, forward, counts, backward, backward_counts, mask, ranges, masks,
                              weights, weight_meta, reinterpret_cast<uint8_t*>(&t));
        }
        notify_dispatch_barrier_kernel_do(stream, 64U);
        copy(output, workspace + t.output_offset_bytes, uint64_t(tokens) * hidden * 2U, stream);
        if (t.with_weights) {
            copy(weight_output, workspace + t.weight_recv_offset_bytes, uint64_t(tokens) * topk * sizeof(float),
                 stream);
        }
    });
}
}  // namespace deepep
