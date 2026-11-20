// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "forward.hpp"
#include "shmem.h"
#include "kernel_launch.hpp"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernels/mega_moe/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"
#include "kernels/mega_moe/common/mega_moe_peermem.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ep32_C, module)
{
    using deepep::megamoe_ep32::Plan;
    module.attr("api_protocol") = 17;
    module.attr("combine_reduce_version") = deepep::megamoe_config::PROTOCOL;
    module.attr("combine_qps") = deepep::megamoe_config::COMBINE_QPS;
    module.attr("dispatch_qps") = deepep::megamoe_config::DISPATCH_QPS;
    module.attr("dispatch_workers") = deepep::megamoe_config::DISPATCH_WORKERS;
    module.attr("small_batch_max_tokens") = MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS;
    module.attr("hierarchical_max_tokens") = MegaMoeImpl::HIERARCHICAL_MAX_TOKENS;
    module.def("buffer_size", &deepep::megamoe_ep32::buffer_size, py::arg("world"), py::arg("experts"), py::arg("tokens"),
               py::arg("topk"), py::arg("hidden"), py::arg("ranks_per_node") = 0, py::arg("configs") = 1);
    py::class_<Plan>(module, "Plan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, std::array<int64_t, 4>, uint32_t, uint32_t>())
        .def("set_wave", &Plan::set_wave)
        .def_property_readonly("effective_wave", &Plan::effective_wave)
        .def("forward", &Plan::forward)
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes);
    module.def("ffts_config", [] { return shmemx_get_ffts_config(); });
    module.def("benchmark_sleep", [](uint64_t cycles) {
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        TORCH_CHECK(platform != nullptr && platform->GetCoreNumAic() > 0, "invalid Ascend platform");
        TORCH_CHECK(cycles <= 1000000000ULL, "benchmark sleep must be <= 1 billion system ticks");
        ep32_benchmark_sleep_do(platform->GetCoreNumAic(), c10_npu::getCurrentNPUStream().stream(), cycles);
    });
    module.def("benchmark_barrier", [](bool roce) {
        const auto stream = c10_npu::getCurrentNPUStream().stream();
        if (roce) {
#ifdef DEEPEP_MEGAMOE_ROCE
            ep32_benchmark_roce_barrier_do(stream, shmemx_get_ffts_config());
#else
            TORCH_CHECK(false, "SHMEM RoCE profiling requires a wheel built with an RDMA backend");
#endif
        } else {
            aclshmemx_barrier_all_on_stream(stream);
        }
    });
    module.attr("timer_numel") = AscendTimer::TOTAL_BUFFER_SIZE;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
#endif
}
