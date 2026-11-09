// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "forward.hpp"
#include "urma_forward.hpp"
#include "shmem.h"
#include "kernels/mega_moe/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, module)
{
    using deepep::megamoe::Plan;
    module.attr("api_protocol") = 4;
    module.attr("small_batch_max_tokens") = MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS;
    module.def("buffer_size", &deepep::megamoe::buffer_size);
    module.attr("urma_gmm_experiment") = "abc-fused-fixub-both-v5";
    module.attr("urma_routed_weight_layout") = "gate128-up128-v1";
    module.attr("urma_dimension_contract") = "consumer-submit-e2048-cv-policy-v1";
    module.attr("urma_api_protocol") = MegaMoeUrma::kProtocol;
    module.attr("urma_compute_schedule") = "unitflag-shared-fixub-both-v5";
    module.attr("urma_routing_protocol") = MegaMoeUrma::kDeviceRoutingProtocol;
    module.attr("urma_shared_contract") = "unitflag-shared-a8w8-early-reduce-v1";
    module.attr("urma_local_pipeline") = "pack6-reduce6-output2";
    module.attr("urma_tail_schedule") = "combine-ready-scan-reduce-world-v1";
    module.attr("urma_timer_numel") = MegaMoeUrma::kTimerElements;
    module.attr("urma_timer_schema") = MegaMoeUrma::kProtocol;
    module.attr("urma_trace_schema") = MegaMoeUrma::kTraceSchema;
    module.attr("urma_trace_max_capacity") = MegaMoeUrma::kTraceMaxCapacity;
    module.attr("urma_trace_num_engines") = MegaMoeUrma::kTimerCores;
    module.attr("urma_trace_record_elements") = MegaMoeUrma::kTraceRecordElements;
    module.attr("urma_trace_header_elements") = MegaMoeUrma::kTraceHeaderElements;
    module.def("urma_buffer_size", &deepep::megamoe::urma_buffer_size);
    py::class_<deepep::megamoe::UrmaPlan>(module, "UrmaPlan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, uint64_t, uint64_t,
                      int64_t, int64_t>())
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, uint64_t, uint64_t,
                      int64_t, int64_t, int64_t, int64_t>())
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, uint64_t, uint64_t,
                      int64_t, int64_t, int64_t, int64_t, int64_t>())
        .def("routing_tensors", &deepep::megamoe::UrmaPlan::routing_tensors)
        .def("route_stats", &deepep::megamoe::UrmaPlan::route_stats)
        .def_property_readonly("expert_counts", &deepep::megamoe::UrmaPlan::expert_counts)
        .def("forward", &deepep::megamoe::UrmaPlan::forward, py::arg("x"), py::arg("x_scales"), py::arg("topk_idx"),
             py::arg("topk_weights"), py::arg("weight1"), py::arg("weight2"), py::arg("weight_scales1"),
             py::arg("weight_scales2"), py::arg("timer"), py::arg("out"), py::arg("epoch"),
             py::arg("trace") = py::none(), py::arg("coarse_trace") = false, py::arg("shared_w1") = py::none(),
             py::arg("shared_w2") = py::none(), py::arg("shared_s1") = py::none(), py::arg("shared_s2") = py::none())
        .def_property_readonly("workspace_bytes", &deepep::megamoe::UrmaPlan::workspace_bytes)
        .def("debug_snapshot", &deepep::megamoe::UrmaPlan::debug_snapshot);
    py::class_<Plan>(module, "Plan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t>())
        .def("forward", &Plan::forward)
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes);
    module.def("ffts_config", [] { return shmemx_get_ffts_config(); });
    module.attr("timer_numel") = AscendTimer::TOTAL_BUFFER_SIZE;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
    module.attr("urma_timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
    module.attr("urma_timer_enabled") = false;
#endif
}
