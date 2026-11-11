// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "forward.hpp"
#include "shmem.h"
#include "kernels/mega_moe_common/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, module)
{
    using deepep::megamoe::Plan;
    module.attr("api_protocol") = 7;
    module.attr("small_batch_max_tokens") = MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS;
    module.def("buffer_size", &deepep::megamoe::buffer_size, py::arg("world"), py::arg("experts"), py::arg("tokens"),
               py::arg("topk"), py::arg("hidden"), py::arg("backend") = "pod");
    module.def("allocate_timer", &deepep::megamoe::allocate_timer, py::arg("example"), py::arg("backend") = "pod");
    py::class_<Plan>(module, "Plan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, const std::string&>(),
             py::arg("example"), py::arg("intermediate"), py::arg("topk"), py::arg("experts"), py::arg("world"),
             py::arg("max_tokens"), py::arg("is_a8w4"), py::arg("shared_experts"), py::arg("ranks_per_node"),
             py::arg("ffts"), py::arg("symmetric_address"), py::arg("symmetric_bytes"), py::arg("backend") = "pod")
        .def("forward", &Plan::forward, py::arg("x"), py::arg("x_scales"), py::arg("topk_ids"), py::arg("topk_weights"),
             py::arg("weight1"), py::arg("weight2"), py::arg("weight_scales1"), py::arg("weight_scales2"),
             py::arg("timer"), py::arg("shared_weight1"), py::arg("shared_weight2"), py::arg("shared_weight_scales1"),
             py::arg("shared_weight_scales2"), py::arg("out"), py::arg("generation") = 0)
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes);
    module.def("ffts_config", [] { return shmemx_get_ffts_config(); });
    module.attr("timer_numel") = AscendTimer::TOTAL_BUFFER_SIZE;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
#endif
}
