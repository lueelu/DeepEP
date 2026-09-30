// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "forward.hpp"
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
    py::class_<Plan>(module, "Plan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t>())
        .def("forward", &Plan::forward)
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes);
    module.def("ffts_config", [] { return shmemx_get_ffts_config(); });
    module.attr("timer_numel") = AscendTimer::TOTAL_BUFFER_SIZE;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
#endif
}
