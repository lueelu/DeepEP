// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#include "bf16/training.hpp"

namespace py = pybind11;
PYBIND11_MODULE(_C, module)
{
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
#endif
    using deepep::megamoe_train::BF16Executor;
    namespace ops = deepep::megamoe_train::ops;
    py::class_<ops::ForwardResult>(module, "ForwardResult")
        .def_readonly("y", &ops::ForwardResult::y)
        .def_readonly("counts", &ops::ForwardResult::counts)
        .def_readonly("saved_state", &ops::ForwardResult::saved_state)
        .def_readonly("dispatch_output", &ops::ForwardResult::dispatch_output)
        .def_readonly("timer", &ops::ForwardResult::timer);
    py::class_<ops::BackwardResult>(module, "BackwardResult")
        .def_readonly("dx", &ops::BackwardResult::dx)
        .def_readonly("dtopk_weights", &ops::BackwardResult::dtopk_weights)
        .def_readonly("dw1", &ops::BackwardResult::dw1)
        .def_readonly("dw2", &ops::BackwardResult::dw2)
        .def_readonly("timer", &ops::BackwardResult::timer);
    py::class_<BF16Executor>(module, "BF16")
        .def(py::init<uintptr_t, int64_t, int, int, torch::Tensor>())
        .def("forward", &BF16Executor::forward)
        .def("forward_out", &BF16Executor::forward_out)
        .def("backward", &BF16Executor::backward)
        .def("forward_with_timer", &BF16Executor::forward_with_timer)
        .def("backward_with_timer", &BF16Executor::backward_with_timer)
        .def("close", &BF16Executor::close);
}
