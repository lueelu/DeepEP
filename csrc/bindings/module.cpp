// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#include <pybind11/pybind11.h>
#include <memory>
#include <dlfcn.h>
#include "host/init/shmem_host_init.h"
#include "runtime/session.hpp"
#ifdef DEEPEP_BUILD_EP_GMM_FUSED
#include <pybind11/stl.h>
#endif

#include "ops/notify.hpp"
#include "ops/dispatch.hpp"
#include "ops/combine.hpp"
#include "ops/low_latency.hpp"

namespace py = pybind11;
PYBIND11_MODULE(_C, module)
{
    module.attr("runtime_protocol") = 1;
    module.def("shmem_library_path", []() {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&aclshmemx_init_attr), &info) == 0 || info.dli_fname == nullptr) {
            throw std::runtime_error("Cannot identify the loaded SHMEM library");
        }
        return std::string(info.dli_fname);
    });
    module.def("runtime_busy", &deepep::Session::busy, py::call_guard<py::gil_scoped_release>());
    py::class_<deepep::Session>(module, "Runtime")
        .def(py::init<int, int, uint64_t, const std::string&, unsigned, bool>(), py::arg("rank"), py::arg("size"),
             py::arg("bytes"), py::arg("endpoint"), py::arg("timeout"), py::arg("udma") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("barrier", &deepep::Session::barrier, py::call_guard<py::gil_scoped_release>())
        .def("close", &deepep::Session::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("closed", &deepep::Session::closed)
        .def_property_readonly("capacity", &deepep::Session::capacity)
        .def_property_readonly("peer_table", &deepep::Session::peer_table);
#ifdef DEEPEP_BUILD_MEGAMOE
    module.def(
        "create_megamoe_runtime",
        [](int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout) {
            return std::make_unique<deepep::Session>(rank, size, bytes, endpoint, timeout, false, true);
        },
        py::call_guard<py::gil_scoped_release>());
    module.def("megamoe_workspace_address", &deepep::Session::workspace_address);
#endif
#ifdef DEEPEP_BUILD_EP_GMM_FUSED
    module.def("create_udma_runtime", &deepep::Session::create_udma, py::arg("rank"), py::arg("size"),
               py::arg("heap_bytes"), py::arg("workspace_bytes"), py::arg("endpoint"), py::arg("uid_bytes"),
               py::call_guard<py::gil_scoped_release>());
    module.def("udma_unique_id_size", &deepep::Session::udma_unique_id_size);
    module.def("udma_unique_id", &deepep::Session::udma_unique_id, py::call_guard<py::gil_scoped_release>());
    module.def("udma_workspace_address", &deepep::Session::udma_workspace_address,
               py::call_guard<py::gil_scoped_release>());
#endif

    deepep::bind_notify(module);
    deepep::bind_dispatch(module);
    deepep::bind_combine(module);
    deepep::bind_low_latency(module);
}
