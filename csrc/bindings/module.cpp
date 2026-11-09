// Copyright (c) 2026, Lu Lu

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
    module.def(
        "create_mpi_runtime",
        [](int rank, int size, uint64_t bytes, unsigned timeout, bool use_1g_huge_pages) {
            return std::make_unique<deepep::Session>(rank, size, bytes, "", timeout, true, false, true, false,
                                                     use_1g_huge_pages);
        },
        py::arg("rank"), py::arg("size"), py::arg("bytes"), py::arg("timeout"), py::arg("use_1g_huge_pages") = false,
        py::call_guard<py::gil_scoped_release>());
    module.def(
        "create_combine_runtime",
        [](int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout, bool mpi,
           bool use_1g_huge_pages) {
            return std::make_unique<deepep::Session>(rank, size, bytes, endpoint, timeout, true, false, mpi, true,
                                                     use_1g_huge_pages);
        },
        py::arg("rank"), py::arg("size"), py::arg("bytes"), py::arg("endpoint"), py::arg("timeout"), py::arg("mpi"),
        py::arg("use_1g_huge_pages") = false, py::call_guard<py::gil_scoped_release>());
    py::class_<deepep::Session>(module, "Runtime")
        .def(py::init([](int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout, bool udma,
                         bool use_1g_huge_pages) {
                 return std::make_unique<deepep::Session>(rank, size, bytes, endpoint, timeout, udma, false, false,
                                                          false, use_1g_huge_pages);
             }),
             py::arg("rank"), py::arg("size"), py::arg("bytes"), py::arg("endpoint"), py::arg("timeout"),
             py::arg("udma") = false, py::arg("use_1g_huge_pages") = false, py::call_guard<py::gil_scoped_release>())
        .def("barrier", &deepep::Session::barrier, py::call_guard<py::gil_scoped_release>())
        .def("close", &deepep::Session::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("closed", &deepep::Session::closed)
        .def_property_readonly("capacity", &deepep::Session::capacity)
        .def_property_readonly("use_1g_huge_pages", &deepep::Session::use_1g_huge_pages)
        .def_property_readonly("workspace_allocation_bytes", &deepep::Session::workspace_allocation_bytes)
        .def_property_readonly("sparse_qp_enabled", &deepep::Session::sparse_qp_enabled)
        .def_property_readonly("megamoe_urma", &deepep::Session::megamoe_urma)
        .def_property_readonly("megamoe_urma_topology_error", &deepep::Session::megamoe_urma_topology_error)
        .def_property_readonly("peer_table", &deepep::Session::peer_table);
#ifdef DEEPEP_BUILD_MEGAMOE
    module.def(
        "create_megamoe_runtime",
        [](int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout) {
            return std::make_unique<deepep::Session>(rank, size, bytes, endpoint, timeout, false, true);
        },
        py::call_guard<py::gil_scoped_release>());
    module.def("megamoe_workspace_address", &deepep::Session::workspace_address);
    module.def(
        "create_megamoe_urma_runtime",
        [](int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout) {
            return std::make_unique<deepep::Session>(rank, size, bytes, endpoint, timeout, true, true, false, false,
                                                     false, true);
        },
        py::arg("rank"), py::arg("size"), py::arg("bytes"), py::arg("endpoint"), py::arg("timeout"),
        py::call_guard<py::gil_scoped_release>());
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
