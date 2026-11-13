# Copyright (c) 2026, Lu Lu

cmake_minimum_required(VERSION 3.20)
project(deepep_runtime LANGUAGES CXX)
if(CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR "Use a single-configuration generator such as Ninja or Unix Makefiles")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()
file(WRITE "${CMAKE_BINARY_DIR}/compiler-info.json"
    "{\"id\":\"${CMAKE_CXX_COMPILER_ID}\",\"version\":\"${CMAKE_CXX_COMPILER_VERSION}\"}\n")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(DEEPEP_BUILD_EP_GMM_FUSED "Build Ascend950 communication/GMM experiment" OFF)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
include(cmake/Dependencies.cmake)
option(DEEPEP_BUILD_MEGAMOE "Build the optional MegaMoE Beta" OFF)
option(DEEPEP_MEGAMOE_TIMER "Compile MegaMoE device tracing" OFF)
if(NOT TORCH_NPU_PATH)
    message(FATAL_ERROR "DeepEP requires -DTORCH_NPU_PATH (installed torch_npu directory)")
endif()
find_package(Torch REQUIRED)
include(cmake/DeepJIT.cmake)

# One runtime shared by every native extension. DeepJIT is mandatory.
add_library(deepep_jit SHARED csrc/runtime/jit_runtime.cpp)
set_target_properties(deepep_jit PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_include_directories(deepep_jit PRIVATE
    csrc
    "${ASCEND_HOME_PATH}/include"
    "${TORCH_NPU_PATH}/include"
    "${TORCH_NPU_PATH}/include/third_party/acl/inc"
    ${TORCH_INCLUDE_DIRS})
target_link_directories(deepep_jit PRIVATE "${ASCEND_HOME_PATH}/lib64" "${TORCH_NPU_PATH}/lib" ${TORCH_LIBRARY_DIRS})
target_link_libraries(deepep_jit PRIVATE ascendcl torch_npu ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})
target_compile_definitions(deepep_jit PRIVATE
    DEEPEP_JIT_PI_ON_CHIP=${DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP})
deepep_apply_deepjit(deepep_jit)
set_target_properties(deepep_jit PROPERTIES INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS deepep_jit LIBRARY DESTINATION .)

# DeepJIT is a mandatory Host dependency, like DeepEP-Ascend. The single _C
# extension owns both the runtime/ops bindings and the JIT compile/load/launch
# bindings; there is no separate optional extension.
pybind11_add_module(_C
    csrc/bindings/module.cpp
    csrc/runtime/session.cpp
    csrc/runtime/ll_jit.cpp
    csrc/ops/notify.cpp
    csrc/ops/ll_abi.cpp
    csrc/ops/dispatch.cpp
    csrc/ops/dispatch_jit.cpp
    csrc/ops/combine.cpp
    csrc/ops/combine_jit.cpp
    csrc/ops/low_latency.cpp
    csrc/ubx/v2.cpp)
set_target_properties(_C PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_include_directories(_C PRIVATE
    csrc
    "${ASCEND_HOME_PATH}/include"
    "${SHMEM_ROOT}/include"
    "${TORCH_NPU_PATH}/include"
    "${TORCH_NPU_PATH}/include/third_party/acl/inc"
    ${TORCH_INCLUDE_DIRS})
find_path(DEEPEP_PLATFORM_INCLUDE tiling/platform/platform_ascendc.h
    HINTS "${ASCEND_HOME_PATH}/asc/include" "${ASCEND_HOME_PATH}/include/op_common"
          "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include" NO_DEFAULT_PATH REQUIRED)
target_include_directories(_C PRIVATE "${DEEPEP_PLATFORM_INCLUDE}")
target_link_directories(_C PRIVATE "${ASCEND_HOME_PATH}/lib64" "${TORCH_NPU_PATH}/lib" ${TORCH_LIBRARY_DIRS})
target_link_libraries(_C PRIVATE platform tiling_api unified_dlog)
target_compile_definitions(_C PRIVATE DEEPEP_LL_RDMA=1)
target_link_libraries(_C PRIVATE ascendcl shmem deepep_jit torch_npu ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})
target_compile_options(_C PRIVATE "$<$<NOT:$<CONFIG:Debug>>:-O2>" -Wall -Wextra -Werror -fstack-protector-strong)
target_compile_options(_C PRIVATE "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
    "-ffile-prefix-map=${SHMEM_ROOT}=SHMEM_SDK" "-ffile-prefix-map=${ASCEND_HOME_PATH}=CANN_SDK")
target_compile_definitions(_C PRIVATE "$<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>")
# Match the Torch/torch_npu C++ ABI selected by find_package(Torch).
separate_arguments(_deepep_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(_C PRIVATE ${_deepep_torch_cxx_flags})
# The JIT reproduces the device compile line from the AOT build, so the selected
# roots and doorbell mode travel with the main module.
target_compile_definitions(_C PRIVATE
    DEEPEP_JIT_PI_ON_CHIP=${DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP})
if(DEEPEP_LL_REFERENCE_PROFILE)
    target_compile_definitions(_C PRIVATE DEEPEP_LL_REFERENCE_PROFILE=1 DEEPEP_JIT_REFERENCE_PROFILE=1)
endif()
if(DEEPEP_LL_COUNT_HISTOGRAM)
    target_compile_definitions(_C PRIVATE DEEPEP_LL_COUNT_HISTOGRAM=1 DEEPEP_JIT_COUNT_HISTOGRAM=1)
endif()
deepep_apply_deepjit(_C)
target_link_options(_C PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
# SDKs are external dependencies. Never embed checkout/toolkit absolute RPATHs.
set_target_properties(_C PROPERTIES INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS _C LIBRARY DESTINATION .)
include(cmake/Direct.cmake)
if(DEEPEP_BUILD_MEGAMOE)
    target_compile_definitions(_C PRIVATE DEEPEP_BUILD_MEGAMOE=1)
    add_subdirectory(experiments/megamoe)
elseif(DEEPEP_MEGAMOE_TIMER)
    message(FATAL_ERROR "DEEPEP_MEGAMOE_TIMER requires DEEPEP_BUILD_MEGAMOE")
endif()
if(DEEPEP_BUILD_EP_GMM_FUSED)
    target_compile_definitions(_C PRIVATE DEEPEP_BUILD_EP_GMM_FUSED=1)
    add_subdirectory(experiments/ep_gmm_fused)
endif()


# Capability declarations live alongside the targets actually built above.
# Packaging consumes this file instead of maintaining another operator list.
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/capabilities.json" CONTENT [=[
{
  "capabilities": ["runtime-mte", "runtime-urma", "runtime-mpi-bootstrap", "runtime-shmem-huge1g-v1", "event", "elastic-notify-metadata-v7", "elastic-notify-profile-v1", "elastic-dispatch-expanded-v1", "elastic-combine-expanded-v3", "elastic-combine-rank-return-v1", "elastic-combine-return-slots-v2", "elastic-combine-chunk-tuning-v1", "elastic-combine-profile-v4", "elastic-combine-peers-v1", "elastic-combine-chunks-v2"],
  "communication_operations": ["ElasticBuffer.dispatch", "ElasticBuffer.combine"],
  "machine_backends": {
    "mesh_clos": {"module": "deep_ep._C", "device_library": null},
    "ubx": {"module": "deep_ep._C", "device_library": null,
      "interfaces": ["ElasticBuffer"], "transport": "mte+roce", "scheme": "route-plan"},
    "mesh_pcie": {
      "module": "deep_ep._clos_C", "device_library": null,
      "interfaces": ["Buffer", "ElasticBuffer"], "transport": "mte", "topk_index_bits": 64
    },
    "global_clos": {
      "module": "deep_ep._clos_C", "device_library": null,
      "interfaces": ["Buffer", "ElasticBuffer"],
      "transport": "udma", "topk_index_bits": 64
    }
  }
}
]=])
