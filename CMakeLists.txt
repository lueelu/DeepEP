# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

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
include(cmake/Device.cmake)
pybind11_add_module(_C csrc/bindings/module.cpp csrc/runtime/session.cpp csrc/ops/notify.cpp csrc/ops/dispatch.cpp csrc/ops/combine.cpp csrc/ops/low_latency.cpp)
target_include_directories(_C PRIVATE csrc "${ASCEND_HOME_PATH}/include" "${SHMEM_ROOT}/include")
target_link_libraries(_C PRIVATE ascendcl shmem deepep_device ${CMAKE_DL_LIBS})
target_compile_options(_C PRIVATE "$<$<NOT:$<CONFIG:Debug>>:-O2>" -Wall -Wextra -Werror -fstack-protector-strong)
target_compile_options(_C PRIVATE "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
    "-ffile-prefix-map=${SHMEM_ROOT}=SHMEM_SDK" "-ffile-prefix-map=${ASCEND_HOME_PATH}=CANN_SDK")
target_compile_definitions(_C PRIVATE "$<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>")
target_link_options(_C PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
# SDKs are external dependencies. Never embed checkout/toolkit absolute RPATHs.
set_target_properties(_C PROPERTIES INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS _C LIBRARY DESTINATION .)
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
  "capabilities": ["runtime-mte", "runtime-urma", "event", "elastic-notify-metadata-v6", "elastic-dispatch-expanded-v1", "elastic-combine-expanded-v3"],
  "communication_operations": ["ElasticBuffer.dispatch", "ElasticBuffer.combine"]
}
]=])
