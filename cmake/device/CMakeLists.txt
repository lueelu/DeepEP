# Copyright (c) 2026, Lu Lu
# Modified by candy_cloud_fly 2026

cmake_minimum_required(VERSION 3.20)
project(deepep_device LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include("${CMAKE_CURRENT_LIST_DIR}/../Dependencies.cmake")
set(_sdk_include "${ASCEND_HOME_PATH}/include")
set(_arch_include "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
set(_tikcpp "${ASCEND_HOME_PATH}/compiler/tikcpp")
# Some toolkit layouts expose a legacy rt.h compatibility header separately.
# Do not require that header to configure the CANN 9.2 -xasc target: the
# compiler/toolkit headers determine which runtime API is actually needed.
# Re-evaluate on configure rather than retaining a path from another SDK.
unset(DEVICE_RT_INCLUDE CACHE)
find_path(DEVICE_RT_INCLUDE rt.h HINTS
    "${_sdk_include}/runtime" "${_sdk_include}/experiment/runtime" "${_sdk_include}"
    "${_sdk_include}/acl/runtime" "${ASCEND_HOME_PATH}/runtime/include"
    "${_arch_include}/runtime" "${_arch_include}/experiment/runtime"
    "${_arch_include}/acl/runtime" "${_arch_include}"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/runtime/include"
    NO_DEFAULT_PATH)
add_library(deepep_kernels SHARED "${DEEPEP_SOURCE_DIR}/csrc/kernels/notify/notify.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/dispatch/dispatch.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/combine.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/dispatch/low_latency.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/low_latency.cpp")
target_compile_options(deepep_kernels PRIVATE "$<$<NOT:$<CONFIG:Debug>>:-O2>" -xasc --npu-arch=dav-3510
    --cce-auto-infer-kernel-type=false
    "-ffile-prefix-map=${DEEPEP_SOURCE_DIR}=."
    "-ffile-prefix-map=${ASCEND_HOME_PATH}=CANN_SDK"
    "-ffile-prefix-map=${SHMEM_ROOT}=SHMEM_SDK")
target_compile_definitions(deepep_kernels PRIVATE CATLASS_ARCH=3510 USE_SIMT L2_CACHE_HINT)
target_include_directories(deepep_kernels PRIVATE
    "${_sdk_include}" "${_arch_include}"
    "${_tikcpp}" "${_tikcpp}/tikcfw"
    "${_tikcpp}/tikcfw/impl" "${_tikcpp}/tikcfw/interface"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device"
    "${SHMEM_ROOT}/src/device_simt")
if(DEVICE_RT_INCLUDE)
    target_include_directories(deepep_kernels PRIVATE "${DEVICE_RT_INCLUDE}")
    message(STATUS "Device runtime compatibility headers: ${DEVICE_RT_INCLUDE}")
else()
    message(STATUS "No standalone rt.h found; using the selected CANN -xasc toolchain headers")
endif()
target_link_options(deepep_kernels PRIVATE --cce-fatobj-link -Wl,-z,relro,-z,now,-z,noexecstack)
target_link_libraries(deepep_kernels PRIVATE ascend_runtime ascendcl shmem ${CMAKE_DL_LIBS})
set_target_properties(deepep_kernels PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
