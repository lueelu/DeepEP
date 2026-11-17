# Copyright (c) 2026, Lu Lu
# Modified by candy_cloud_fly 2026

cmake_minimum_required(VERSION 3.20)
project(deepep_device LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include("${CMAKE_CURRENT_LIST_DIR}/../Dependencies.cmake")
# Reconfigure when the SHMEM development identity changes, so a wheel cannot
# silently pair new sources with a stale SDK manifest.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${SHMEM_ROOT}/deepep-sdk.json")
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

# Keep the low-latency translation units in plain and reference object targets
# so an ON diagnostic build reuses dependency-tracked objects. Profiles are a
# compile-time identity: the device never enables them from a runtime flag.
set(_ll_sources
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/runtime/clock_alignment.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/dispatch/low_latency.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/low_latency.cpp")
# Other device kernels keep the upstream multi-backend build unchanged.
set(_other_sources
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/notify/notify.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/dispatch/dispatch.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/combine.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/reuse.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/combine_ep8.cpp"
    "${DEEPEP_SOURCE_DIR}/csrc/ubx/kernels/combine.cpp")
if(DEEPEP_LL_STAGE_TIMER)
    message(FATAL_ERROR "Legacy LL tracing was removed; use DEEPEP_LL_REFERENCE_PROFILE=ON")
endif()
add_library(deepep_kernel_ll_plain OBJECT ${_ll_sources})
add_library(deepep_kernel_ll_reference OBJECT ${_ll_sources})
set_target_properties(deepep_kernel_ll_plain deepep_kernel_ll_reference
    PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_compile_definitions(deepep_kernel_ll_reference PRIVATE DEEPEP_LL_REFERENCE_PROFILE=1)
if(DEEPEP_LL_REFERENCE_PROFILE)
    set(_ll_variant deepep_kernel_ll_reference)
else()
    set(_ll_variant deepep_kernel_ll_plain)
endif()

add_library(deepep_kernels SHARED $<TARGET_OBJECTS:${_ll_variant}> ${_other_sources})

set(_device_compile_options "$<$<NOT:$<CONFIG:Debug>>:-O2>" -xasc --npu-arch=dav-3510
    --cce-auto-infer-kernel-type=false
    "-ffile-prefix-map=${DEEPEP_SOURCE_DIR}=."
    "-ffile-prefix-map=${ASCEND_HOME_PATH}=CANN_SDK"
    "-ffile-prefix-map=${SHMEM_ROOT}=SHMEM_SDK")
set(_device_defines CATLASS_ARCH=3510 USE_SIMT L2_CACHE_HINT
    DEEPEP_LL_RDMA=1 ACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1
    ACLSHMEMI_HNS_1825_PI_ON_CHIP=${DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP} deepep_kernels_EXPORTS)
set(_device_includes
    "${_sdk_include}" "${_arch_include}"
    "${_tikcpp}" "${_tikcpp}/tikcfw"
    "${_tikcpp}/tikcfw/impl" "${_tikcpp}/tikcfw/interface"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device"
    "${SHMEM_ROOT}/src/device_simt")
if(DEVICE_RT_INCLUDE)
    list(APPEND _device_includes "${DEVICE_RT_INCLUDE}")
endif()

add_library(deepep_kernel_options INTERFACE)
target_compile_options(deepep_kernel_options INTERFACE ${_device_compile_options})
target_compile_definitions(deepep_kernel_options INTERFACE ${_device_defines})
target_include_directories(deepep_kernel_options INTERFACE ${_device_includes})
foreach(_objects deepep_kernel_ll_plain deepep_kernel_ll_reference)
    target_link_libraries(${_objects} PRIVATE deepep_kernel_options)
endforeach()

target_compile_options(deepep_kernels PRIVATE ${_device_compile_options})
target_compile_definitions(deepep_kernels PRIVATE ${_device_defines})
target_include_directories(deepep_kernels PRIVATE ${_device_includes})
if(DEVICE_RT_INCLUDE)
    message(STATUS "Device runtime compatibility headers: ${DEVICE_RT_INCLUDE}")
else()
    message(STATUS "No standalone rt.h found; using the selected CANN -xasc toolchain headers")
endif()
target_link_options(deepep_kernels PRIVATE --cce-fatobj-link -Wl,-z,relro,-z,now,-z,noexecstack)
target_link_libraries(deepep_kernels PRIVATE ascend_runtime ascendcl shmem ${CMAKE_DL_LIBS})
set_target_properties(deepep_kernels PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)

# Separate translation units and library preserve both backends' device ABI.
set(_direct "${DEEPEP_SOURCE_DIR}/csrc/direct")
add_library(deepep_clos_kernels SHARED
    "${_direct}/kernels/dispatch/dispatch_layout.cpp" "${_direct}/kernels/notify/notify_dispatch.cpp"
    "${_direct}/kernels/dispatch/dispatch.cpp" "${_direct}/kernels/dispatch/dispatch_udma.cpp"
    "${_direct}/kernels/notify/notify_combine.cpp" "${_direct}/kernels/combine/combine.cpp"
    "${_direct}/kernels/combine/combine_udma.cpp" "${_direct}/kernels/expand/expand.cpp"
    "${_direct}/kernels/aggregate/aggregate.cpp")
get_target_property(_direct_options deepep_kernels COMPILE_OPTIONS)
get_target_property(_direct_includes deepep_kernels INCLUDE_DIRECTORIES)
target_compile_options(deepep_clos_kernels PRIVATE ${_direct_options} -Xhost-start -ftrapv -Xhost-end)
target_include_directories(deepep_clos_kernels PRIVATE ${_direct_includes}
    "${_direct}")
target_compile_definitions(deepep_clos_kernels PRIVATE ASCEND_DEEPEP_ARCH=3510 EP_NUM_TOPK_IDX_BITS=64 USE_SIMT L2_CACHE_HINT
    ASCEND_DEEPEP_COMBINE_SIMT_WEIGHTS=1)
target_link_options(deepep_clos_kernels PRIVATE --cce-fatobj-link)
target_link_libraries(deepep_clos_kernels PRIVATE ascend_runtime ascendcl shmem ${CMAKE_DL_LIBS})
set_target_properties(deepep_clos_kernels PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 16)
# Preserve the source build's translation-unit grouping. In particular its
# local expand/aggregate helpers share the SHMEM/SIMT setup from notify.
set_source_files_properties(
    "${_direct}/kernels/dispatch/dispatch.cpp" "${_direct}/kernels/dispatch/dispatch_udma.cpp"
    "${_direct}/kernels/combine/combine.cpp" "${_direct}/kernels/combine/combine_udma.cpp"
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
