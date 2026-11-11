# Copyright (c) 2026, Lu Lu
cmake_minimum_required(VERSION 3.20)
project(deepep_ubx_device LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Dependencies.cmake")
set(_tik "${ASCEND_HOME_PATH}/compiler/tikcpp")
add_library(deepep_ubx_kernels SHARED combine.cpp)
target_compile_options(deepep_ubx_kernels PRIVATE -O2 -xasc --npu-arch=dav-3510
    --cce-auto-infer-kernel-type=false)
target_compile_definitions(deepep_ubx_kernels PRIVATE CATLASS_ARCH=3510 USE_SIMT L2_CACHE_HINT
    ACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1 ACLSHMEMI_HNS_1825_PI_ON_CHIP=${DEEPEP_UBX_PI_ON_CHIP}
    DEEPEP_UBX_RANKS_PER_HOST=${DEEPEP_UBX_RANKS_PER_HOST})
target_include_directories(deepep_ubx_kernels PRIVATE "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${_tik}" "${_tik}/tikcfw" "${_tik}/tikcfw/impl" "${_tik}/tikcfw/interface"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device" "${SHMEM_ROOT}/src/device_simt")
target_link_options(deepep_ubx_kernels PRIVATE --cce-fatobj-link)
target_link_libraries(deepep_ubx_kernels PRIVATE ascendcl ascend_runtime shmem ${CMAKE_DL_LIBS})
set_target_properties(deepep_ubx_kernels PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
