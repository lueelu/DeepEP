cmake_minimum_required(VERSION 3.20)
project(udma_bench_device LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include("${DEEPEP_SOURCE_DIR}/cmake/Dependencies.cmake")
add_library(udma_bench_kernel SHARED "${DEEPEP_SOURCE_DIR}/benchmarks/udma/kernel.cpp")
target_compile_options(udma_bench_kernel PRIVATE -O2 -xasc --npu-arch=dav-3510
    --cce-auto-infer-kernel-type=false -Xhost-start -ftrapv -Xhost-end)
target_compile_definitions(udma_bench_kernel PRIVATE ASCEND_DEEPEP_ARCH=3510 USE_SIMT L2_CACHE_HINT)
target_include_directories(udma_bench_kernel PRIVATE "${DEEPEP_SOURCE_DIR}/csrc/direct"
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${ASCEND_HOME_PATH}/include/experiment/runtime" "${ASCEND_HOME_PATH}/include/runtime"
    "${ASCEND_HOME_PATH}/compiler/tikcpp" "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw"
    "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl" "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device" "${SHMEM_ROOT}/src/device_simt")
target_link_options(udma_bench_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(udma_bench_kernel PRIVATE ascend_runtime ascendcl shmem ${CMAKE_DL_LIBS})
