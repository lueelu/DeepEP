cmake_minimum_required(VERSION 3.20)
project(hierarchical_alltoall LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
foreach(required ASCEND_HOME_PATH SHMEM_ROOT CATLASS_SOURCE_PATH GROUP_SIZE COMM_BLOCK_M RDMA_BACKEND)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required; use scripts/build.sh")
    endif()
endforeach()
if(NOT GROUP_SIZE MATCHES "^(1|2|4|8|16)$" OR NOT COMM_BLOCK_M MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "Invalid GROUP_SIZE or COMM_BLOCK_M")
endif()
if(NOT RDMA_BACKEND MATCHES "^(HNS_1825|XSCALE)$")
    message(FATAL_ERROR "RDMA_BACKEND must be HNS_1825 or XSCALE")
endif()
if(RDMA_BACKEND STREQUAL "HNS_1825" AND NOT HNS_PI_ON_CHIP MATCHES "^[01]$")
    message(FATAL_ERROR "HNS_PI_ON_CHIP must match the installed SDK and firmware: 0 or 1")
endif()
get_filename_component(DEEPEP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_subdirectory("${DEEPEP_SOURCE_DIR}/csrc/comm" "${CMAKE_CURRENT_BINARY_DIR}/comm")
foreach(header "${CATLASS_SOURCE_PATH}/include/catlass/catlass.hpp"
               "${SHMEM_ROOT}/include/shmem.h")
    if(NOT EXISTS "${header}")
        message(FATAL_ERROR "Missing ${header}; select installed SHMEM and initialized Catlass sources")
    endif()
endforeach()
file(READ "${SHMEM_ROOT}/include/host/shmem_host_def.h" shmem_def)
file(READ "${SHMEM_ROOT}/include/host/init/shmem_host_init.h" shmem_init)
if(shmem_def MATCHES "aclshmem_runtime_config_t runtime_config")
    set(runtime_config 1)
elseif(shmem_init MATCHES "aclshmemx_set_shared_jetty")
    set(runtime_config 0)
else()
    message(FATAL_ERROR "SDK cannot explicitly disable UDMA shared jetty")
endif()
add_executable(hierarchical_alltoall main.cpp kernel.cpp)
# Host stays ordinary C++; only the device translation unit needs AscendC.
set_source_files_properties(kernel.cpp PROPERTIES COMPILE_OPTIONS "-xasc;--npu-arch=dav-3510;--cce-auto-infer-kernel-type=false")
target_compile_options(hierarchical_alltoall PRIVATE -O2 -fPIC)
target_compile_definitions(hierarchical_alltoall PRIVATE CATLASS_ARCH=3510 L2_CACHE_HINT
    HIERARCHICAL_GROUP_SIZE=${GROUP_SIZE} HIERARCHICAL_COMM_BLOCK_M=${COMM_BLOCK_M}
    HIERARCHICAL_SHMEM_RUNTIME_CONFIG=${runtime_config} ACLSHMEMI_RDMA_K_BACKEND_${RDMA_BACKEND}=1)
if(RDMA_BACKEND STREQUAL "HNS_1825")
    target_compile_definitions(hierarchical_alltoall PRIVATE ACLSHMEMI_HNS_1825_PI_ON_CHIP=${HNS_PI_ON_CHIP})
endif()
set(tikcpp "${ASCEND_HOME_PATH}/compiler/tikcpp")
target_include_directories(hierarchical_alltoall PRIVATE
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device" "${SHMEM_ROOT}/src/device_simt"
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/experiment"
    "${ASCEND_HOME_PATH}/include/runtime" "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/runtime"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime"
    "${tikcpp}" "${tikcpp}/tikcfw" "${tikcpp}/tikcfw/impl" "${tikcpp}/tikcfw/interface")
# A full library path pins the selected SHMEM SDK at link time.
find_library(shmem_library NAMES shmem PATHS "${SHMEM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ascend_runtime NAMES runtime PATHS "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64" NO_DEFAULT_PATH REQUIRED)
target_link_directories(hierarchical_alltoall PRIVATE "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
target_link_libraries(hierarchical_alltoall PRIVATE deepep_comm "${shmem_library}" "${ascend_runtime}" ascendcl tiling_api platform ${CMAKE_DL_LIBS})
target_link_options(hierarchical_alltoall PRIVATE --cce-fatobj-link)
set_target_properties(hierarchical_alltoall PROPERTIES BUILD_RPATH "${SHMEM_ROOT}/lib;${ASCEND_HOME_PATH}/lib64")
