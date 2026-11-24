# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

# MegaMoE is an optional Torch-bound extension. Its device code is compiled at
# runtime through DeepJIT from the kernel sources shipped in the wheel, so this
# target only builds the Host module; no Bisheng device library is produced.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Headers.cmake")
ascend_deepep_find_mega_moe_headers()
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import torch,torch_npu; from pathlib import Path; print(torch.utils.cmake_prefix_path); print(Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE framework_paths OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" framework_paths "${framework_paths}")
list(GET framework_paths 0 torch_cmake)
list(GET framework_paths 1 torch_npu_root)
list(APPEND CMAKE_PREFIX_PATH "${torch_cmake}")
find_package(Torch REQUIRED)
find_library(TORCH_NPU_LIBRARY torch_npu PATHS "${torch_npu_root}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

set(beta_include
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mega_moe"
    "${CATLASS_ROOT}/include" "${CATLASS_ROOT}/examples/common"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device"
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/include/op_common"
    "${ASCEND_HOME_PATH}/pkg_inc/op_common"
    "${ASCEND_HOME_PATH}/compiler/ascendc/act"
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/atcos"
    "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}"
    "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}/tensor_api"
    "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}/tensor_api/include")

# Content identity of everything the runtime compiler will read. It feeds the
# DeepJIT cache key so a wheel with different kernel sources never reuses a
# binary compiled from the previous tree. Paths are stripped to stay
# machine-independent; the flags themselves are hashed by DeepJIT separately.
set(_mega_jit_identity "")
file(GLOB_RECURSE _mega_jit_own CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*")
file(GLOB_RECURSE _mega_jit_catlass CONFIGURE_DEPENDS
    "${CATLASS_ROOT}/include/*.h" "${CATLASS_ROOT}/include/*.hpp"
    "${CATLASS_ROOT}/examples/common/*.h" "${CATLASS_ROOT}/examples/common/*.hpp")
file(GLOB_RECURSE _mega_jit_shmem CONFIGURE_DEPENDS
    "${SHMEM_ROOT}/include/*.h" "${SHMEM_ROOT}/include/*.hpp"
    "${SHMEM_ROOT}/src/*.h" "${SHMEM_ROOT}/src/*.hpp")
foreach(_file IN LISTS _mega_jit_own _mega_jit_catlass _mega_jit_shmem)
    file(SHA256 "${_file}" _digest)
    string(APPEND _mega_jit_identity "${_digest}\n")
endforeach()
string(APPEND _mega_jit_identity "pi=${DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP}\n")
if(DEEPEP_MEGAMOE_TIMER)
    string(APPEND _mega_jit_identity "timer=1\n")
endif()
string(SHA256 DEEPEP_MEGAMOE_JIT_SIGNATURE "${_mega_jit_identity}")
message(STATUS "MegaMoE runtime kernel source identity: ${DEEPEP_MEGAMOE_JIT_SIGNATURE}")

pybind11_add_module(megamoe_ext csrc/bindings.cpp csrc/forward.cpp csrc/kernel_launch.cpp csrc/jit/mega_moe_jit.cpp)
set_target_properties(megamoe_ext PROPERTIES OUTPUT_NAME _C CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
target_include_directories(megamoe_ext PRIVATE ${beta_include} ${TORCH_INCLUDE_DIRS} "${torch_npu_root}/include")
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(megamoe_ext PRIVATE ${torch_flags} -O2 -fstack-protector-strong)
target_compile_definitions(megamoe_ext PRIVATE CATLASS_ARCH=3510
    "DEEPEP_MEGAMOE_PI_ON_CHIP=${DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP}"
    "DEEPEP_MEGAMOE_JIT_SIGNATURE=\"${DEEPEP_MEGAMOE_JIT_SIGNATURE}\"")
if(DEEPEP_MEGAMOE_TIMER)
    target_compile_definitions(megamoe_ext PRIVATE DEEPEP_MEGAMOE_TIMER=1)
endif()
deepep_apply_deepjit(megamoe_ext)
target_link_libraries(megamoe_ext PRIVATE ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}"
    "${TORCH_NPU_LIBRARY}" shmem ascendcl)
target_link_directories(megamoe_ext PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(megamoe_ext PRIVATE platform tiling_api ${CMAKE_DL_LIBS})
target_link_options(megamoe_ext PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
install(TARGETS megamoe_ext LIBRARY DESTINATION ../deep_ep_experimental/megamoe)
