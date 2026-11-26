# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

# MegaMoE is an optional Torch-bound extension. The core _C keeps its original ABI.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Headers.cmake")
ascend_deepep_find_mega_moe_headers()
find_program(BISHENG bisheng PATHS "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin" NO_DEFAULT_PATH REQUIRED)
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

set(DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE "0" CACHE STRING "BF16 training forward: 0=interleaved, 1=staged")
if(NOT DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE MATCHES "^[01]$")
    message(FATAL_ERROR "DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE must be 0 or 1")
endif()

set(megamoe_definitions CATLASS_ARCH=3510)
if(DEEPEP_MEGAMOE_TIMER)
    list(APPEND megamoe_definitions DEEPEP_MEGAMOE_TIMER=1)
endif()
set(quantized_include_dirs
    "${PROJECT_SOURCE_DIR}"
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
set(bf16_include_dirs "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mega_moe"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mega_moe/common"
    "${CATLASS_ROOT}/include" "${CATLASS_ROOT}/examples/common"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device"
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime" "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime"
    "${ASCEND_HOME_PATH}/aarch64-linux/include" "${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc"
    "${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include"
    "${ASCEND_HOME_PATH}/compiler/tikcpp" "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw"
    "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface" "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl"
    "${PROJECT_SOURCE_DIR}")
# BF16 Host and device share the schedule and workspace definitions.
set(bf16_definitions DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE=${DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE})

# One compile configuration for every entry: the training kernels align to
# the quantized regime (auto-sync off, static compile, tensor API), keeping
# only the BF16 schedule macros the code consumes.
set(megamoe_device_flags -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510
    --cce-auto-infer-kernel-type=false --cce-auto-sync=off -DL2_CACHE_HINT
    -DASCEND_DEEPEP_DIRECT_TENSOR_INPUT=1
    -DASCEND_DEEPEP_DISABLE_MEGA_MOE_EXCEPTION_DUMP=1 -DENABLE_TENSOR_API
    -DNOT_DYNAMIC_COMPILE -DHIGH_PERFORMANCE=1 -DDETERMINISTIC_MODE=0)
foreach(definition IN LISTS megamoe_definitions bf16_definitions)
    list(APPEND megamoe_device_flags "-D${definition}")
endforeach()
foreach(path IN LISTS quantized_include_dirs bf16_include_dirs)
    list(APPEND megamoe_device_flags "-I${path}")
endforeach()
find_file(api_header ascendc_api_version.h
    PATHS "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/lib"
    "${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/lib" NO_DEFAULT_PATH)
if(api_header)
    file(STRINGS "${api_header}" api_version REGEX "^#define ASCENDC_API_VERSION [0-9]+")
    string(REGEX REPLACE ".*ASCENDC_API_VERSION ([0-9]+).*" "\\1" api_version "${api_version}")
    list(APPEND megamoe_device_flags "-DASCENDC_API_VERSION=${api_version}")
endif()

file(GLOB_RECURSE megamoe_headers CONFIGURE_DEPENDS "csrc/*.hpp" "${PROJECT_SOURCE_DIR}/csrc/kernels/common/*.hpp")
set(megamoe_objects)
set(entry_source "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mega_moe/mega_moe.cpp")
set(object "${CMAKE_CURRENT_BINARY_DIR}/mega_moe.o")
add_custom_command(OUTPUT "${object}"
    COMMAND "${BISHENG}" ${megamoe_device_flags} -c "${entry_source}" -o "${object}"
    DEPENDS "${entry_source}" ${megamoe_headers} VERBATIM)
list(APPEND megamoe_objects "${object}")

set(kernel_library "${CMAKE_CURRENT_BINARY_DIR}/libmegamoe_kernel.so")
add_custom_command(OUTPUT "${kernel_library}"
    COMMAND "${BISHENG}" -shared --cce-fatobj-link ${megamoe_objects}
    "-L${ASCEND_HOME_PATH}/lib64" "-L${SHMEM_ROOT}/lib"
    -Wl,--no-as-needed -lruntime -lascendcl -lshmem -lgraph_base -ltiling_api -lplatform -lregister -ldl
    -Wl,--as-needed -lstdc++ -lm "-Wl,-rpath,$ORIGIN"
    -Wl,-soname,libmegamoe_kernel.so -o "${kernel_library}"
    DEPENDS ${megamoe_objects} VERBATIM)
add_custom_target(megamoe_device DEPENDS "${kernel_library}")
add_library(megamoe_kernel SHARED IMPORTED)
set_target_properties(megamoe_kernel PROPERTIES IMPORTED_LOCATION "${kernel_library}")

pybind11_add_module(megamoe_ext csrc/bindings.cpp csrc/plan_quantized.cpp csrc/plan_bf16.cpp
    csrc/forward.cpp csrc/backward.cpp)
set_target_properties(megamoe_ext PROPERTIES OUTPUT_NAME _C
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
target_include_directories(megamoe_ext PRIVATE ${quantized_include_dirs} ${bf16_include_dirs}
    ${TORCH_INCLUDE_DIRS} "${torch_npu_root}/include" "${torch_npu_root}/..")
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(megamoe_ext PRIVATE ${torch_flags} -O2 -fstack-protector-strong)
target_compile_definitions(megamoe_ext PRIVATE ${megamoe_definitions} ${bf16_definitions})
target_link_libraries(megamoe_ext PRIVATE ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}"
    "${TORCH_NPU_LIBRARY}" megamoe_kernel shmem ascendcl ascend_runtime)
target_link_directories(megamoe_ext PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(megamoe_ext PRIVATE platform tiling_api)
target_link_options(megamoe_ext PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
add_dependencies(megamoe_ext megamoe_device)

install(TARGETS megamoe_ext LIBRARY DESTINATION ../deep_ep_experimental/megamoe)
install(FILES "${kernel_library}" DESTINATION ../deep_ep_experimental/megamoe)
