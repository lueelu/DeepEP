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
set(kernel_flags
    -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false
    --cce-auto-sync=off --cce-fatobj-link -shared -fPIC -Wl,-soname,libmegamoe_kernel.so
    -DCATLASS_ARCH=3510 -DASCEND_DEEPEP_DIRECT_TENSOR_INPUT=1
    -DASCEND_DEEPEP_DISABLE_MEGA_MOE_EXCEPTION_DUMP=1 -DENABLE_TENSOR_API
    -DNOT_DYNAMIC_COMPILE -DHIGH_PERFORMANCE=1 -DDETERMINISTIC_MODE=0 -DL2_CACHE_HINT)
if(DEEPEP_MEGAMOE_TIMER)
    list(APPEND kernel_flags -DDEEPEP_MEGAMOE_TIMER=1)
endif()
foreach(path IN LISTS beta_include)
    list(APPEND kernel_flags "-I${path}")
endforeach()
find_file(api_header ascendc_api_version.h
    PATHS "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/lib"
          "${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/lib" NO_DEFAULT_PATH)
if(api_header)
    file(STRINGS "${api_header}" api_version REGEX "^#define ASCENDC_API_VERSION [0-9]+")
    string(REGEX REPLACE ".*ASCENDC_API_VERSION ([0-9]+).*" "\\1" api_version "${api_version}")
    list(APPEND kernel_flags "-DASCENDC_API_VERSION=${api_version}")
endif()
set(kernel_library "${CMAKE_CURRENT_BINARY_DIR}/libmegamoe_kernel.so")
file(GLOB_RECURSE kernel_headers CONFIGURE_DEPENDS "csrc/kernels/*.hpp")
add_custom_command(OUTPUT "${kernel_library}"
    COMMAND "${BISHENG}" ${kernel_flags} csrc/kernels/mega_moe/mega_moe.cpp
        "-L${ASCEND_HOME_PATH}/lib64" "-L${SHMEM_ROOT}/lib"
        -lruntime -lascendcl -lshmem -lgraph_base -ltiling_api -lplatform -lregister -ldl -lstdc++ -lm
        -o "${kernel_library}"
    DEPENDS csrc/kernels/mega_moe/mega_moe.cpp csrc/kernel_launch.hpp ${kernel_headers}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" VERBATIM)
add_custom_target(megamoe_device DEPENDS "${kernel_library}")
pybind11_add_module(megamoe_ext csrc/bindings.cpp csrc/forward.cpp)
set_target_properties(megamoe_ext PROPERTIES OUTPUT_NAME _C
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
target_include_directories(megamoe_ext PRIVATE ${beta_include} ${TORCH_INCLUDE_DIRS} "${torch_npu_root}/include")
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(megamoe_ext PRIVATE ${torch_flags} -O2 -fstack-protector-strong)
target_compile_definitions(megamoe_ext PRIVATE CATLASS_ARCH=3510)
if(DEEPEP_MEGAMOE_TIMER)
    target_compile_definitions(megamoe_ext PRIVATE DEEPEP_MEGAMOE_TIMER=1)
endif()
add_library(megamoe_kernel SHARED IMPORTED)
set_target_properties(megamoe_kernel PROPERTIES IMPORTED_LOCATION "${kernel_library}")
add_dependencies(megamoe_ext megamoe_device)
target_link_libraries(megamoe_ext PRIVATE ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}"
    "${TORCH_NPU_LIBRARY}" megamoe_kernel shmem ascendcl)
target_link_directories(megamoe_ext PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(megamoe_ext PRIVATE platform tiling_api)
target_link_options(megamoe_ext PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
install(TARGETS megamoe_ext LIBRARY DESTINATION ../deep_ep_experimental/megamoe)
install(FILES "${kernel_library}" DESTINATION ../deep_ep_experimental/megamoe)
