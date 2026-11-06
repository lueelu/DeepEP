# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

# BF16 training is an independent optional component; root owns the SDKs.
option(DEEPEP_BF16_FORCE_EXPERT_READY "Wait for whole-expert Dispatch in training forward and backward" OFF)
if(DEEPEP_BF16_FORCE_EXPERT_READY)
    set(train_force_expert_ready 1)
else()
    set(train_force_expert_ready 0)
endif()
find_program(TRAIN_BISHENG bisheng PATHS "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin" NO_DEFAULT_PATH REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import torch,torch_npu; from pathlib import Path; print(torch.utils.cmake_prefix_path); print(Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE framework_paths OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" framework_paths "${framework_paths}")
list(GET framework_paths 0 torch_cmake)
list(GET framework_paths 1 torch_npu_root)
list(APPEND CMAKE_PREFIX_PATH "${torch_cmake}")
find_package(Torch REQUIRED)
find_library(TRAIN_TORCH_NPU torch_npu PATHS "${torch_npu_root}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(TRAIN_TORCH_PYTHON torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

set(train_include "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/bf16"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/bf16/common"
    "${CATLASS_ROOT}/include" "${CATLASS_ROOT}/examples/common"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device"
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime" "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime"
    "${ASCEND_HOME_PATH}/aarch64-linux/include" "${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc"
    "${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include"
    "${ASCEND_HOME_PATH}/compiler/tikcpp" "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw"
    "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface" "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl")
# Training uses adaptive GMM1 scheduling and automatic SwiGLU handoff by default.
# Forward SwiGLU only: 0 = force UB handoff, 1 = force GM staging, 2 = auto.
# Host queries the built device SO for workspace sizing; the shared header's
# fallback value 0 is not the default of this device build, which uses 2 below.
set(train_device_flags -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510
    --cce-auto-infer-kernel-type=false -Xhost-start -ftrapv -D__ASC_NPU_HOST__ -Xhost-end
    -DCATLASS_ARCH=3510 -DL2_CACHE_HINT
    -DDEEPEP_BF16_FORCE_EXPERT_READY=${train_force_expert_ready}
    -DDEEPEP_BF16_SWIGLU_DECOUPLE=2)
foreach(path IN LISTS train_include)
    list(APPEND train_device_flags "-I${path}")
endforeach()
file(GLOB_RECURSE train_headers CONFIGURE_DEPENDS "csrc/*.h" "csrc/*.hpp")
set(train_objects)
foreach(source mega_moe_bf16_forward mega_moe_bf16_backward)
    set(object "${CMAKE_CURRENT_BINARY_DIR}/${source}.o")
    add_custom_command(OUTPUT "${object}"
        COMMAND "${TRAIN_BISHENG}" ${train_device_flags} -c
            "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/bf16/${source}.cpp" -o "${object}"
        DEPENDS "csrc/kernels/bf16/${source}.cpp" ${train_headers} VERBATIM)
    list(APPEND train_objects "${object}")
endforeach()
set(train_library "${CMAKE_CURRENT_BINARY_DIR}/libdeepep_megamoe_train_bf16.so")
add_custom_command(OUTPUT "${train_library}"
    COMMAND "${TRAIN_BISHENG}" -shared --cce-fatobj-link ${train_objects}
        "-L${ASCEND_HOME_PATH}/lib64" "-L${SHMEM_ROOT}/lib"
        -Wl,--no-as-needed -lruntime -lascendcl -lshmem -lgraph_base -ltiling_api -lplatform -lregister -ldl
        -Wl,--as-needed -lstdc++ -lm "-Wl,-rpath,$ORIGIN"
        -Wl,-soname,libdeepep_megamoe_train_bf16.so -o "${train_library}"
    DEPENDS ${train_objects} VERBATIM)
add_custom_target(megamoe_train_device DEPENDS "${train_library}")
add_library(megamoe_train_kernel SHARED IMPORTED)
set_target_properties(megamoe_train_kernel PROPERTIES IMPORTED_LOCATION "${train_library}")
pybind11_add_module(megamoe_train_ext csrc/bindings.cpp csrc/bf16/launcher.cpp
    csrc/bf16/tiling.cpp csrc/bf16/training.cpp)
set_target_properties(megamoe_train_ext PROPERTIES OUTPUT_NAME _C
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
target_include_directories(megamoe_train_ext PRIVATE ${train_include} ${TORCH_INCLUDE_DIRS}
    "${torch_npu_root}/include" "${torch_npu_root}/..")
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(megamoe_train_ext PRIVATE ${torch_flags} -O2 -fstack-protector-strong)
target_compile_definitions(megamoe_train_ext PRIVATE
    DEEPEP_BF16_FORCE_EXPERT_READY=${train_force_expert_ready})
target_link_directories(megamoe_train_ext PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(megamoe_train_ext PRIVATE ${TORCH_LIBRARIES} "${TRAIN_TORCH_PYTHON}"
    "${TRAIN_TORCH_NPU}" megamoe_train_kernel shmem ascendcl ascend_runtime platform tiling_api)
target_link_options(megamoe_train_ext PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
add_dependencies(megamoe_train_ext megamoe_train_device)
install(TARGETS megamoe_train_ext LIBRARY DESTINATION ../deep_ep_experimental/megamoe_train)
install(FILES "${train_library}"
    DESTINATION ../deep_ep_experimental/megamoe_train)
