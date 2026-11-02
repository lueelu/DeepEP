# The Clos backend uses the same external CANN/SHMEM SDK as the POD backend.
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import torch, torch_npu; from pathlib import Path; print(Path(torch.__file__).parent); print(Path(torch_npu.__file__).parent); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    RESULT_VARIABLE _clos_torch_result OUTPUT_VARIABLE _clos_torch OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _clos_torch_result EQUAL 0)
    message(FATAL_ERROR "V1/V2 Clos build requires matching torch and torch_npu in the build Python environment")
endif()
string(REPLACE "\n" ";" _clos_torch "${_clos_torch}")
list(GET _clos_torch 0 _torch_root)
list(GET _clos_torch 1 _torch_npu_root)
list(GET _clos_torch 2 _torch_abi)
set(Torch_DIR "${_torch_root}/share/cmake/Torch")
find_package(Torch REQUIRED)
find_library(_torch_python NAMES torch_python PATHS "${_torch_root}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(_torch_npu NAMES torch_npu PATHS "${_torch_npu_root}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(_tiling NAMES tiling_api PATHS "${ASCEND_HOME_PATH}/lib64" REQUIRED)
find_library(_platform NAMES platform PATHS "${ASCEND_HOME_PATH}/lib64" REQUIRED)

set(_clos_root "${CMAKE_SOURCE_DIR}/csrc/clos")
pybind11_add_module(_clos_C NO_EXTRAS
    "${_clos_root}/bindings/pybind.cpp" "${_clos_root}/bindings/torch_classes.cpp"
    "${_clos_root}/ops/launcher.cpp" "${_clos_root}/runtime/shmem_instance.cpp"
    "${_clos_root}/runtime/buffer_runtime.cpp"
    "${_clos_root}/ops/aggregate.cpp" "${_clos_root}/ops/combine.cpp"
    "${_clos_root}/ops/dispatch.cpp" "${_clos_root}/ops/expand.cpp")
target_include_directories(_clos_C PRIVATE "${_clos_root}"
    ${TORCH_INCLUDE_DIRS} "${_torch_npu_root}/include" "${_torch_npu_root}/.."
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src")
target_compile_definitions(_clos_C PRIVATE TORCH_EXTENSION_NAME=_clos_C
    _GLIBCXX_USE_CXX11_ABI=${_torch_abi} ASCEND_DEEPEP_ARCH=3510 EP_NUM_TOPK_IDX_BITS=64)
target_compile_options(_clos_C PRIVATE -O2)
target_link_libraries(_clos_C PRIVATE ${TORCH_LIBRARIES} "${_torch_python}" "${_torch_npu}"
    ascendcl ascend_runtime shmem deepep_clos_device "${_tiling}" "${_platform}" ${CMAKE_DL_LIBS})
set_target_properties(_clos_C PROPERTIES INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS _clos_C LIBRARY DESTINATION .)
