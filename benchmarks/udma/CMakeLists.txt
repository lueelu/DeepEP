cmake_minimum_required(VERSION 3.20)
project(udma_microbench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(DEEPEP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
foreach(sdk ASCEND_HOME_PATH SHMEM_ROOT)
    if(NOT DEFINED ${sdk})
        set(${sdk} "$ENV{${sdk}}")
    endif()
endforeach()
include("${DEEPEP_SOURCE_DIR}/cmake/Dependencies.cmake")
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import torch, pybind11; from pathlib import Path; print(Path(torch.__file__).parent); print(int(torch._C._GLIBCXX_USE_CXX11_ABI)); print(pybind11.get_cmake_dir())"
    COMMAND_ERROR_IS_FATAL ANY OUTPUT_VARIABLE _python OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" _python "${_python}")
list(GET _python 0 _torch)
list(GET _python 1 _abi)
list(GET _python 2 pybind11_DIR)
set(Torch_DIR "${_torch}/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Compile only this diagnostic kernel in a separate Bisheng build tree.
add_custom_target(udma_bench_device_build
    COMMAND "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_LIST_DIR}/device" -B "${CMAKE_BINARY_DIR}/device"
        "-DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng"
        "-DDEEPEP_SOURCE_DIR=${DEEPEP_SOURCE_DIR}" "-DASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
        "-DSHMEM_ROOT=${SHMEM_ROOT}" "-DCMAKE_BUILD_TYPE=Release"
    COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}/device" --parallel 2
    BYPRODUCTS "${CMAKE_BINARY_DIR}/device/libudma_bench_kernel.so" VERBATIM)
add_library(udma_bench_device SHARED IMPORTED)
set_target_properties(udma_bench_device PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/device/libudma_bench_kernel.so")
add_dependencies(udma_bench_device udma_bench_device_build)
pybind11_add_module(_udma_microbench NO_EXTRAS native.cpp
    "${DEEPEP_SOURCE_DIR}/csrc/direct/runtime/shmem_instance.cpp")
target_include_directories(_udma_microbench PRIVATE "${DEEPEP_SOURCE_DIR}/csrc/direct"
    ${TORCH_INCLUDE_DIRS} "${ASCEND_HOME_PATH}/include" "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src")
target_compile_definitions(_udma_microbench PRIVATE _GLIBCXX_USE_CXX11_ABI=${_abi})
target_compile_options(_udma_microbench PRIVATE -O2 -Wall -Wextra)
set_source_files_properties(native.cpp PROPERTIES COMPILE_OPTIONS "-Werror")
target_link_libraries(_udma_microbench PRIVATE ${TORCH_LIBRARIES} ascendcl ascend_runtime shmem
    udma_bench_device ${CMAKE_DL_LIBS})
