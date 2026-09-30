# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

# Keep Bisheng in its own build tree, separate from the Host C++ compiler.
set(DEEPEP_DEVICE_BINARY_DIR "${CMAKE_BINARY_DIR}/device")
set(DEEPEP_DEVICE_LIBRARY "${DEEPEP_DEVICE_BINARY_DIR}/libdeepep_kernels.so")
set(DEEPEP_DEVICE_JOBS "4" CACHE STRING "Parallel Device compilation jobs")
if(NOT DEEPEP_DEVICE_JOBS MATCHES "^[0-9]+$" OR
   DEEPEP_DEVICE_JOBS LESS 1 OR DEEPEP_DEVICE_JOBS GREATER 64)
    message(FATAL_ERROR "DEEPEP_DEVICE_JOBS must be in [1, 64]")
endif()

# Always enter the child build. Its compiler-generated dependencies track all
# included headers, including shared/SDK headers outside an operator directory.
# Unchanged inputs leave the library timestamp unchanged (no compile/relink).
add_custom_target(device_build
    COMMAND "${CMAKE_COMMAND}" -S "${CMAKE_SOURCE_DIR}/cmake/device" -B "${DEEPEP_DEVICE_BINARY_DIR}"
        -G "${CMAKE_GENERATOR}"
        "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng"
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}"
        "-DCMAKE_SHARED_LINKER_FLAGS:STRING=${CMAKE_SHARED_LINKER_FLAGS}"
        "-DDEEPEP_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        "-DASCEND_HOME_PATH=${ASCEND_HOME_PATH}" "-DSHMEM_ROOT=${SHMEM_ROOT}"
    COMMAND "${CMAKE_COMMAND}" --build "${DEEPEP_DEVICE_BINARY_DIR}" --parallel "${DEEPEP_DEVICE_JOBS}"
    BYPRODUCTS "${DEEPEP_DEVICE_LIBRARY}"
    VERBATIM)
add_library(deepep_device SHARED IMPORTED)
set_target_properties(deepep_device PROPERTIES IMPORTED_LOCATION "${DEEPEP_DEVICE_LIBRARY}")
add_dependencies(deepep_device device_build)
install(FILES "${DEEPEP_DEVICE_LIBRARY}" DESTINATION .)
