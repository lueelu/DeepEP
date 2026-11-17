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

# Configure with the parent, rather than before every child build. Forwarded
# options therefore change together, while a no-op build skips configuration.
# The child build system still regenerates itself when its CMake inputs change.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${SHMEM_ROOT}/deepep-sdk.json")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${CMAKE_SOURCE_DIR}/cmake/device" -B "${DEEPEP_DEVICE_BINARY_DIR}"
        -G "${CMAKE_GENERATOR}"
        "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng"
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}"
        "-DCMAKE_SHARED_LINKER_FLAGS:STRING=${CMAKE_SHARED_LINKER_FLAGS}"
        "-DDEEPEP_LL_STAGE_TIMER=${DEEPEP_LL_STAGE_TIMER}"
        "-DDEEPEP_LL_REFERENCE_PROFILE=${DEEPEP_LL_REFERENCE_PROFILE}"
        "-DDEEPEP_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        "-DASCEND_HOME_PATH=${ASCEND_HOME_PATH}" "-DSHMEM_ROOT=${SHMEM_ROOT}"
    RESULT_VARIABLE _deepep_device_configure_result)
if(NOT _deepep_device_configure_result STREQUAL "0")
    message(FATAL_ERROR "Device CMake configuration failed: ${_deepep_device_configure_result}")
endif()

# Always enter the child build: compiler-generated dependencies track shared
# and SDK headers as well as operator sources. Do not replace this with a stamp
# depending only on the .cpp files; that would silently reuse stale kernels.
add_custom_target(device_build
    COMMAND "${CMAKE_COMMAND}" --build "${DEEPEP_DEVICE_BINARY_DIR}" --parallel "${DEEPEP_DEVICE_JOBS}"
    BYPRODUCTS "${DEEPEP_DEVICE_LIBRARY}" "${DEEPEP_DEVICE_BINARY_DIR}/libdeepep_clos_kernels.so"
    VERBATIM)
add_library(deepep_device SHARED IMPORTED)
set_target_properties(deepep_device PROPERTIES IMPORTED_LOCATION "${DEEPEP_DEVICE_LIBRARY}")
add_dependencies(deepep_device device_build)
install(FILES "${DEEPEP_DEVICE_LIBRARY}" DESTINATION .)
add_library(deepep_clos_device SHARED IMPORTED)
set_target_properties(deepep_clos_device PROPERTIES
    IMPORTED_LOCATION "${DEEPEP_DEVICE_BINARY_DIR}/libdeepep_clos_kernels.so")
add_dependencies(deepep_clos_device device_build)
install(FILES "${DEEPEP_DEVICE_BINARY_DIR}/libdeepep_clos_kernels.so" DESTINATION .)
