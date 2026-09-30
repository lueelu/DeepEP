# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

foreach(sdk ASCEND_HOME_PATH SHMEM_ROOT)
    if(NOT IS_DIRECTORY "${${sdk}}")
        message(FATAL_ERROR "${sdk} must identify an installed SDK")
    endif()
endforeach()
# Resolve from the selected SDK on every configure, including direct CMake use.
unset(ASCENDCL_LIBRARY CACHE)
unset(ASCEND_RUNTIME_LIBRARY CACHE)
unset(SHMEM_LIBRARY CACHE)
set(_cann_library_paths "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
find_library(ASCENDCL_LIBRARY NAMES ascendcl PATHS ${_cann_library_paths} NO_DEFAULT_PATH REQUIRED)
find_library(ASCEND_RUNTIME_LIBRARY NAMES runtime PATHS ${_cann_library_paths} NO_DEFAULT_PATH REQUIRED)
find_library(SHMEM_LIBRARY NAMES shmem PATHS "${SHMEM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
add_library(ascendcl SHARED IMPORTED)
set_target_properties(ascendcl PROPERTIES IMPORTED_LOCATION "${ASCENDCL_LIBRARY}")
add_library(ascend_runtime SHARED IMPORTED)
set_target_properties(ascend_runtime PROPERTIES IMPORTED_LOCATION "${ASCEND_RUNTIME_LIBRARY}")
add_library(shmem SHARED IMPORTED)
set_target_properties(shmem PROPERTIES IMPORTED_LOCATION "${SHMEM_LIBRARY}")
