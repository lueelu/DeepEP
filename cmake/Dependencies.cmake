# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

# Shared by direct Host and Device CMake entry points. This branch has one
# compiled capability; backend selection happens at runtime.
string(TOUPPER "${DEEPEP_LL_RDMA}" _legacy_ll_rdma)
if(DEFINED DEEPEP_LL_RDMA AND NOT _legacy_ll_rdma MATCHES "^(ON|1|TRUE)$")
    message(WARNING "DEEPEP_LL_RDMA is deprecated and ignored: NDR always compiles HNS_1825 support. Use --backend mte at runtime or kernel/ll-configurable-groups for a build without RDMA.")
endif()
set(DEEPEP_LL_RDMA ON CACHE INTERNAL "Fixed NDR capability" FORCE)

foreach(sdk ASCEND_HOME_PATH SHMEM_ROOT)
    if(NOT IS_DIRECTORY "${${sdk}}")
        message(FATAL_ERROR "${sdk} must identify an installed SDK")
    endif()
endforeach()
set(_sdk_manifest "${SHMEM_ROOT}/deepep-sdk.json")
if(NOT EXISTS "${_sdk_manifest}")
    message(FATAL_ERROR "Require an identified native SHMEM SDK")
endif()
file(READ "${_sdk_manifest}" _sdk_identity)
string(JSON _field_count ERROR_VARIABLE _fields_error LENGTH "${_sdk_identity}" build)
string(JSON _patches_type ERROR_VARIABLE _type_error TYPE "${_sdk_identity}" build source_patches)
string(JSON _patches_count ERROR_VARIABLE _count_error LENGTH "${_sdk_identity}" build source_patches)
string(JSON _target ERROR_VARIABLE _target_error GET "${_sdk_identity}" build target)
if(_fields_error OR _type_error OR _count_error OR _target_error
   OR NOT _patches_type STREQUAL "ARRAY" OR NOT _target STREQUAL "Ascend950"
   OR NOT _field_count EQUAL 6)
    message(FATAL_ERROR "Require a current native SHMEM SDK identity; stop SDK users and rebuild explicitly")
endif()
foreach(_identity_key cann_version cann_timestamp target)
    string(JSON _identity_type ERROR_VARIABLE _identity_error TYPE "${_sdk_identity}" build ${_identity_key})
    string(JSON _identity_value ERROR_VARIABLE _value_error GET "${_sdk_identity}" build ${_identity_key})
    if(_identity_error OR _value_error OR NOT _identity_type STREQUAL "STRING" OR _identity_value STREQUAL "")
        message(FATAL_ERROR "Incomplete native SHMEM SDK build identity")
    endif()
endforeach()
string(JSON _rdma_backend ERROR_VARIABLE _rdma_error GET "${_sdk_identity}" build rdma_backend)
string(JSON _hns_mode_type ERROR_VARIABLE _hns_type_error TYPE "${_sdk_identity}" build hns_1825_pi_on_chip)
string(JSON DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP ERROR_VARIABLE _hns_mode_error GET "${_sdk_identity}" build hns_1825_pi_on_chip)
if(_rdma_error OR NOT _rdma_backend STREQUAL "HNS_1825"
   OR _hns_type_error OR _hns_mode_error OR NOT _hns_mode_type STREQUAL "NUMBER"
   OR NOT DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP MATCHES "^[01]$")
    message(FATAL_ERROR "NDR requires the native HNS_1825 SDK and its recorded doorbell mode; rebuild the SDK explicitly")
endif()
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
