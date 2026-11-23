# Copyright (c) 2026, Lu Lu

# SHMEM !944 also affects inline Device code compiled outside the SDK build.
set(DEEPEP_SHMEM_DOORBELL_FLAGS "")
if(EXISTS "${SHMEM_ROOT}/deepep-sdk.json")
    file(READ "${SHMEM_ROOT}/deepep-sdk.json" _doorbell_identity)
    string(JSON _doorbell_backend ERROR_VARIABLE _doorbell_error
        GET "${_doorbell_identity}" build rdma_backend)
    if(NOT _doorbell_error AND _doorbell_backend STREQUAL "HNS_1825")
        string(JSON _doorbell_mode ERROR_VARIABLE _doorbell_error
            GET "${_doorbell_identity}" build hns_1825_pi_on_chip)
        if(_doorbell_error OR NOT _doorbell_mode MATCHES "^[01]$")
            message(FATAL_ERROR "HNS_1825 SDK must record hns_1825_pi_on_chip=0 or 1; rebuild/record the SDK")
        endif()
        if(DEFINED ACLSHMEMI_HNS_1825_PI_ON_CHIP AND
           NOT "${ACLSHMEMI_HNS_1825_PI_ON_CHIP}" STREQUAL "${_doorbell_mode}")
            message(FATAL_ERROR "HNS_1825 doorbell setting disagrees with the SHMEM SDK identity")
        endif()
        set(ACLSHMEMI_HNS_1825_PI_ON_CHIP "${_doorbell_mode}")
        add_compile_definitions("ACLSHMEMI_HNS_1825_PI_ON_CHIP=${_doorbell_mode}")
        set(DEEPEP_SHMEM_DOORBELL_FLAGS "-DACLSHMEMI_HNS_1825_PI_ON_CHIP=${_doorbell_mode}")
        message(STATUS "SHMEM HNS_1825 PI-on-chip: ${_doorbell_mode} (SDK identity)")
    endif()
endif()
