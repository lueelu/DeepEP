# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

# Resolve the Tensor API used by the stage and epilogue adapters. These
# headers may be installed in different CANN operator packages.
function(ascend_deepep_find_mega_moe_headers)
    set(_common_candidates
        "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common"
        "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common"
        "${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common"
    )
    # Keep the common package root for both flat and nested Tensor API installs.
    # The target also includes common/tensor_api and common/tensor_api/include.
    # Search each package before trying the next: find_path otherwise searches
    # every package for the first name before considering the nested layout.
    foreach(_common_candidate IN LISTS _common_candidates)
        find_path(ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR
            NAMES tensor_api/tensor.h tensor_api/include/tensor_api/tensor.h
            HINTS "${_common_candidate}"
            DOC "MegaMoE common root containing the Tensor API package (flat or nested include layout)"
            NO_DEFAULT_PATH
        )
        if(ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR)
            break()
        endif()
    endforeach()
    # find_path trusts pre-existing cache entries: validate overrides as well.
    if(NOT EXISTS "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}/tensor_api/tensor.h"
       AND NOT EXISTS "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}/tensor_api/include/tensor_api/tensor.h")
        message(FATAL_ERROR
            "MegaMoE Tensor API header tensor_api/tensor.h or "
            "tensor_api/include/tensor_api/tensor.h was not found. "
            "Set ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR to the common package root "
            "from the active CANN installation (${ASCEND_HOME_PATH}). "
            "Selected root: ${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}")
    endif()

    message(STATUS "MegaMoE Tensor API include: ${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}")
    # Also propagate non-cache overrides supplied by a parent scope.
    set(ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}" PARENT_SCOPE)
endfunction()
