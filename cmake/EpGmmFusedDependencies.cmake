# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
set(TORCH_NPU_INCLUDE_PATH "${TORCH_NPU_PATH}/include")
set(TORCH_NPU_LIB_PATH "${TORCH_NPU_PATH}/lib")

set(CATCCOS_TAG_ID "878c2e0a504be8c49c03b5e1157a7c714a628e29")
if(NOT CATCCOS_SOURCE_PATH)
    if(DEFINED ENV{CATCCOS_SOURCE_PATH})
        set(CATCCOS_SOURCE_PATH "$ENV{CATCCOS_SOURCE_PATH}")
    else()
        set(CATCCOS_SOURCE_PATH "${PROJECT_SOURCE_DIR}/third-party/catccos")
    endif()
endif()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CATCCOS_SOURCE_PATH}"
    OUTPUT_VARIABLE CATCCOS_REV OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE CATCCOS_STATUS)
execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${CATCCOS_SOURCE_PATH}"
    OUTPUT_VARIABLE CATCCOS_DIRTY OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE CATCCOS_DIRTY_STATUS)
execute_process(COMMAND "${GIT_EXECUTABLE}" submodule status --recursive 3rdparty/catlass
    WORKING_DIRECTORY "${CATCCOS_SOURCE_PATH}"
    OUTPUT_VARIABLE SUBMODULE_STATUS RESULT_VARIABLE SUBMODULE_RESULT)
if(NOT CATCCOS_STATUS EQUAL 0 OR NOT CATCCOS_REV STREQUAL CATCCOS_TAG_ID OR
   NOT CATCCOS_DIRTY_STATUS EQUAL 0 OR NOT CATCCOS_DIRTY STREQUAL "" OR
   NOT SUBMODULE_RESULT EQUAL 0 OR SUBMODULE_STATUS MATCHES "(^|\n)[-+U]")
    message(FATAL_ERROR "CatCCOS and CATLASS must be clean and pinned. Run scripts/prepare_ep_gmm_fused_dependencies.sh")
endif()
set(CATCCOS_INCLUDE_DIR "${CATCCOS_SOURCE_PATH}/include")
set(CATCCOS_CATLASS_INCLUDE_DIR "${CATCCOS_SOURCE_PATH}/3rdparty/catlass/include")
set(CATCCOS_TOOLS_INCLUDE_DIR "${CATCCOS_SOURCE_PATH}/tools")
set(CATCCOS_UTILS_INCLUDE_DIR "${CATCCOS_SOURCE_PATH}/utils")
if(NOT EXISTS "${CATCCOS_CATLASS_INCLUDE_DIR}/catlass/catlass.hpp")
    message(FATAL_ERROR "CatCCOS CATLASS submodule is missing")
endif()

set(ASCEND_HOME "${ASCEND_HOME_PATH}")
# Device implementation headers must match the core runtime's external SDK.
file(READ "${PROJECT_SOURCE_DIR}/dependencies.lock.json" DEEPEP_DEPENDENCY_LOCK)
string(JSON SHMEM_TAG_ID GET "${DEEPEP_DEPENDENCY_LOCK}" shmem revision)
if(SHMEM_SOURCE_PATH)
    get_filename_component(SHMEM_SOURCE_PATH "${SHMEM_SOURCE_PATH}" REALPATH)
elseif(DEFINED ENV{SHMEM_SOURCE_PATH} AND IS_DIRECTORY "$ENV{SHMEM_SOURCE_PATH}")
    get_filename_component(SHMEM_SOURCE_PATH "$ENV{SHMEM_SOURCE_PATH}" REALPATH)
elseif(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/third-party/shmem")
    get_filename_component(SHMEM_SOURCE_PATH "${PROJECT_SOURCE_DIR}/third-party/shmem" REALPATH)
else()
    message(FATAL_ERROR
        "The SHMEM checkout matching dependencies.lock.json was not found. "
        "Run scripts/prepare_ep_gmm_fused_dependencies.sh first.")
endif()
if(NOT EXISTS "${SHMEM_SOURCE_PATH}/include/shmem.h")
    message(FATAL_ERROR "Invalid SHMEM_SOURCE_PATH=${SHMEM_SOURCE_PATH}: include/shmem.h was not found")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${SHMEM_SOURCE_PATH}"
    OUTPUT_VARIABLE SHMEM_ACTUAL_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SHMEM_REVISION_STATUS
)
if(NOT SHMEM_REVISION_STATUS EQUAL 0 OR NOT SHMEM_ACTUAL_REVISION STREQUAL SHMEM_TAG_ID)
    message(FATAL_ERROR
        "SHMEM revision mismatch: expected locked revision ${SHMEM_TAG_ID}, "
        "got ${SHMEM_ACTUAL_REVISION}")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${SHMEM_SOURCE_PATH}"
    OUTPUT_VARIABLE SHMEM_WORKTREE_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SHMEM_WORKTREE_CHECK_STATUS
)
if(NOT SHMEM_WORKTREE_CHECK_STATUS EQUAL 0 OR NOT SHMEM_WORKTREE_STATUS STREQUAL "")
    message(FATAL_ERROR
        "SHMEM source has local changes or cannot be inspected; its headers must match the pinned library. "
        "${SHMEM_WORKTREE_STATUS}")
endif()
