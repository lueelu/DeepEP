# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "EP_GMM_FUSED requires Linux and the Ascend950 CANN toolkit")
endif()
# Keep the experiment's compiler and SDK variables in this directory scope.
# The parent runtime continues to use its host compiler and external SHMEM SDK.
include(${PROJECT_SOURCE_DIR}/cmake/ascend/ascend.cmake)
find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)
enable_language(ASC AICPU)
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
include(${PROJECT_SOURCE_DIR}/cmake/ascend/func.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/EpGmmFusedDependencies.cmake)
set(EP_GMM_FUSED_INSTALL_DIR "../deep_ep_experimental/ep_gmm_fused")
set(NPU_SOC_VERSION ascend950)
set(EXTENSION_MODULE_NAME deep_ep_ep_gmm_fused)
if(NOT DEEPEP_EXTENSION_SUFFIX)
    set(DEEPEP_EXTENSION_SUFFIX ".so")
endif()
set(INCLUDE_DIRECTORIES
    ${PROJECT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/catccos_overlay/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_INCLUDE_DIRS}
    ${CATCCOS_INCLUDE_DIR}
    ${CATCCOS_CATLASS_INCLUDE_DIR}
    ${CATCCOS_TOOLS_INCLUDE_DIR}
    ${CATCCOS_UTILS_INCLUDE_DIR}
    ${SHMEM_ROOT}/include
    ${SHMEM_SOURCE_PATH}/include
    ${SHMEM_SOURCE_PATH}/src
    ${SHMEM_SOURCE_PATH}/src/device
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/include/experiment/runtime
    ${ASCEND_HOME}/include/experiment/msprof
    ${ASCEND_HOME}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME}/${SYSTEM_PREFIX}/include
    ${ASCEND_HOME}/${SYSTEM_PREFIX}/pkg_inc/runtime
    ${ASCEND_HOME}/pkg_inc/base
)

set(LINK_DIRECTORIES
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_DIR}/lib64
    ${ASCEND_HOME}/lib64
)

set(LINK_LIBRARIES
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
    graph_base
    shmem
    dl
)

set(COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O2
    -fdiagnostics-color=always
    -Wno-macro-redefined
    -Wno-ignored-attributes
    -Wno-defaulted-function-deleted
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
    -DCATLASS_ARCH=3510
    -DL2_CACHE_HINT
    -DTORCH_MODE
)

set(OBJECTS_LIST "" CACHE INTERNAL "CatCCOS UDMA operator objects" FORCE)
add_subdirectory(csrc)

add_library(ep_gmm_fused_C SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp
    ${OBJECTS_LIST}
)
set_target_properties(ep_gmm_fused_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX "${DEEPEP_EXTENSION_SUFFIX}"
    OUTPUT_NAME "_C"
    INSTALL_RPATH ""
    INSTALL_RPATH_USE_LINK_PATH FALSE
)
target_compile_definitions(ep_gmm_fused_C PRIVATE Py_LIMITED_API=0x03080000)
target_compile_options(ep_gmm_fused_C PRIVATE ${COMPILE_OPTIONS})
target_include_directories(ep_gmm_fused_C PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(ep_gmm_fused_C PRIVATE ${LINK_DIRECTORIES})
target_link_libraries(ep_gmm_fused_C PRIVATE ${LINK_LIBRARIES})


install(TARGETS ep_gmm_fused_C LIBRARY DESTINATION ${EP_GMM_FUSED_INSTALL_DIR})
install(FILES "${SHMEM_SOURCE_PATH}/LICENSE"
    DESTINATION ${EP_GMM_FUSED_INSTALL_DIR}/licenses/shmem)
install(FILES "${CATCCOS_SOURCE_PATH}/LICENSE"
    DESTINATION ${EP_GMM_FUSED_INSTALL_DIR}/licenses/catccos)
install(FILES "${CATCCOS_SOURCE_PATH}/3rdparty/catlass/LICENSE"
    DESTINATION ${EP_GMM_FUSED_INSTALL_DIR}/licenses/catlass)
