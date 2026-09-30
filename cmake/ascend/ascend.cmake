# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Modified by zhu-mingzhe71 2026

# find ascend toolkit
if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()

# The root build validates ASCEND_HOME_PATH before enabling this component.
set(ASCEND_DIR "${ASCEND_HOME_PATH}")

message(STATUS "Using Ascend toolkit path: ${ASCEND_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${ASCEND_DIR}/")
set(BISHENG "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "Bisheng compiler path: ${BISHENG}")

# set the default compiler and linker to bisheng
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

# set ASCEND_INCLUDE_DIRS
set(ASCEND_INCLUDE_DIRS
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/pkg_inc
    ${ASCEND_DIR}/include/hcomm
    ${ASCEND_DIR}/compiler/tikcpp/include
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/interface
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/tiling
    ${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api
)
