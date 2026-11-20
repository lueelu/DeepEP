# Copyright (c) 2026, Lu Lu

# Older CMake versions can capture shell startup warnings while probing uname,
# leaving CMAKE_SYSTEM_PROCESSOR with embedded newlines. Packaging supplies
# platform.machine() directly; direct CMake callers may supply the same override.
if(NOT DEFINED DEEPEP_HOST_PROCESSOR)
    set(DEEPEP_HOST_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(NOT DEEPEP_HOST_PROCESSOR MATCHES "^[A-Za-z0-9_]+$")
    message(FATAL_ERROR
        "Invalid host processor for CANN SDK paths. Use the Python build entry "
        "or set -DDEEPEP_HOST_PROCESSOR to the host architecture (for example, "
        "x86_64 or aarch64). Check shell startup warnings from the build environment.")
endif()
