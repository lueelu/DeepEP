# Header-only communication building blocks. Consumers select their CANN/SHMEM
# includes, backend definitions and device compilation flags explicitly.
add_library(deepep_comm INTERFACE)
target_compile_features(deepep_comm INTERFACE cxx_std_17)
target_include_directories(deepep_comm INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/.." "${CATLASS_SOURCE_PATH}/include")
