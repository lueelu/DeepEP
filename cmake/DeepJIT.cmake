# Copyright (c) 2026, Lu Lu

# Locked DeepJIT build headers and the two Host-standard-library adaptations.
#
# DeepJIT is a pinned third-party checkout. We never edit it in place: a build
# output compatibility directory supplies the missing preprocessor surface.
# Both adaptations are verified by compiling the feature that will actually be
# used, not by comparing compiler version strings.
set(DEEPJIT_ROOT "${CMAKE_SOURCE_DIR}/third-party/deep_jit" CACHE PATH
    "Locked DeepJIT checkout that provides include/deep_jit")

if(NOT IS_DIRECTORY "${DEEPJIT_ROOT}/include/deep_jit")
    message(FATAL_ERROR
        "DeepJIT requires the locked headers at ${DEEPJIT_ROOT}/include/deep_jit. "
        "An sdist carries them; a checkout must initialize third-party/deep_jit.")
endif()
if(NOT EXISTS "${DEEPJIT_ROOT}/README.md")
    message(FATAL_ERROR "Locked DeepJIT checkout is incomplete: missing ${DEEPJIT_ROOT}/README.md")
endif()

# Content identity of the actual headers, used by the Host disk-cache key so a
# changed DeepJIT revision invalidates old artifacts even with one compiler.
file(GLOB_RECURSE DEEPEP_DEEPJIT_HEADERS CONFIGURE_DEPENDS "${DEEPJIT_ROOT}/include/deep_jit/*.hpp")
list(SORT DEEPEP_DEEPJIT_HEADERS)
if(NOT DEEPEP_DEEPJIT_HEADERS)
    message(FATAL_ERROR "Locked DeepJIT checkout has no headers under ${DEEPJIT_ROOT}/include/deep_jit")
endif()
set(_deepep_deepjit_text "")
foreach(_header IN LISTS DEEPEP_DEEPJIT_HEADERS)
    file(READ "${_header}" _content)
    string(APPEND _deepep_deepjit_text "${_content}")
endforeach()
string(SHA256 DEEPEP_DEEPJIT_SIGNATURE "${_deepep_deepjit_text}")
message(STATUS "Locked DeepJIT header identity: ${DEEPEP_DEEPJIT_SIGNATURE}")

# DeepJIT headers use std::format. The adaptation is selected from an actual
# C++20 compile check, not from CMAKE_CXX_COMPILER_VERSION.
include(CheckCXXSourceCompiles)
set(_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
set(_saved_required_definitions "${CMAKE_REQUIRED_DEFINITIONS}")
set(_saved_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
set(CMAKE_REQUIRED_FLAGS "-std=c++20")
check_cxx_source_compiles([=[
#include <format>
#include <string>
int main() {
    const std::string value = std::format("{:016x} {}", 7ull, "ok");
    return value.empty() ? 1 : 0;
}
]=] DEEPEP_DEEPJIT_HAS_STD_FORMAT)
set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")

set(DEEPEP_DEEPJIT_COMPAT_INCLUDE "" CACHE INTERNAL "Build-output DeepJIT compatibility headers")
set(DEEPEP_DEEPJIT_FMT_INCLUDE "" CACHE INTERNAL "fmt headers used by the GCC 11 adaptation")
set(DEEPEP_DEEPJIT_COMPAT_MODE "native" CACHE INTERNAL "How DeepJIT formatting was adapted")
set(DEEPEP_FMT_SIGNATURE "")
set(DEEPEP_FMT_LICENSE_SHA256 "")
set(_deepep_compat_dir "${CMAKE_BINARY_DIR}/generated/deepjit-compat")

if(NOT DEEPEP_DEEPJIT_HAS_STD_FORMAT)
    # GCC 11's libstdc++ has no <format>. Provide a generated <format> header
    # that forwards the used surface to fmt, then compile-check the shim.
    set(DEEPEP_DEEPJIT_FMT_ROOT "" CACHE PATH
        "Directory that contains fmt/format.h for the GCC 11 DeepJIT adaptation")
    find_path(DEEPEP_FMT_INCLUDE_DIR NAMES fmt/format.h
        HINTS "${DEEPEP_DEEPJIT_FMT_ROOT}" "${CMAKE_SOURCE_DIR}/third-party/fmt"
        PATH_SUFFIXES include)
    if(NOT DEEPEP_FMT_INCLUDE_DIR)
        message(FATAL_ERROR
            "The selected Host standard library has no usable <format>. For GCC 11 provide fmt headers with "
            "-DDEEPEP_DEEPJIT_FMT_ROOT=<directory containing fmt/format.h>; "
            "this build will not silently fall back to a version check.")
    endif()
    file(MAKE_DIRECTORY "${_deepep_compat_dir}")
    file(WRITE "${_deepep_compat_dir}/format" [=[
#pragma once
// Build-output adaptation for a standard library without <format>.
// The locked DeepJIT headers keep using std::format; fmt provides the same
// formatting surface for the subset they use. The consuming target defines
// FMT_HEADER_ONLY so no separate fmt library is required.
#include <fmt/format.h>
#include <string>
#include <utility>
namespace std {
using fmt::format;
using fmt::format_error;
}  // namespace std
]=])
    # A header-only fmt build must also be used by the probe and the actual
    # _C target; including fmt/format.h without it leaves vformat undefined.
    file(GLOB_RECURSE _fmt_headers CONFIGURE_DEPENDS "${DEEPEP_FMT_INCLUDE_DIR}/fmt/*.h")
    list(SORT _fmt_headers)
    set(_fmt_text "")
    foreach(_header IN LISTS _fmt_headers)
        file(READ "${_header}" _content)
        string(APPEND _fmt_text "${_content}")
    endforeach()
    string(SHA256 DEEPEP_FMT_SIGNATURE "${_fmt_text}")
    foreach(_license IN ITEMS "${DEEPEP_FMT_INCLUDE_DIR}/LICENSE"
                              "${DEEPEP_FMT_INCLUDE_DIR}/LICENSE.rst"
                              "${DEEPEP_FMT_INCLUDE_DIR}/LICENSE.md")
        if(EXISTS "${_license}" AND DEEPEP_FMT_LICENSE_SHA256 STREQUAL "")
            file(SHA256 _license_digest "${_license}")
            set(DEEPEP_FMT_LICENSE_SHA256 "${_license_digest}")
        endif()
    endforeach()
    set(DEEPEP_DEEPJIT_COMPAT_INCLUDE "${_deepep_compat_dir}")
    set(DEEPEP_DEEPJIT_FMT_INCLUDE "${DEEPEP_FMT_INCLUDE_DIR}")
    set(DEEPEP_DEEPJIT_COMPAT_MODE "fmt")
    set(CMAKE_REQUIRED_INCLUDES "${_deepep_compat_dir}" "${DEEPEP_FMT_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_DEFINITIONS "-DFMT_HEADER_ONLY")
    set(CMAKE_REQUIRED_FLAGS "-std=c++20")
    check_cxx_source_compiles([=[
#include <format>
#include <string>
int main() {
    const std::string value = std::format("{:016x} {:.2f} {}", 7ull, 1.5, "ok");
    return value.empty() ? 1 : 0;
}
]=] DEEPEP_DEEPJIT_FORMAT_SHIM_COMPILES)
    unset(CMAKE_REQUIRED_INCLUDES)
    set(CMAKE_REQUIRED_DEFINITIONS "${_saved_required_definitions}")
    set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")
    if(NOT DEEPEP_DEEPJIT_FORMAT_SHIM_COMPILES)
        message(FATAL_ERROR
            "The generated fmt adaptation for <format> does not compile/link with FMT_HEADER_ONLY; "
            "check DEEPEP_DEEPJIT_FMT_ROOT (a C++20-compatible fmt release is required)")
    endif()
    message(STATUS "DeepJIT formatting adapted to fmt at ${DEEPEP_FMT_INCLUDE_DIR}")
else()
    message(STATUS "DeepJIT formatting uses the standard library <format>")
endif()

# Compile the locked headers' actual formatting expressions, not a synthetic
# printf. This also catches a fmt mode that forgot FMT_HEADER_ONLY.
set(CMAKE_REQUIRED_INCLUDES "${DEEPJIT_ROOT}/include")
if(NOT DEEPEP_DEEPJIT_COMPAT_INCLUDE STREQUAL "")
    list(APPEND CMAKE_REQUIRED_INCLUDES "${DEEPEP_DEEPJIT_COMPAT_INCLUDE}")
endif()
if(NOT DEEPEP_DEEPJIT_FMT_INCLUDE STREQUAL "")
    list(APPEND CMAKE_REQUIRED_INCLUDES "${DEEPEP_DEEPJIT_FMT_INCLUDE}")
    set(CMAKE_REQUIRED_DEFINITIONS "-DFMT_HEADER_ONLY")
endif()
set(CMAKE_REQUIRED_LIBRARIES "pthread")
set(CMAKE_REQUIRED_FLAGS "-std=c++20")
check_cxx_source_compiles([=[
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/uuid.hpp>
int main() {
    const auto digest = deep_jit::hash::FNV1a().get_hex_digest();
    const auto uuid = deep_jit::get_uuid();
    return digest.empty() || uuid.empty();
}
]=] DEEPEP_DEEPJIT_ACTUAL_FORMAT_COMPILES)
unset(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_REQUIRED_DEFINITIONS "${_saved_required_definitions}")
set(CMAKE_REQUIRED_LIBRARIES "${_saved_required_libraries}")
set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")
if(NOT DEEPEP_DEEPJIT_ACTUAL_FORMAT_COMPILES)
    message(FATAL_ERROR
        "The locked DeepJIT headers do not compile/link their actual std::format expressions; "
        "check the <format> adaptation and FMT_HEADER_ONLY configuration")
endif()

# The exception helper optionally enriches diagnostics with libdwfl. Preserve
# that diagnostic path when the development headers exist, but do not make the
# wheel depend on them. The stub only declares the dlsym'ed surface; libdw is
# still dlopen'ed at runtime and a missing library simply disables the trace.
include(CheckIncludeFileCXX)
check_include_file_cxx("elfutils/libdwfl.h" DEEPEP_DEEPJIT_HAS_LIBDWFL)
if(NOT DEEPEP_DEEPJIT_HAS_LIBDWFL)
    file(MAKE_DIRECTORY "${_deepep_compat_dir}/elfutils")
    file(WRITE "${_deepep_compat_dir}/elfutils/libdwfl.h" [=[
#pragma once
// Build-output declaration stub for libdwfl development headers. DeepJIT
// resolves these entry points with dlsym at runtime, so no linking is needed.
#include <cstdint>
#include <sys/types.h>
extern "C" {
typedef struct Dwfl Dwfl;
typedef struct Dwfl_Module Dwfl_Module;
typedef struct Dwfl_Line Dwfl_Line;
typedef uint64_t Dwarf_Addr;
typedef uint32_t GElf_Word;
typedef int (*Dwfl_FindElf)(Dwfl_Module*, void**, const char*, Dwarf_Addr, char**);
typedef int (*Dwfl_FindDebuginfo)(Dwfl_Module*, void**, const char*, Dwarf_Addr, const char*, const char*, GElf_Word,
                                  char**);
struct Dwfl_Callbacks {
    Dwfl_FindElf find_elf;
    Dwfl_FindDebuginfo find_debuginfo;
    void* section_address;
    char* debuginfo_path;
};
Dwfl* dwfl_begin(const Dwfl_Callbacks*);
int dwfl_linux_proc_report(Dwfl*, pid_t);
int dwfl_report_end(Dwfl*, int (*)(Dwfl_Module*, void**, const char*, Dwarf_Addr, const char*, void*), void*);
Dwfl_Module* dwfl_addrmodule(Dwfl*, Dwarf_Addr);
const char* dwfl_module_addrname(Dwfl_Module*, Dwarf_Addr);
Dwfl_Line* dwfl_module_getsrc(Dwfl_Module*, Dwarf_Addr);
const char* dwfl_lineinfo(Dwfl_Line*, Dwarf_Addr*, int*, int*, const char**, int*);
int dwfl_linux_proc_find_elf(Dwfl_Module*, void**, const char*, Dwarf_Addr, char**);
}
]=])
    if(DEEPEP_DEEPJIT_COMPAT_INCLUDE STREQUAL "")
        set(DEEPEP_DEEPJIT_COMPAT_INCLUDE "${_deepep_compat_dir}")
    endif()
    message(STATUS "DeepJIT backtrace diagnostics use the build-output libdwfl declarations")
else()
    message(STATUS "DeepJIT backtrace diagnostics use the installed libdwfl headers")
endif()

# Apply the locked headers and verified adaptations to one Host target.
function(deepep_apply_deepjit target)
    target_include_directories(${target} SYSTEM PRIVATE "${DEEPJIT_ROOT}/include")
    if(NOT DEEPEP_DEEPJIT_COMPAT_INCLUDE STREQUAL "")
        target_include_directories(${target} BEFORE PRIVATE "${DEEPEP_DEEPJIT_COMPAT_INCLUDE}")
    endif()
    if(NOT DEEPEP_DEEPJIT_FMT_INCLUDE STREQUAL "")
        target_include_directories(${target} SYSTEM PRIVATE "${DEEPEP_DEEPJIT_FMT_INCLUDE}")
        # The generated <format> shim includes fmt/format.h, whose non-header
        # entry points would otherwise need a separate fmt link library.
        target_compile_definitions(${target} PRIVATE FMT_HEADER_ONLY=1)
    endif()
endfunction()

# Deployment metadata carries identities, never build-machine paths.
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/deepjit-info.json" CONTENT
"{\n  \"deepjit_signature\": \"${DEEPEP_DEEPJIT_SIGNATURE}\",\n  \"format_compat\": \"${DEEPEP_DEEPJIT_COMPAT_MODE}\",\n  \"fmt_signature\": \"${DEEPEP_FMT_SIGNATURE}\",\n  \"fmt_license_sha256\": \"${DEEPEP_FMT_LICENSE_SHA256}\",\n  \"libdwfl_headers\": \"${DEEPEP_DEEPJIT_HAS_LIBDWFL}\"\n}\n")
