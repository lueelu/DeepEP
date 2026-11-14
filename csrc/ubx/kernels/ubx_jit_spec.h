// Copyright (c) 2026, Lu Lu

// Launch identity axes shared by the Ahead-of-Time UBX Combine and its
// DeepJIT specializations. The JIT unit bakes rank, world size, core count and
// per-rank capacity into the translation unit; the AOT build keeps them in the
// runtime description. Addresses, route contents and window state stay dynamic.
#pragma once

#include <cstdint>

#ifdef DEEPEP_UBX_JIT_SPECIALIZED
#define DEEPEP_UBX_LAUNCH_WORLD(args) (DEEPEP_UBX_JIT_SPEC_WORLD)
#define DEEPEP_UBX_LAUNCH_CORES(core) (DEEPEP_UBX_JIT_SPEC_CORES)
#define DEEPEP_UBX_LAUNCH_RANK(args) (DEEPEP_UBX_JIT_SPEC_RANK)
#define DEEPEP_UBX_LAUNCH_CAPACITY(args) (DEEPEP_UBX_JIT_SPEC_CAPACITY)
#else
#define DEEPEP_UBX_LAUNCH_WORLD(args) (uint32_t((args)->rankSize))
#define DEEPEP_UBX_LAUNCH_CORES(core) (uint32_t((core).count))
#define DEEPEP_UBX_LAUNCH_RANK(args) (uint32_t((args)->rank))
#define DEEPEP_UBX_LAUNCH_CAPACITY(args) (0U)  // AOT keeps capacity in the tiling block.
#endif
