// Copyright (c) 2026, Lu Lu

// Launch identity axes shared by the Ahead-of-Time UBX Combine and its
// DeepJIT specializations. The JIT unit bakes world size and core count into
// the translation unit; the AOT build keeps them in the runtime description.
#pragma once

#include <cstdint>

#ifdef DEEPEP_UBX_JIT_SPECIALIZED
#define DEEPEP_UBX_LAUNCH_WORLD(args) (DEEPEP_UBX_JIT_SPEC_WORLD)
#define DEEPEP_UBX_LAUNCH_CORES(core) (DEEPEP_UBX_JIT_SPEC_CORES)
#else
#define DEEPEP_UBX_LAUNCH_WORLD(args) (uint32_t((args)->rankSize))
#define DEEPEP_UBX_LAUNCH_CORES(core) (uint32_t((core).count))
#endif
