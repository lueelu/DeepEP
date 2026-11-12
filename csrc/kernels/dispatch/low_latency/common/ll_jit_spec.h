// Copyright (c) 2026, Lu Lu

// Launch identity: world size and active core count.
//
// Ahead-of-Time builds read them at run time (CommArgs.rankSize and the
// logical core context); the JIT build bakes them into the translation unit
// together with the shape, so world-derived offsets and per-core partitioning
// fold to constants. The Host adds them to the JIT cache key, so one compiled
// kernel is only ever launched with the world size and core count it was
// built for.
#pragma once

#include <cstdint>

#ifdef DEEPEP_LL_JIT_SPECIALIZED
#define DEEPEP_LL_LAUNCH_WORLD(args) (DEEPEP_LL_JIT_SPEC_WORLD)
#define DEEPEP_LL_LAUNCH_CORES(core) (DEEPEP_LL_JIT_SPEC_CORES)
#else
#define DEEPEP_LL_LAUNCH_WORLD(args) (uint32_t((args)->rankSize))
#define DEEPEP_LL_LAUNCH_CORES(core) (uint32_t((core).count))
#endif
