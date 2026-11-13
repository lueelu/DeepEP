// Copyright (c) 2026, Lu Lu

// Launch identity: rank, world size, active core count and per-rank capacity.
//
// Ahead-of-Time builds read them at run time (CommArgs fields and the logical
// core context); the JIT build bakes them into the translation unit together
// with the shape, so rank-derived offsets and per-core partitioning fold to
// constants. The Host adds them to the JIT cache key, so one compiled kernel is
// only ever launched with the rank, world size, core count and capacity it was
// built for. Addresses, payload, routing values, counters, window generation,
// SQ/CQ state and completion flags never become compile-time constants.
#pragma once

#include <cstdint>

#ifdef DEEPEP_LL_JIT_SPECIALIZED
#define DEEPEP_LL_LAUNCH_WORLD(args) (DEEPEP_LL_JIT_SPEC_WORLD)
#define DEEPEP_LL_LAUNCH_CORES(core) (DEEPEP_LL_JIT_SPEC_CORES)
#define DEEPEP_LL_LAUNCH_RANK(args) (DEEPEP_LL_JIT_SPEC_RANK)
#define DEEPEP_LL_LAUNCH_CAPACITY(args) (DEEPEP_LL_JIT_SPEC_CAPACITY)
#else
#define DEEPEP_LL_LAUNCH_WORLD(args) (uint32_t((args)->rankSize))
#define DEEPEP_LL_LAUNCH_CORES(core) (uint32_t((core).count))
#define DEEPEP_LL_LAUNCH_RANK(args) (uint32_t((args)->rank))
#define DEEPEP_LL_LAUNCH_CAPACITY(args) (0U)  // AOT keeps capacity in the tiling block.
#endif
