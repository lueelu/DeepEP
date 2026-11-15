// Copyright (c) 2026, Lu Lu

// Launch identity accessors for the LL device headers.
//
// The runtime kernel arguments stay the single source of truth. The JIT
// specialization passes the same values as entry template parameters, but the
// shared device bodies read them from the launch block so one source compiles
// for both the ordinary and SuperKernel entries.
#pragma once

#include <cstdint>

#define DEEPEP_LL_LAUNCH_WORLD(args) (uint32_t((args)->rankSize))
#define DEEPEP_LL_LAUNCH_CORES(core) (uint32_t((core).count))
#define DEEPEP_LL_LAUNCH_RANK(args) (uint32_t((args)->rank))
#define DEEPEP_LL_LAUNCH_CAPACITY(args) (0U)
