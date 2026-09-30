// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_DEDUP_EP_SHMEM_H
#define DISPATCH_DEDUP_EP_SHMEM_H

// Public SHMEM headers: MTE performs UB -> peer symmetric GM on PIPE_MTE3.
// Completion is tracked by the caller's existing MTE3 events before UB reuse.
// No SHMEM global barrier is inserted into per-rank, uneven packet loops.
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/ub2gm/engine/shmem_device_mte.h"

#endif
