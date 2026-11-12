// Copyright (c) 2026, Lu Lu
#include "kernel_operator.h"
#include "shmem.h"
#include "../launch.hpp"
#include "combine_entry.h"

extern "C" __attribute__((visibility("default"))) void ubx_combine_launch(void* stream,
                                                                          const deepep::ubx::CombineArgs* args)
{
    const auto& a = *args;
    ubx_combine_route_plan<<<a.cores, 0, stream>>>(a.comm, a.expert, a.destination, a.counts, a.weights, a.types,
                                                   a.relay, a.mask, a.output, a.tiling);
}
