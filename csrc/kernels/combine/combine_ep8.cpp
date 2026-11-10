// Copyright (c) 2026, Lu Lu

// Compile the EP8 specialization separately to preserve the default scheduler.
#define DEEPEP_COMBINE_EP8 1
#define combine_kernel combine_ep8_kernel
#define combine_kernel_do combine_ep8_kernel_do
#define combine_profile_kernel combine_ep8_profile_kernel
#define combine_profile_kernel_do combine_ep8_profile_kernel_do
#include "combine.cpp"
