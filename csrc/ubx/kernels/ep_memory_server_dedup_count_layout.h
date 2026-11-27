// Copyright (c) 2026, Lu Lu
#pragma once
#include <deep_ep/layout/ep/workspace.hpp>
#include <ubx/kernels/comm_args.h>
#include <ubx/kernels/dispatch_dedup_types.h>
#include <ubx/kernels/ll_group_policy.h>

namespace UbxDispatch {
using deep_ep::kEpServerDedupControlPrefixBytes;
using deep_ep::kEpServerDedupMagicStride;
using deep_ep::kEpServerDedupPeerBytes;
using deep_ep::kEpServerDedupWindowBytes;
using deep_ep::kEpServerDedupWindowUsableBytes;
using deep_ep::DispatchServerDedupCountLayout;
using deep_ep::DispatchServerDedupWindowOffset;
}  // namespace UbxDispatch
