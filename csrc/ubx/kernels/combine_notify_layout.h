// Copyright (c) 2026, Lu Lu

#pragma once
#include <cstdint>
#include "ll_core_config.h"
#include "ep_memory_server_dedup_protocol.h"
#include "ep_memory_server_dedup_buffer_layout.h"

namespace UbxCombine {
// Each consumer owns a separate row; no reader clears another reader's flags.
constexpr uint32_t kCoreReadyPeerWords = UbxDispatch::kLlMaxCoreNum;
constexpr uint32_t kCoreReadyRowBytes = UbxDispatch::kLlRanksPerGroup * kCoreReadyPeerWords * 4U;
static_assert(COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CORE_READY_OFFSET +
                      UbxDispatch::kLlMaxRecvCoreNum * kCoreReadyRowBytes <
                  1024U * 1024U,
              "core-ready table must remain below the payload boundary");
}  // namespace UbxCombine
