// Copyright (c) 2026, Lu Lu
#pragma once
#include "ll_dispatch_c_api.h"
#include "../common/ep_memory_server_dedup_data_layout.h"
#include "../common/ep_memory_server_dedup_protocol.h"

namespace DispatchDedup {
__aicore__ inline uint32_t ServerDedupWireOffset(uint32_t logical)
{
    return logical + logical / 480U * 32U;
}
class ServerDedupPacketTail {
public:
    __aicore__ inline ServerDedupPacketTail(__ubuf__ uint8_t* wire, uint32_t header, uint32_t index)
        : header_(reinterpret_cast<__ubuf__ uint32_t*>(wire + header)),
          index_(reinterpret_cast<__ubuf__ uint32_t*>(wire + index)) {}
    __aicore__ inline __ubuf__ uint32_t* Header() { return header_; }
    __aicore__ inline __ubuf__ uint32_t* ReadIndex() { return index_; }
    __aicore__ inline __ubuf__ uint32_t* Entry(uint32_t index) { return index_ + (index + 1U) * 2U; }
private:
    __ubuf__ uint32_t *header_, *index_;
};
__aicore__ inline void ReadServerDedupFlags(GM_ADDR slot, uint32_t blocks, __ubuf__ float* flags,
                                           bool uncached = false)
{
    LlDispatchC::Load(flags, slot + 480U, 32U, blocks, 480U, 0U, uncached);
}
__aicore__ inline void WriteServerDedupPayloadRange(GM_ADDR output, __ubuf__ uint8_t* wire,
                                                   uint32_t logical, uint32_t bytes)
{
    uint32_t written = 0U;
    const uint32_t within = logical % 480U;
    if (within && bytes) {
        const uint32_t available = 480U - within, head = bytes < available ? bytes : available;
        LlDispatchC::Store(output, wire + ServerDedupWireOffset(logical), head);
        logical += head; bytes -= head; written += head;
    }
    const uint32_t full = bytes / 480U;
    if (full) {
        LlDispatchC::Store(output + written, wire + ServerDedupWireOffset(logical), 480U, full, 0U, 32U);
        logical += full * 480U; bytes -= full * 480U; written += full * 480U;
    }
    if (bytes) LlDispatchC::Store(output + written, wire + ServerDedupWireOffset(logical), bytes);
}
__aicore__ inline uint32_t BuildServerDedupMetadata(__ubuf__ uint32_t* type, __ubuf__ uint32_t* destination,
                                                  ServerDedupPacketTail tail, uint32_t slot)
{
    auto* header = tail.Header();
    const uint32_t role = header[3U];
    type[0] = role;
    if (role == uint32_t(EpServerDedupTokenType::Terminal)) {
        destination[0] = slot; destination[1] = UINT32_MAX; destination[2] = UINT32_MAX;
    } else {
        for (uint32_t i = 0; i < 3U; ++i) destination[i] = header[i];
    }
    return role;
}
__aicore__ inline void PrepareServerDedupForward(ServerDedupPacketTail tail)
{
    tail.Header()[3U] = uint32_t(EpServerDedupTokenType::Terminal);
}
}  // namespace DispatchDedup
