// CPU execution tests address/byte ownership, not device ordering or UDMA completion.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>
#define __aicore__
#define __gm__
#define ASCEND_DEEPEP_ARCH 3510
#define GM_ADDR uint8_t*
#include "common/types.h"
#include "common/platform.h"
using ascend_deepep::CombineTilingData;
namespace platform = ascend_deepep::platform;
constexpr int EVENT_ID0 = 0, EVENT_ID1 = 1, EVENT_ID2 = 2;
constexpr uint32_t kBatchWqes = 32, kWqeBytes = 64;
namespace AscendC {
enum class HardEvent { MTE3_MTE2, MTE2_MTE3, MTE3_S };
template <HardEvent> void SetFlag(int) {}
template <HardEvent> void WaitFlag(int) {}
template <typename T> struct GlobalTensor {
    T* data{};
    void SetGlobalBuffer(T* p) { data = p; }
    T GetValue(uint64_t i) const { return data[i]; }
    GlobalTensor operator[](uint64_t i) const { return {data + i}; }
};
template <typename T> struct LocalTensor : GlobalTensor<T> {
    LocalTensor operator[](uint64_t i) const { return {{this->data + i}}; }
};
struct DataCopyExtParams { uint32_t blocks, bytes, src_gap, dst_gap, reserved; };
template <typename T> struct DataCopyPadExtParams { bool pad; uint8_t left, right; T value; };
template <typename T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams p,
                                     DataCopyPadExtParams<T>) {
    assert(reinterpret_cast<uintptr_t>(dst.data) % 32 == 0);
    std::memcpy(dst.data, src.data, p.bytes);
}
template <typename T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p) {
    assert(reinterpret_cast<uintptr_t>(src.data) % 32 == 0);
    std::memcpy(dst.data, src.data, p.bytes);
}
}
struct Resource {
    alignas(32) uint8_t bytes[ascend_deepep::platform::kKernelUbBytes]{};
    template <typename T> AscendC::LocalTensor<T> GetBufferByByte(uint32_t n) {
        assert(n < sizeof(bytes)); return {{reinterpret_cast<T*>(bytes + n)}};
    }
};
// PRODUCTION_MAPPING
// PRODUCTION_COPY
struct Base {
    static inline uint64_t writes = 0;
    static uint32_t QueuePeer(uint32_t p) { return p; }
    static void PutAbsolute(uint32_t, uint64_t dst, uint8_t* src, uint32_t bytes) {
        assert(bytes && bytes <= 65536);
        std::memcpy(reinterpret_cast<void*>(dst), src, bytes);
        ++writes;
    }
    static void Flush() {}
};
struct Sender : Base {
    const CombineTilingData& td_;
    uint64_t weights_;
    struct Peer { uint64_t remote_base; } peers_[1];
    void Flush() {}
    // PRODUCTION_SENDER
};

int main() {
    // Execute the host's actual default/override gate, including legacy callers
    // and paths whose data cannot be sent before aggregate completion.
    for (const char* compact_weights : {static_cast<const char*>(nullptr), "0", "1"})
    for (unsigned excluded = 0; excluded < 9; ++excluded) {
        CombineTilingData tiling{};
        std::optional<int> send_channel_prefix_matrix = 1, topk_weights = 1, expanded_x;
        bool use_udma = true, epilogue = true, aggregate = false;
        unsigned pipeline_chunk_tokens = 0;
        if (excluded == 1) send_channel_prefix_matrix.reset();
        if (excluded == 2) topk_weights.reset();
        if (excluded == 3) use_udma = false;
        if (excluded == 4) epilogue = false;
        if (excluded == 5) tiling.udma_token_schedule = 1;
        if (excluded == 6) aggregate = true;
        if (excluded == 7) pipeline_chunk_tokens = 32;
        if (excluded == 8) expanded_x = 1;
        // PRODUCTION_GATE
        assert(bool(tiling.combine_compact_weights) ==
               (excluded == 0 && (!compact_weights || std::strcmp(compact_weights, "1") == 0)));
    }
    for (uint32_t world : {4U, 8U, 16U, 32U})
    for (uint32_t topk : {1U, 6U, 8U, 10U, 128U})
    for (uint32_t tokens : {3U, 65U, 2051U})
    for (uint32_t channels : {8U, 32U}) {
        const uint32_t channel_tokens = (tokens + channels - 1) / channels;
        const uint32_t capacity = tokens + 7; // Reservation may exceed logical tokens.
        std::vector<int32_t> heads(tokens * world, -1), prefixes(world * channels, 0);
        std::vector<std::vector<uint32_t>> routes(world);
        for (uint32_t peer = 0; peer < world; ++peer) {
            uint32_t count = 0;
            for (uint32_t ch = 0; ch < channels; ++ch) {
                uint32_t ordinal = 0;
                for (uint32_t token = ch * channel_tokens; token < std::min(tokens, (ch + 1) * channel_tokens); ++token) {
                    // Empty contributor, masked routes, uneven counts and tails.
                    if (peer == world - 1 || (token * 7 + peer * 11) % 5 == 0) continue;
                    heads[token * world + peer] = ordinal++;
                    routes[peer].push_back(token);
                }
                count += ordinal;
                prefixes[peer * channels + ch] = count;
            }
        }
        for (uint32_t workers : {1U, 2U, 4U}) {
            const uint32_t row_bytes = topk * sizeof(float);
            std::vector<float> received(world * capacity * topk + 16, -99.0F);
            std::vector<float> golden(tokens * topk, 0.0F);
            CombineTilingData td{};
            td.num_tokens = tokens; td.num_channels = channels;
            td.token_capacity = capacity; td.world_size = world;
            td.num_topk = topk; td.topk_weights_row_stride = row_bytes;
            td.combine_compact_weights = 1;
            for (uint32_t peer = 0; peer < world; ++peer) {
                const uint32_t base = 3, count = routes[peer].size();
                std::vector<float> input((base + count) * topk);
                for (uint32_t row = 0; row < count; ++row)
                    for (uint32_t k = 0; k < topk; ++k) {
                        const float value = float(int((routes[peer][row] * 13 + peer * 7 + k * 3) % 31) - 15) / 8.0F;
                        input[(base + row) * topk + k] = value;
                        golden[routes[peer][row] * topk + k] += value;
                    }
                td.rank = peer;
                Sender sender{Base{}, td, reinterpret_cast<uint64_t>(input.data()),
                              {{reinterpret_cast<uint64_t>(received.data())}}};
                for (uint32_t lane = 0; lane < workers; ++lane) {
                    const uint32_t begin = uint64_t(count) * lane / workers;
                    const uint32_t end = uint64_t(count) * (lane + 1) / workers;
                    sender.PutCompactWeights(0, base + begin, base, end - begin);
                }
                for (uint32_t row = count; row < capacity; ++row)
                    for (uint32_t k = 0; k < topk; ++k)
                        assert(received[(peer * capacity + row) * topk + k] == -99.0F);
            }
            AscendC::LocalTensor<int32_t> prefix{{prefixes.data()}};
            for (uint32_t reduce_cores : {8U, 32U, 48U}) {
                std::vector<float> actual(tokens * topk, 0.0F);
                const uint32_t per_core = (tokens + reduce_cores - 1) / reduce_cores;
                for (uint32_t core = 0; core < reduce_cores; ++core)
                    for (uint32_t token = std::min(tokens, core * per_core);
                         token < std::min(tokens, (core + 1) * per_core); ++token)
                        for (uint32_t peer = 0; peer < world; ++peer) {
                            const int ordinal = heads[token * world + peer];
                            if (ordinal < 0) continue;
                            const uint64_t slot = CompactWeightSlot(td, prefix, peer, token, ordinal);
                            assert(slot < uint64_t(world) * capacity);
                            for (uint32_t k = 0; k < topk; ++k) actual[token * topk + k] += received[slot * topk + k];
                        }
                assert(actual == golden);
            }
            for (uint32_t k = 0; k < 16; ++k) assert(received[world * capacity * topk + k] == -99.0F);
        }
    }
    // Execute the production continuous local copy for aligned and odd top-k.
    Resource resource;
    for (uint32_t topk : {6U, 8U, 10U}) {
        CombineTilingData td{};
        td.rank = 1; td.world_size = 4; td.token_capacity = 65; td.block_dim = 32;
        td.has_topk_weights = 1; td.x_row_stride = 2; td.topk_weights_row_stride = topk * 4;
        td.topk_weights_buffer_offset = 4 * 65 * 2;
        std::vector<int32_t> ranks(16, 0); ranks[1] = 3; ranks[5] = 68;
        std::vector<uint8_t> x(68 * 2, 7), weights(68 * topk * 4);
        for (uint32_t i = 0; i < weights.size(); ++i) weights[i] = uint8_t(i % 251);
        std::vector<uint8_t> output(td.topk_weights_buffer_offset + 4 * 65 * topk * 4, 0);
        for (uint32_t core = 0; core < td.block_dim; ++core)
            CopyCompactSelf(resource, x.data(), weights.data(), reinterpret_cast<uint8_t*>(ranks.data()), output.data(), td, core);
        assert(std::memcmp(output.data() + td.topk_weights_buffer_offset + 65 * topk * 4,
                           weights.data() + 3 * topk * 4, 65 * topk * 4) == 0);
    }
}
