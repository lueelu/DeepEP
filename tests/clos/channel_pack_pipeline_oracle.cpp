// Appended to the production packing bodies by test_channel_metadata.py.
// CPU DMA is immediate: this checks addressing and event credits, not device ordering.
void CheckPack(unsigned world, unsigned tokens, unsigned channels, unsigned k, bool weighted)
{
    DispatchTilingData td{};
    td.num_tokens = tokens;
    td.num_channels = channels;
    td.world_size = world;
    td.num_experts = world * 16;
    td.num_topk = k;
    td.has_topk_weights = weighted;
    td.channel_metadata_tile_tokens = 32;
    const auto tile = MakeChannelMetadataTile(32, world, k, sizeof(topk_idx_t), weighted);
    const unsigned length = (tokens + channels - 1) / channels;
    const unsigned stride = (length + 7) / 8 * 8;
    const unsigned size = world * channels * stride * k;
    std::vector<int32_t> head(tokens * world, -1);
    std::vector<topk_idx_t> idx(tokens * k), expected(size, -77), output(size + 2, -77);
    std::vector<float> weights(tokens * k), expected_w(size, -77), output_w(size + 2, -77);
    for (unsigned t = 0; t < tokens; ++t)
        for (unsigned j = 0; j < k; ++j) {
            idx[t * k + j] = t % 7 == 0 || j == 3 ? -1 : topk_idx_t((t * 13 + j * 5) % td.num_experts);
            weights[t * k + j] = float((t * k + j) % 61) / 64;
        }
    for (unsigned c = 0; c < channels; ++c)
        for (unsigned p = 0; p < world; ++p) {
            unsigned ordinal = 0;
            for (unsigned t = c * length; t < std::min(tokens, (c + 1) * length); ++t) {
                bool hit = false;
                for (unsigned j = 0; j < k; ++j)
                    hit |= idx[t * k + j] >= topk_idx_t(p * 16) && idx[t * k + j] < topk_idx_t((p + 1) * 16);
                if (!hit) continue;
                head[t * world + p] = ordinal;
                for (unsigned j = 0; j < k; ++j) {
                    const bool valid = idx[t * k + j] >= topk_idx_t(p * 16) && idx[t * k + j] < topk_idx_t((p + 1) * 16);
                    const unsigned out = ((p * channels + c) * stride + ordinal) * k + j;
                    expected[out] = valid ? idx[t * k + j]-p * 16 : topk_idx_t(-1);
                    if (weighted) expected_w[out] = valid ? weights[t * k + j] : 0;
                }
                ++ordinal;
            }
        }
    td.notify_topk_idx = reinterpret_cast<uint64_t>(idx.data());
    td.notify_topk_weights = reinterpret_cast<uint64_t>(weights.data());
    td.packed_channel_indices = reinterpret_cast<uint64_t>(output.data() + 1);
    td.packed_channel_weights = reinterpret_cast<uint64_t>(output_w.data() + 1);
    for (unsigned db : {0U, 1U}) {
        Resource resource;
        std::fill(resource.ub.begin(), resource.ub.end(), 0xababababababababULL);
        std::fill(output.begin(), output.end(), -77);
        std::fill(output_w.begin(), output_w.end(), -77);
        td.channel_metadata_double_buffer = db;
        for (unsigned c = 0; c < channels; ++c) {
            const auto sets_before = AscendC::event_sets;
            PackChannelMetadata(resource, reinterpret_cast<GM_ADDR>(head.data()), td, c);
            for (auto& events : AscendC::credits) {
                for (int credit : events) assert(credit == 0);
            }
            if (c * length >= tokens) assert(AscendC::event_sets == sets_before);
        }
        assert(output.front() == -77 && output.back() == -77);
        assert(output_w.front() == -77 && output_w.back() == -77);
        assert(std::equal(expected.begin(), expected.end(), output.begin() + 1));
        assert(std::equal(expected_w.begin(), expected_w.end(), output_w.begin() + 1));
        for (unsigned i = (db + 1) * tile.bytes / 8; i < resource.ub.size(); ++i)
            assert(resource.ub[i] == 0xababababababababULL);
    }
}

int main()
{
    for (unsigned world : {8U, 16U, 256U}) {
        for (unsigned tokens : {0U, 1U, 65U, 513U, 2057U}) {
            for (unsigned channels : {1U, 8U, 32U}) {
                for (unsigned k : {7U, 8U}) {
                    for (bool weighted : {false, true}) CheckPack(world, tokens, channels, k, weighted);
                }
            }
        }
    }
}
