// Appended to the production packing bodies by test_channel_metadata.py.
// CPU DMA is immediate: this checks addressing and event credits, not device ordering.
struct MetadataSender {
    std::vector<uint32_t> peers;
    uint32_t Count() const
    {
        return peers.size();
    }
    uint32_t Rank(uint32_t p) const
    {
        return peers[p];
    }
};
void CheckPack(unsigned world, unsigned tokens, unsigned channels, unsigned k, bool weighted, unsigned workers)
{
    DispatchTilingData td{};
    td.num_tokens = tokens;
    td.num_channels = channels;
    td.world_size = world;
    td.num_experts = world * 16;
    td.num_topk = k;
    td.has_topk_weights = weighted;
    td.channel_metadata_tile_tokens = 32;
    td.udma_workers_per_peer = workers;
    const auto tile = MakeChannelMetadataTile(32, 64, k, sizeof(topk_idx_t), weighted);
    const unsigned length = (tokens + channels - 1) / channels;
    const unsigned stride = (length + 7) / 8 * 8;
    const unsigned size = world * channels * stride * k;
    std::vector<int32_t> plan(world * channels * stride, -1);
    std::vector<unsigned> counts(world * channels);
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
                plan[(p * channels + c) * stride + ordinal] = t;
                for (unsigned j = 0; j < k; ++j) {
                    const bool valid =
                        idx[t * k + j] >= topk_idx_t(p * 16) && idx[t * k + j] < topk_idx_t((p + 1) * 16);
                    const unsigned out = ((p * channels + c) * stride + ordinal) * k + j;
                    expected[out] = valid ? idx[t * k + j] - p * 16 : topk_idx_t(-1);
                    if (weighted) expected_w[out] = valid ? weights[t * k + j] : 0;
                }
                ++ordinal;
            }
            counts[p * channels + c] = ordinal;
        }
    td.notify_topk_idx = reinterpret_cast<uint64_t>(idx.data());
    td.notify_topk_weights = reinterpret_cast<uint64_t>(weights.data());
    td.packed_channel_indices = reinterpret_cast<uint64_t>(output.data() + 1);
    td.packed_channel_weights = reinterpret_cast<uint64_t>(output_w.data() + 1);
    std::vector<int32_t> prefixes(world * channels);
    for (unsigned p = 0; p < world; ++p) {
        unsigned sum = 0;
        for (unsigned c = 0; c < channels; ++c) {
            sum += counts[p * channels + c];
            prefixes[p * channels + c] = sum;
        }
    }
    constexpr unsigned scratch = DispatchLiveSendUbBytes(64, 2), desc_bytes = 64 * 4 * 4;
    for (unsigned db : {0U, 1U}) {
        Resource resource;
        std::fill(resource.ub.begin(), resource.ub.end(), 0xababababababababULL);
        std::fill(output.begin(), output.end(), -77);
        std::fill(output_w.begin(), output_w.end(), -77);
        td.channel_metadata_double_buffer = db;
        unsigned used = scratch;
        for (unsigned first = 0; first < world; first += 64) {
            const auto sets_before = AscendC::event_sets;
            MetadataSender sender;
            for (unsigned p = first; p < std::min(first + 64, world); ++p) sender.peers.push_back(p);
            const auto cached = ExtendMetadataPlanCache(tile, sender.Count(), stride,
                                                        (platform::kKernelUbBytes - scratch) / (db + 1) - desc_bytes);
            used = std::max(used, scratch + (db + 1) * (desc_bytes + cached.input.bytes));
            plan_begin = reinterpret_cast<uintptr_t>(plan.data());
            plan_end = plan_begin + plan.size() * sizeof(int32_t);
            plan_bytes = 0;
            unsigned posted = 0;
            auto post = [&](unsigned ch) {
                ++posted;
                // A channel can be published only after its exact outputs are ready.
                for (unsigned p : sender.peers) {
                    const unsigned b = (p * channels + ch) * stride * k, n = counts[p * channels + ch] * k;
                    assert(std::equal(expected.begin() + b, expected.begin() + b + n, output.begin() + 1 + b));
                    assert(std::equal(expected_w.begin() + b, expected_w.begin() + b + n, output_w.begin() + 1 + b));
                    // Future channels must not be packed before this callback.
                    if (ch + workers < channels) {
                        const unsigned next = (p * channels + ch + workers) * stride * k;
                        assert(std::all_of(output.begin() + 1 + next, output.begin() + 1 + next + stride * k,
                                           [](auto x) { return x == -77; }));
                    }
                }
            };
            for (unsigned core = 0; core < workers; ++core)
                PackRemoteChannelMetadata<64>(resource, sender, reinterpret_cast<GM_ADDR>(plan.data()),
                                              reinterpret_cast<GM_ADDR>(prefixes.data()), td, core, scratch, post);
            unsigned expected_posts = 0;
            uint64_t total_plan_rows = 0;
            for (unsigned ch = 0; ch < channels; ++ch) {
                unsigned n = 0;
                for (unsigned p : sender.peers) n += counts[p * channels + ch];
                expected_posts += n != 0;
                total_plan_rows += n;
            }
            assert(posted == expected_posts);
            if (cached.plan_tokens >= stride) {
                // A resident channel loads at most once per input slot, rather
                // than reloading its suffix for every token window.
                assert(plan_bytes <= (db + 1) * total_plan_rows * 4U);
                if (!db) assert(plan_bytes == total_plan_rows * 4U);
            }
            plan_begin = plan_end = 0;
            for (auto& events : AscendC::credits) {
                for (int credit : events) assert(credit == 0);
            }
            (void)sets_before;
        }
        assert(output.front() == -77 && output.back() == -77);
        assert(output_w.front() == -77 && output_w.back() == -77);
        assert(std::equal(expected.begin(), expected.end(), output.begin() + 1));
        assert(std::equal(expected_w.begin(), expected_w.end(), output_w.begin() + 1));
        for (unsigned i = 0; i < scratch / 8; ++i) assert(resource.ub[i] == 0xababababababababULL);
        for (unsigned i = used / 8; i < resource.ub.size(); ++i) assert(resource.ub[i] == 0xababababababababULL);
    }
}

void CheckDirectSelf(bool cached, bool direct, unsigned count, unsigned double_buffer, bool omit)
{
    DispatchTilingData td{};
    td.rank = 1;
    td.world_size = 4;
    td.num_channels = 3;
    td.num_topk = 8;
    td.has_topk_weights = 1;
    td.cached_mode = cached;
    td.metadata_direct_output = direct;
    td.omit_topk_output = omit;
    td.num_output_tokens = count + 3;
    td.num_experts = 64;
    td.channel_metadata_tile_tokens = 32;
    td.channel_metadata_double_buffer = double_buffer;
    const unsigned stride = count + 1, slot = (td.rank * td.num_channels + 2) * stride;
    std::vector<int32_t> plan(12 * stride);
    std::vector<topk_idx_t> indices((count + 100) * 8);
    std::vector<float> weights((count + 100) * 8);
    for (unsigned i = 0; i < count; ++i) {
        plan[slot + i] = 100 + i;
        for (unsigned k = 0; k < 8; ++k) {
            indices[(100 + i) * 8 + k] = k == 3 ? -1 : 16 + k;
            weights[(100 + i) * 8 + k] = float(i * 8 + k + 1);
        }
    }
    td.notify_topk_idx = reinterpret_cast<uint64_t>(indices.data());
    td.notify_topk_weights = reinterpret_cast<uint64_t>(weights.data());
    td.src_idx_buffer_offset = 32;
    td.topk_idx_buffer_offset = 32 + (count + 3) * 4;
    td.topk_weights_buffer_offset = td.topk_idx_buffer_offset + (count + 3) * 8 * sizeof(topk_idx_t);
    std::vector<uint8_t> workspace(td.topk_weights_buffer_offset + (count + 3) * 32 + 32, 0xab);
    std::vector<int32_t> src(count + 3, -77);
    std::vector<topk_idx_t> out_idx((count + 3) * 8, -77);
    std::vector<float> out((count + 3) * 8, -77);
    td.metadata_src_output = reinterpret_cast<uint64_t>(src.data());
    td.metadata_weight_output = reinterpret_cast<uint64_t>(out.data());
    td.metadata_topk_output = omit ? 0 : reinterpret_cast<uint64_t>(out_idx.data());
    const auto untouched = workspace;
    Resource resource;
    CopySelfChannelMetadata(resource, reinterpret_cast<GM_ADDR>(plan.data()), workspace.data(), td, 2, 1, count,
                            stride);
    for (unsigned i = 0; i < count; ++i) {
        int32_t source;
        std::memcpy(&source, workspace.data() + td.src_idx_buffer_offset + (1 + i) * 4, 4);
        if (!cached) assert((direct ? src[1 + i] : source) == plan[slot + i]);
        if (cached || direct) assert(source == int32_t(0xabababab));
        for (unsigned k = 0; k < 8; ++k) {
            topk_idx_t index;
            float weight;
            std::memcpy(&index, workspace.data() + td.topk_idx_buffer_offset + ((1 + i) * 8 + k) * sizeof(topk_idx_t),
                        sizeof(index));
            std::memcpy(&weight, workspace.data() + td.topk_weights_buffer_offset + ((1 + i) * 8 + k) * 4, 4);
            assert((direct && !omit ? out_idx[(1 + i) * 8 + k] : index) ==
                   (k == 3 ? topk_idx_t(-1) : topk_idx_t(k)));
            assert(out[(1 + i) * 8 + k] == (k == 3 ? 0 : weights[(100 + i) * 8 + k]));
        }
    }
    assert(src.front() == -77 && src.back() == -77 && out.front() == -77 && out.back() == -77);
    if (!count) assert(workspace == untouched);
    assert(out_idx.front() == -77 && out_idx.back() == -77);
    if (direct && !omit)
        assert(std::equal(untouched.begin() + td.topk_idx_buffer_offset,
                          untouched.begin() + td.topk_weights_buffer_offset,
                          workspace.begin() + td.topk_idx_buffer_offset));
    else
        assert(std::all_of(out_idx.begin(), out_idx.end(), [](auto x) { return x == -77; }));
    assert(std::equal(untouched.begin() + td.topk_weights_buffer_offset, untouched.end(),
                      workspace.begin() + td.topk_weights_buffer_offset));
    for (auto& events : AscendC::credits)
        for (int credit : events) assert(credit == 0);
}

int main()
{
    for (bool cached : {false, true})
        for (bool direct : {false, true})
            for (unsigned count : {0U, 1U, 65U, 1025U})
                for (unsigned db : {0U, 1U})
                    for (bool omit : {false, true}) {
                        if (omit && !cached) continue;
                        CheckDirectSelf(cached, direct, count, db, omit);
                    }
    for (unsigned world : {8U, 16U, 32U, 64U, 128U, 256U}) {
        for (unsigned tokens : {0U, 1U, 65U, 513U, 2057U}) {
            for (unsigned channels : {1U, 8U, 32U}) {
                for (unsigned k : {7U, 8U}) {
                    for (bool weighted : {false, true})
                        for (unsigned workers : {1U, 2U}) CheckPack(world, tokens, channels, k, weighted, workers);
                }
            }
        }
    }
}
