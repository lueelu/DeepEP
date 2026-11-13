// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#include "launch.h"
#include "acl/acl.h"
#include "shmem.h"
#include "host/utils/shmem_log.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace deepep::hierarchical;
namespace {
uint32_t logRank;
void Check(int status, const char* operation)
{
    if (status != 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}
void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
uint32_t Number(const char* text)
{
    Require(text && *text, "Empty integer argument");
    uint64_t value = 0;
    for (const char* c = text; *c; ++c) {
        Require(*c >= '0' && *c <= '9', "Expected unsigned decimal integer");
        value = value * 10 + (*c - '0');
        Require(value <= UINT32_MAX, "Integer exceeds uint32 range");
    }
    return static_cast<uint32_t>(value);
}
void Log(int level, const char* message)
{
    const char* requested = std::getenv("SHMEM_LOG_LEVEL");
    const int threshold = requested && std::strcmp(requested, "DEBUG") == 0 ? 0 :
                          requested && std::strcmp(requested, "INFO") == 0 ? 1 : 2;
    if (level >= threshold && message) std::fprintf(stderr, "[shmem] rank=%u level=%d %s\n", logRank, level, message);
}
void Upload(void* dst, const std::vector<uint8_t>& src)
{
    Check(aclrtMemcpy(dst, src.size(), src.data(), src.size(), ACL_MEMCPY_HOST_TO_DEVICE), "upload");
}
std::vector<uint8_t> Download(void* src, size_t bytes)
{
    std::vector<uint8_t> result(bytes);
    Check(aclrtMemcpy(result.data(), bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "download");
    return result;
}
void FillInput(std::vector<uint8_t>& bytes, Layout layout, uint32_t rank, bool scale)
{
    const uint64_t stride = scale ? layout.ScaleStride() : layout.DataStride();
    const uint64_t count = uint64_t(layout.m) * (scale ? layout.n / 32 : layout.n);
    for (uint32_t peer = 0; peer < layout.world; ++peer)
        for (uint64_t i = 0; i < count; ++i) bytes[peer * stride + i] = Pattern(rank, peer, i, scale);
}
void Validate(const std::vector<uint8_t>& bytes, Layout layout, uint32_t rank, bool scale)
{
    const uint64_t stride = scale ? layout.ScaleStride() : layout.DataStride();
    const uint64_t count = uint64_t(layout.m) * (scale ? layout.n / 32 : layout.n);
    for (uint32_t source = 0; source < layout.world; ++source) {
        for (uint64_t i = 0; i < stride; ++i) {
            const uint8_t expected = i < count ? Pattern(source, rank, i, scale) : kPadding;
            if (bytes[source * stride + i] != expected) {
                throw std::runtime_error(std::string(scale ? "scale" : "data") + " mismatch: source=" +
                    std::to_string(source) + " byte=" + std::to_string(i));
            }
        }
    }
}
uint64_t ExpectedHash(Layout layout, uint32_t rank, uint32_t window)
{
    uint64_t hash = kHashSeed;
    for (uint32_t source = 0; source < layout.world; ++source) {
        for (bool scale : {false, true}) {
            const uint64_t columns = scale ? layout.n / 32 : layout.n;
            const uint64_t start = uint64_t(window) * layout.windowM * columns;
            for (uint64_t i = 0; i < uint64_t(layout.Rows(window)) * columns; ++i)
                hash = HashByte(hash, Pattern(source, rank, start + i, scale));
        }
    }
    return hash;
}
int Run(int argc, char** argv)
{
    Require(argc == 10, "Usage: hierarchical_alltoall WORLD RANK tcp://IP:PORT M N WINDOW_M DEVICE QP_NUM VERIFY_READY");
    Layout layout{Number(argv[4]), Number(argv[5]), Number(argv[1]), Number(argv[6]), Number(argv[8])};
    const uint32_t rank = Number(argv[2]), device = Number(argv[7]), verify = Number(argv[9]);
    logRank = rank;
    Require(layout.world > 0 && layout.world <= kCoreCount && rank < layout.world, "Invalid world/rank");
    Require(layout.world <= HIERARCHICAL_GROUP_SIZE || layout.world == 2 * HIERARCHICAL_GROUP_SIZE,
            "Use one group or two complete groups");
    Require(layout.m && layout.n && layout.n % 32 == 0 && layout.windowM, "M/WINDOW_M must be positive; N must be a positive multiple of 32");
    Require(layout.qps == 1 || layout.qps == 2 || layout.qps == 4, "QP_NUM must be 1, 2 or 4");
    Require(verify <= 1 && device <= INT32_MAX, "Invalid device or VERIFY_READY");
    Require(uint64_t(std::min(layout.windowM, layout.m)) * layout.n <= UINT32_MAX,
            "A single window must fit uint32 element count; decrease WINDOW_M");
    // Prevent arithmetic overflow before any allocation; there is no implicit truncation.
    Require(uint64_t(layout.m) * layout.n <= (std::numeric_limits<uint64_t>::max() / 8 / layout.world) - 64,
            "Tensor byte count overflows");
    Require(layout.Windows() < UINT32_MAX, "Epoch counter would wrap");
    const std::string endpoint(argv[3]);
    const std::regex address("tcp://([0-9]{1,3}\\.){3}[0-9]{1,3}:[0-9]{1,5}");
    Require(std::regex_match(endpoint, address), "Endpoint must be tcp://IPv4:PORT without whitespace");
    const auto port = std::stoul(endpoint.substr(endpoint.rfind(':') + 1));
    Require(port > 0 && port <= 65535, "Invalid TCP port");

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t requiredAiv = layout.world + 1; // Peer senders plus an independent coordinator.
    Require(platform && platform->GetCoreNumAiv() >= requiredAiv &&
            platform->GetCoreNumAic() >= (requiredAiv + 1) / 2,
            "Insufficient AIC/AIV cores for peer senders and coordinator");
    const uint32_t blocks = platform->GetCoreNumAic();
    const char* soc = aclrtGetSocName();
    Require(soc && std::string(soc).find("Ascend950") == 0, "Ascend950 is required");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    Check(aclshmemx_set_extern_logger(Log), "SHMEM logger");
    Check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, layout.qps), "UDMA QPs");
    const bool crossNode = layout.world > HIERARCHICAL_GROUP_SIZE;
    if (crossNode) Check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, layout.qps), "RDMA QPs");
    aclshmemx_init_attr_t attr{}; // Preserve the selected SDK's default ABI version.
    aclshmemx_uniqueid_t uid{};
    uid.version = ACLSHMEM_UNIQUEID_VERSION;
    uid.my_pe = rank;
    uid.n_pes = layout.world;
    attr.my_pe = rank;
    attr.n_pes = layout.world;
    Require(endpoint.size() < sizeof(attr.ip_port), "Endpoint too long");
    std::memcpy(attr.ip_port, endpoint.c_str(), endpoint.size() + 1);
    attr.comm_args = &uid;
    attr.local_mem_size = Align(layout.TensorBytes() + layout.Ready().Bytes() + BarrierBytes(layout.world) + (16ULL << 20), 2ULL << 20);
    attr.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(
        ACLSHMEM_DATA_OP_UDMA | (crossNode ? ACLSHMEM_DATA_OP_ROCE : 0));
#if HIERARCHICAL_SHMEM_RUNTIME_CONFIG
    attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_DISABLED;
#else
    Check(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, false), "disable shared jetty");
#endif
    std::printf("rank=%u/%u MTE=1 UDMA=%u RDMA=%u qp=%u windows=%u heap=%llu\n", rank, layout.world,
        std::min(layout.world, uint32_t(HIERARCHICAL_GROUP_SIZE)) - 1,
        crossNode ? layout.world - HIERARCHICAL_GROUP_SIZE : 0, layout.qps, layout.Windows(),
        static_cast<unsigned long long>(attr.local_mem_size));
    std::fflush(stdout);
    Check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "aclshmemx_init_attr");
    for (uint32_t peer = 0; peer < layout.world; ++peer) {
        if (peer == rank || peer / HIERARCHICAL_GROUP_SIZE != rank / HIERARCHICAL_GROUP_SIZE) continue;
        for (uint32_t qp = 0; qp < layout.qps; ++qp) {
            aclshmemx_net_layer_t layer;
            Check(aclshmemx_get_qp_net_layer(ACLSHMEM_DATA_OP_UDMA, peer, qp, &layer), "same-group UDMA route");
        }
    }
    // Symmetric allocations must have identical sizes and order on all ranks.
    auto* tensors = static_cast<uint8_t*>(aclshmem_align(64, layout.TensorBytes()));
    void* control = aclshmem_align(64, layout.Ready().Bytes());
    void* barrier = aclshmem_calloc(1, BarrierBytes(layout.world));
    Require(tensors && control && barrier, "Symmetric allocation failed");
    void* checks = nullptr;
    Check(aclrtMalloc(&checks, CheckBytes(layout), ACL_MEM_MALLOC_HUGE_FIRST), "allocate checks");
    auto* input = tensors;
    auto* output = input + layout.DataBytes();
    auto* inputScale = output + layout.DataBytes();
    auto* outputScale = inputScale + layout.ScaleBytes();
    std::vector<uint8_t> data(layout.DataBytes(), kPadding), scale(layout.ScaleBytes(), kPadding);
    Upload(output, data);
    Upload(outputScale, scale);
    FillInput(data, layout, rank, false);
    FillInput(scale, layout, rank, true);
    Upload(input, data);
    Upload(inputScale, scale);
    std::vector<uint8_t> signals(layout.Ready().Bytes(), 0);
    for (uint32_t window = 0; window < layout.Windows(); ++window) {
        const uint32_t epoch = window + 1;
        std::memcpy(signals.data() + layout.Ready().ValueOffset(window), &epoch, sizeof(epoch));
    }
    Upload(control, signals);
    Check(aclrtMemset(checks, CheckBytes(layout), 0, CheckBytes(layout)), "clear checks");
    const uint64_t ffts = shmemx_get_ffts_config();
    LaunchBarrier(rank, layout.world, 1, barrier, stream, ffts);
    Check(aclrtSynchronizeStream(stream), "input-ready barrier");
    const LaunchArgs args{layout, rank, reinterpret_cast<uintptr_t>(input), reinterpret_cast<uintptr_t>(output),
        reinterpret_cast<uintptr_t>(inputScale), reinterpret_cast<uintptr_t>(outputScale),
        reinterpret_cast<uintptr_t>(control), {reinterpret_cast<uintptr_t>(checks), verify != 0}};
    LaunchAllToAll(args, blocks, stream, ffts);
    Check(aclrtSynchronizeStream(stream), "multi-round AllToAll");
    // Wait for every sender's final drain before validation/free/finalize.
    LaunchBarrier(rank, layout.world, 2, barrier, stream, ffts);
    Check(aclrtSynchronizeStream(stream), "output-ready barrier");
    Validate(Download(output, layout.DataBytes()), layout, rank, false);
    Validate(Download(outputScale, layout.ScaleBytes()), layout, rank, true);
    if (verify) {
        const auto actual = Download(checks, CheckBytes(layout));
        for (uint32_t window = 0; window < layout.Windows(); ++window) {
            uint64_t hash;
            std::memcpy(&hash, actual.data() + uint64_t(window) * sizeof(hash), sizeof(hash));
            Require(hash == ExpectedHash(layout, rank, window), "AIC early-consumption checksum mismatch");
        }
    }
    // All participants must finish checking before collective teardown.
    LaunchBarrier(rank, layout.world, 3, barrier, stream, ffts);
    Check(aclrtSynchronizeStream(stream), "validation barrier");
    Check(aclrtFree(checks), "free checks");
    aclshmem_free(barrier);
    aclshmem_free(control);
    aclshmem_free(tensors);
    Check(aclshmem_finalize(), "aclshmem_finalize");
    Check(aclrtDestroyStream(stream), "destroy stream");
    Check(aclrtResetDevice(device), "reset device");
    Check(aclFinalize(), "aclFinalize");
    std::printf("rank=%u PASS data+scale windows=%u qp=%u verify_ready=%u\n", rank, layout.Windows(), layout.qps, verify);
    return 0;
}
} // namespace
int main(int argc, char** argv)
{
    try { return Run(argc, argv); }
    catch (const std::exception& error) {
        // Do not enter collective cleanup after a rank fails; preserve the original error.
        std::fprintf(stderr, "rank=%u ERROR: %s\n", logRank, error.what());
        return 1;
    }
}
