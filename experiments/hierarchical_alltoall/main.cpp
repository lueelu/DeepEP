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
using deepep::comm::Align;
using deepep::comm::kernel::GetCommLoops;
using deepep::comm::tile::GetReadyValueOffset;
using deepep::comm::tile::GetReadyWorkspaceSize;
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
    // SHMEM invokes the external callback before applying its own log filter.
    constexpr const char* levels[] = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
    int threshold = 2;
    if (requested) {
        for (int i = 0; i < 5; ++i) {
            if (std::strcmp(requested, levels[i]) == 0) {
                threshold = i;
                break;
            }
        }
    }
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
void FillInput(
    std::vector<uint8_t>& bytes, uint32_t rankIdx, uint32_t rankSize, uint64_t peerStrideBytes,
    uint64_t peerBytes, bool scale)
{
    for (uint32_t peer = 0; peer < rankSize; ++peer) {
        for (uint64_t i = 0; i < peerBytes; ++i) {
            bytes[peer * peerStrideBytes + i] = Pattern(rankIdx, peer, i, scale);
        }
    }
}

void Validate(
    const std::vector<uint8_t>& bytes, uint32_t rankIdx, uint32_t rankSize, uint64_t peerStrideBytes,
    uint64_t peerBytes, bool scale)
{
    for (uint32_t srcRankIdx = 0; srcRankIdx < rankSize; ++srcRankIdx) {
        for (uint64_t i = 0; i < peerStrideBytes; ++i) {
            const uint8_t expected = i < peerBytes ? Pattern(srcRankIdx, rankIdx, i, scale) : kPadding;
            if (bytes[srcRankIdx * peerStrideBytes + i] != expected) {
                throw std::runtime_error(std::string(scale ? "scale" : "data") + " mismatch: source=" +
                    std::to_string(srcRankIdx) + " byte=" + std::to_string(i));
            }
        }
    }
}

int Run(int argc, char** argv)
{
    Require(argc == 9 || argc == 11,
            "Usage: hierarchical_alltoall WORLD RANK tcp://IP:PORT M N WINDOW_M DEVICE QP_NUM [WARMUP ITERATIONS]");
    const bool perf = argc == 11;
    const uint32_t warmup = perf ? Number(argv[9]) : 0;
    const uint32_t iterations = perf ? Number(argv[10]) : 1;
    Require(warmup <= 10000 && iterations > 0 && iterations <= 10000, "Invalid WARMUP/ITERATIONS");
    const uint32_t rankSize = Number(argv[1]);
    const uint32_t rankIdx = Number(argv[2]);
    const uint32_t m = Number(argv[4]);
    const uint32_t n = Number(argv[5]);
    const uint32_t commSizeM = Number(argv[6]);
    const uint32_t device = Number(argv[7]);
    const uint32_t qpNum = Number(argv[8]);
    logRank = rankIdx;
    Require(rankSize > 0 && rankSize <= kCoreCount && rankIdx < rankSize, "Invalid world/rank");
    Require(rankSize <= HIERARCHICAL_GROUP_SIZE || rankSize == 2 * HIERARCHICAL_GROUP_SIZE,
            "Use one group or two complete groups");
    Require(m && n && n % kScaleGroup == 0 && commSizeM,
            "M/WINDOW_M must be positive; N must be a positive multiple of 32");
    Require(qpNum == 1 || qpNum == 2 || qpNum == 4, "QP_NUM must be 1, 2 or 4");
    Require(device <= INT32_MAX, "Invalid device");
    const char* udmaNetLayerEnv = std::getenv("UDMA_NET_LAYER");
    auto udmaNetLayer = ACLSHMEM_NET_LAYER_DEFAULT;
    if (udmaNetLayerEnv && *udmaNetLayerEnv) {
        const uint32_t layer = Number(udmaNetLayerEnv);
        Require(layer <= ACLSHMEM_NET_LAYER_1, "UDMA_NET_LAYER must be 0 or 1");
        udmaNetLayer = static_cast<aclshmemx_net_layer_t>(layer);
    }
    Require(uint64_t(std::min(commSizeM, m)) * n <= UINT32_MAX,
            "A single window must fit uint32 element count; decrease WINDOW_M");
    // Prevent arithmetic overflow before any allocation; there is no implicit truncation.
    Require(uint64_t(m) * n <= (std::numeric_limits<uint64_t>::max() / 8 / rankSize) - 64,
            "Tensor byte count overflows");
    const uint32_t commLoops = GetCommLoops(m, commSizeM);
    Require(commLoops < UINT32_MAX, "Epoch counter would wrap");
    uint64_t peerBytes = 0;
    uint64_t scalePeerBytes = 0;
    GetAllToAllTensorBytes(m, n, peerBytes, scalePeerBytes);
    const uint64_t peerStrideBytes = Align(peerBytes);
    const uint64_t scalePeerStrideBytes = Align(scalePeerBytes);
    const uint64_t dataBytes = peerStrideBytes * rankSize;
    const uint64_t scaleBytes = scalePeerStrideBytes * rankSize;
    const uint64_t tensorBytes = 2 * (dataBytes + scaleBytes);
    const uint64_t readyBytes = GetReadyWorkspaceSize(rankSize, qpNum, commLoops);
    const uint64_t barrierBytes = BarrierBytes(rankSize);
    const std::string endpoint(argv[3]);
    const std::regex address("tcp://([0-9]{1,3}\\.){3}[0-9]{1,3}:[0-9]{1,5}");
    Require(std::regex_match(endpoint, address), "Endpoint must be tcp://IPv4:PORT without whitespace");
    const auto port = std::stoul(endpoint.substr(endpoint.rfind(':') + 1));
    Require(port > 0 && port <= 65535, "Invalid TCP port");

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t requiredAiv = rankSize + 1; // Peer senders plus an independent coordinator.
    Require(platform && platform->GetCoreNumAiv() >= requiredAiv &&
            platform->GetCoreNumAic() >= (requiredAiv + 1) / 2,
            "Insufficient AIC/AIV cores for peer senders and coordinator");
    const uint32_t blocks = platform->GetCoreNumAic();
    const char* soc = aclrtGetSocName();
    Require(soc && std::string(soc).find("Ascend950") == 0, "Ascend950 is required");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    Check(aclshmemx_set_extern_logger(Log), "SHMEM logger");
    Check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qpNum), "UDMA QPs");
    if (udmaNetLayer != ACLSHMEM_NET_LAYER_DEFAULT) {
        for (uint32_t qpIdx = 0; qpIdx < qpNum; ++qpIdx) {
            Check(aclshmemx_set_qp_net_layer(ACLSHMEM_DATA_OP_UDMA, qpIdx, udmaNetLayer), "UDMA_NET_LAYER");
        }
    }
    const bool crossNode = rankSize > HIERARCHICAL_GROUP_SIZE;
    if (crossNode) Check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, qpNum), "RDMA QPs");
    aclshmemx_init_attr_t attr{}; // Preserve the selected SDK's default ABI version.
    aclshmemx_uniqueid_t uid{};
    uid.version = ACLSHMEM_UNIQUEID_VERSION;
    uid.my_pe = rankIdx;
    uid.n_pes = rankSize;
    attr.my_pe = rankIdx;
    attr.n_pes = rankSize;
    Require(endpoint.size() < sizeof(attr.ip_port), "Endpoint too long");
    std::memcpy(attr.ip_port, endpoint.c_str(), endpoint.size() + 1);
    attr.comm_args = &uid;
    attr.local_mem_size = Align(tensorBytes + readyBytes + barrierBytes + (16ULL << 20), 2ULL << 20);
    attr.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(
        ACLSHMEM_DATA_OP_UDMA | (crossNode ? ACLSHMEM_DATA_OP_ROCE : 0));
#if HIERARCHICAL_SHMEM_RUNTIME_CONFIG
    attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_DISABLED;
#else
    Check(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, false), "disable shared jetty");
#endif
    std::printf("rank=%u/%u MTE=1 UDMA=%u RDMA=%u qp=%u windows=%u heap=%llu\n", rankIdx, rankSize,
        std::min(rankSize, uint32_t(HIERARCHICAL_GROUP_SIZE)) - 1,
        crossNode ? rankSize - HIERARCHICAL_GROUP_SIZE : 0, qpNum, commLoops,
        static_cast<unsigned long long>(attr.local_mem_size));
    std::fflush(stdout);
    Check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "aclshmemx_init_attr");
    for (uint32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rankIdx || peer / HIERARCHICAL_GROUP_SIZE != rankIdx / HIERARCHICAL_GROUP_SIZE) continue;
        for (uint32_t qp = 0; qp < qpNum; ++qp) {
            aclshmemx_net_layer_t layer;
            Check(aclshmemx_get_qp_net_layer(ACLSHMEM_DATA_OP_UDMA, peer, qp, &layer), "same-group UDMA route");
        }
    }
    // Symmetric allocations must have identical sizes and order on all ranks.
    auto* tensors = static_cast<uint8_t*>(aclshmem_align(64, tensorBytes));
    void* control = aclshmem_align(64, readyBytes);
    void* barrier = aclshmem_calloc(1, barrierBytes);
    Require(tensors && control && barrier, "Symmetric allocation failed");
    auto* input = tensors;
    auto* output = input + dataBytes;
    auto* inputScale = output + dataBytes;
    auto* outputScale = inputScale + scaleBytes;
    std::vector<uint8_t> data(dataBytes, kPadding), scale(scaleBytes, kPadding);
    Upload(output, data);
    Upload(outputScale, scale);
    FillInput(data, rankIdx, rankSize, peerStrideBytes, peerBytes, false);
    FillInput(scale, rankIdx, rankSize, scalePeerStrideBytes, scalePeerBytes, true);
    Upload(input, data);
    Upload(inputScale, scale);
    std::vector<uint8_t> signals(readyBytes, 0);
    for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
        const uint32_t epoch = commIdx + 1;
        std::memcpy(signals.data() + GetReadyValueOffset(rankSize, qpNum, commIdx), &epoch, sizeof(epoch));
    }
    const uint64_t ffts = shmemx_get_ffts_config();
    const LaunchArgs args{
        m, n, rankIdx, rankSize, commSizeM, qpNum, reinterpret_cast<uintptr_t>(input), reinterpret_cast<uintptr_t>(output),
        reinterpret_cast<uintptr_t>(inputScale), reinterpret_cast<uintptr_t>(outputScale),
        reinterpret_cast<uintptr_t>(control)};
    aclrtEvent begin = nullptr, kernelDone = nullptr, complete = nullptr;
    if (perf) {
        Check(aclrtCreateEventWithFlag(&begin, ACL_EVENT_TIME_LINE), "create begin event");
        Check(aclrtCreateEventWithFlag(&kernelDone, ACL_EVENT_TIME_LINE), "create kernel event");
        Check(aclrtCreateEventWithFlag(&complete, ACL_EVENT_TIME_LINE), "create completion event");
        std::printf("PERF_CONFIG rank=%u world=%u m=%u n=%u window_m=%u qp=%u warmup=%u iterations=%u "
                    "data_peer_bytes=%llu scale_peer_bytes=%llu udma_layer=%s\n",
            rankIdx, rankSize, m, n, commSizeM, qpNum, warmup, iterations,
            static_cast<unsigned long long>(peerBytes), static_cast<unsigned long long>(scalePeerBytes),
            udmaNetLayerEnv && *udmaNetLayerEnv ? udmaNetLayerEnv : "auto");
    }
    uint32_t barrierEpoch = 0;
    for (uint32_t step = 0; step < warmup + iterations; ++step) {
        // Previous completion barrier ensures all ranks stopped writing old ready slots.
        // Reset locally, then wait for every rank's reset before sending new flags.
        Upload(control, signals);
        if (perf && step + 1 == warmup + iterations) {
            // A stale output from warmup must not mask an incomplete final transfer.
            Check(aclrtMemset(output, dataBytes, kPadding, dataBytes), "reset data output");
            Check(aclrtMemset(outputScale, scaleBytes, kPadding, scaleBytes), "reset scale output");
        }
        LaunchBarrier(rankIdx, rankSize, ++barrierEpoch, barrier, stream, ffts);
        Check(aclrtSynchronizeStream(stream), "input-ready barrier");
        const bool measure = perf && step >= warmup;
        if (measure) Check(aclrtRecordEvent(begin, stream), "record begin");
        LaunchAllToAll(args, blocks, stream, ffts);
        if (measure) Check(aclrtRecordEvent(kernelDone, stream), "record kernel completion");
        // The same stream orders the barrier after the local senders' final drain.
        LaunchBarrier(rankIdx, rankSize, ++barrierEpoch, barrier, stream, ffts);
        if (measure) Check(aclrtRecordEvent(complete, stream), "record world completion");
        Check(aclrtSynchronizeStream(stream), "AllToAll and output-ready barrier");
        if (measure) {
            float kernelMs = 0, completeMs = 0;
            Check(aclrtEventElapsedTime(&kernelMs, begin, kernelDone), "measure kernel");
            Check(aclrtEventElapsedTime(&completeMs, begin, complete), "measure completion");
            std::printf("PERF_SAMPLE rank=%u sample=%u kernel_us=%.3f complete_us=%.3f\n",
                rankIdx, step - warmup, kernelMs * 1000.0, completeMs * 1000.0);
        }
    }
    if (perf) {
        Check(aclrtDestroyEvent(complete), "destroy completion event");
        Check(aclrtDestroyEvent(kernelDone), "destroy kernel event");
        Check(aclrtDestroyEvent(begin), "destroy begin event");
    }
    Validate(Download(output, dataBytes), rankIdx, rankSize, peerStrideBytes, peerBytes, false);
    Validate(Download(outputScale, scaleBytes), rankIdx, rankSize, scalePeerStrideBytes, scalePeerBytes, true);
    // All participants must finish checking before collective teardown.
    LaunchBarrier(rankIdx, rankSize, ++barrierEpoch, barrier, stream, ffts);
    Check(aclrtSynchronizeStream(stream), "validation barrier");
    aclshmem_free(barrier);
    aclshmem_free(control);
    aclshmem_free(tensors);
    Check(aclshmem_finalize(), "aclshmem_finalize");
    Check(aclrtDestroyStream(stream), "destroy stream");
    Check(aclrtResetDevice(device), "reset device");
    Check(aclFinalize(), "aclFinalize");
    std::printf("rank=%u PASS data+scale windows=%u qp=%u\n", rankIdx, commLoops, qpNum);
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
