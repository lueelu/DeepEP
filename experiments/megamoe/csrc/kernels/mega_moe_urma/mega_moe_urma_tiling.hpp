// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include <cstdint>

namespace MegaMoeUrma {
// Fixed descriptor/cache capacity. The active communicator is Config::world.
constexpr uint32_t kWorld = 16;
constexpr uint32_t kMinWorld = 4;
constexpr uint32_t kExperts = 2048;  // capacity only; active E = world * localExperts
constexpr uint32_t kMaxLocalExperts = 128;
constexpr uint32_t kAic = 28;
constexpr uint32_t kLocalWorkers = 2U * kAic;
// EP16 bounds/defaults; active work partitions are derived from Config::world.
constexpr uint32_t kCommunicationWorkers = kWorld - 1U;
constexpr uint32_t kActivationWorkers = kAic - kCommunicationWorkers;
// Host planner limit and per-peer Combine scan metadata capacity.
constexpr uint32_t kMaxTokensPerRank = 16384U;
constexpr uint32_t kMaxPeerChunks = kMaxTokensPerRank * 8U / 128U;
// One source contributes at most BS16384 * TopK8 rows; every reorder span
// contains at least one row. Each peer therefore needs at most 16KiB of bits.
constexpr uint32_t kMaxPeerReorderSpans = kMaxTokensPerRank * 8U;
constexpr uint64_t kFlagBytes = 128;
// Per-invocation counters in localFlags, cleared before the start handshake.
constexpr uint32_t kLocalFlagDispatchDone = 0U;
constexpr uint32_t kLocalFlagAicDone = 1U;
constexpr uint32_t kLocalFlagIncomingDone = 2U;
constexpr uint32_t kLocalFlagPackDone = 3U;
constexpr uint32_t kLocalFlagReorderDone = 4U;
constexpr uint32_t kLocalFlagSharedGmm1Done = 5U;
constexpr uint32_t kLocalFlagRouteSummaryDone = 6U;
constexpr uint32_t kLocalFlagRouteSourcesDone = 7U;  // committed AIV0 source snapshots
// Separate cache lines: per-core L1 ownership handoff shared W8 -> routed W4 -> shared W8.
constexpr uint32_t kLocalFlagSharedL1Released = 16U;
constexpr uint32_t kLocalFlagRoutedPrologueReleased = kLocalFlagSharedL1Released + kAic;
constexpr uint32_t kSharedLocalFlagCount = kLocalFlagRoutedPrologueReleased + kAic;
constexpr uint32_t kLocalFlagRouteExpertsDone = 8U;  // committed AIV0 expert shards
// Reduce dependency values are uint64 bit patterns stored in an int64 tensor.
// Low values address a producer/CV completion flag; tagged values address a
// local GMM2 group that writes its self return rows directly.
constexpr uint64_t kReduceSelfDependency = uint64_t{1} << 63U;
constexpr uint32_t kProtocol = 1;
constexpr uint32_t kDeviceRoutingProtocol = 8;
#define MEGAMOE_URMA_ROUTE_REORDER 1
constexpr uint32_t kTimerCores = kAic * 3U;
constexpr uint32_t kTimerPhases = 8U;
constexpr uint32_t kTimerRecordElements = 16U;
constexpr uint64_t kTimerElements = static_cast<uint64_t>(kTimerCores) * kTimerPhases * kTimerRecordElements;

// Detailed traces have independent, bounded single-writer slices per engine.
// A slice contains a 128B header followed by capacity 128B event records.
constexpr uint32_t kTraceSchema = 1U;
constexpr uint32_t kTraceRecordElements = 16U;
constexpr uint32_t kTraceHeaderElements = 16U;
constexpr uint32_t kTraceMaxCapacity = 65536U;
constexpr uint64_t kTraceUnknown = UINT64_MAX;

enum class TraceEvent : uint32_t {
    Pack = 0,
    DispatchPutX,
    DispatchPutScale,
    DispatchQuiet,
    DispatchPublish,
    SelfDispatch,  // Reserved: Pack now writes self input directly.
    WaitStartPeer,
    PublishStart,
    WaitLocalDispatch,
    WaitCombineReady,
    CombinePut,
    CombineQuiet,
    SelfCombine,  // Reserved: GMM2 now writes self return rows directly.
    FinalPeerHandshake,
    WaitDispatch,
    Gmm1Tile,
    Gmm1TileDrain,
    Gmm1Dequant,
    WaitActivation,
    Gmm2Tile,
    Gmm2Dequant,
    Gmm2Drain,
    WaitGmm1,
    ActivationSlice,
    RankBarrier,
    Reduce,
    LocalSync,
    Gmm1Drain,
    InputScaleGather,
    WaitInputScale,
    CombinePublish,
    WaitReduceReady,  // Reserved for historical per-token Reduce traces.
    ReduceDrain,      // Reserved: the Reduce envelope now includes its drain.
    // IDs 33..39 are reserved for historical JSON-only GMM envelopes.
    Gmm2Reorder = 40U,
    WaitGmm2Reorder,
    CombineSendComplete,
    DispatchSendComplete,  // Reserved here, shared coarse event IDs start at 44.
    CoarsePack,
    CoarseLocalSyncStart,
    CoarseGMM,
    CoarseDequant,
    CoarseDispatch,
    CoarseCombine,
    CoarseActivation,
    CoarseReduce,
    CoarseLocalSyncEnd,
    CoarseDispatchSubmit,
    PackRetire,
    SharedGmm1Tile,
    SharedGmm1Drain,
    SharedActivation,
    SharedGmm2Tile,
    SharedGmm2Drain,
    SharedAdd,
    CoarseSharedCombineSubmit,
    RoutePrepare,
    RouteSummary,
    WaitRouteSummary,
    DispatchReceiveComplete
};

struct TraceMeta {
    uint64_t peer = kTraceUnknown;
    uint64_t cv = kTraceUnknown;
    uint64_t fragment = kTraceUnknown;
    uint64_t expert = kTraceUnknown;
    uint64_t row = kTraceUnknown;
    uint64_t rows = kTraceUnknown;
    uint64_t tileM = kTraceUnknown;
    uint64_t tileN = kTraceUnknown;
    uint64_t columns = kTraceUnknown;
    uint64_t bytes = 0U;
};

// One expert-major M<=256 group, gathering rows across source/CV boundaries.
struct Fragment {
    uint32_t expert;
    uint32_t segmentBegin;
    uint64_t rowBegin;
    uint32_t rows;
    uint32_t segmentCount;
    uint64_t gmm1TileBegin;
    uint64_t gmm2TileBegin;
};

// One contiguous input/output run inside a compute group and a transport CV.
// receiveRow addresses source-major input/GMM2 output. returnRow is the source's
// expanded-row offset, including direct self output in the local return window.
struct RowSegment {
    uint32_t sourceRank;
    uint32_t cv;
    uint64_t receiveRow;
    uint64_t returnRow;
    uint32_t tileRow;
    uint32_t rows;
    uint32_t inputReadyIndex;
    uint32_t reserved;
};

// Full-width row copies owned by the communication core for this peer/CV.
struct PeerReorderSpan {
    uint64_t sourceRow;
    uint64_t destinationRow;
    uint32_t fragmentIndex;
    uint32_t rows;
};
struct PeerReorderRange {
    uint32_t spanBegin;
    uint32_t spanCount;
};

// One batch per local expert, including an empty range for an empty expert.
struct ComputeBatch {
    uint32_t fragmentBegin;
    uint32_t fragmentCount;
};

enum class ComputeBatchStage : uint32_t { None = 0U, Gmm1 = 1U, Gmm2 = 2U };

// Host-planned order shared by each AIC and its paired AIV0. Whole expert
// batches retain their existing context and global tile ownership boundaries.
struct ComputeBatchTask {
    ComputeBatchStage stage;
    uint32_t batchIndex;
};

// Send chunks use local expanded-row offsets. Receive chunks describe the
// corresponding source-major input and full-width GMM2 output interval.
// A receive chunk's fragmentBegin/Count index chunkGroupIndices, not fragments:
// all referenced compute groups must finish before the contiguous return PUT.
struct Chunk {
    uint32_t peer;
    uint32_t cv;
    uint64_t localRow;
    uint64_t remoteRow;
    uint32_t rows;
    uint32_t fragmentBegin;
    uint32_t fragmentCount;
    uint32_t reserved;
};

// A fixed Pack round contains CV r for every destination, including self.
// Empty peer streams retain their fixed peer slot with rows=0. The device
// divides the concatenated virtual row interval equally among Pack workers.
struct PackRound {
    Chunk chunks[kWorld];
    uint64_t rows;
};

struct Config {
    uint32_t bs;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t topk;
    uint32_t localExperts;
    uint32_t cvRows;
    uint32_t rank;
    uint32_t world = kWorld;
    uint32_t maxPeerChunks;
    uint32_t fragmentCount;
    uint32_t sendChunkCount;
    uint32_t recvChunkCount;
    uint32_t gmm1AheadMGroups = 12U;
    uint32_t computeTaskCount;
    uint32_t packRoundCount;
    uint64_t receiveRows;
    uint64_t gmm1Tiles;
    uint64_t gmm2Tiles;
    uint64_t epoch;
    uint64_t receiveXOffset;
    uint64_t receiveScaleOffset;
    uint64_t returnYOffset;
    uint64_t dispatchReadyOffset;
    uint64_t combineReadyOffset;
    uint64_t barrierOffset;
    uint64_t symmetricBytes;
    // One rank's self routes form a contiguous interval in each row layout.
    // They need no intermediate expanded/GMM2-output copy.
    uint64_t selfSendRow;
    uint64_t selfRecvRow;
    uint64_t selfRows;
    uint64_t sourceRowCapacity;  // fixed pitch; no receiver offset reply
    uint64_t routeInboxOffset, routeEntryOffset, routeReadyOffset;
    uint32_t routeSegmentCount, routeLinkCount;
};

// This POD is copied as kernel launch arguments; every address is device GM.
struct Arguments {
    Config config;
    uint64_t symmetric;
    uint64_t x;
    uint64_t xScale;
    uint64_t order;
    uint64_t reduceIndex;
    uint64_t reduceDependencies;  // Cached token-major [BS,K] ready descriptors.
    uint64_t weights;
    uint64_t weight1;
    uint64_t weight2;
    uint64_t weightScale1;
    uint64_t weightScale2;
    uint64_t output;
    uint64_t expandedX;
    uint64_t expandedScale;
    uint64_t gatheredInputScale;
    uint64_t gmm1Output;
    uint64_t activation;
    uint64_t activationScale;
    uint64_t gmm2Output;
    uint64_t gmm2Contiguous;  // Expert-major BF16 scratch; not visible to Combine.
    uint64_t fragments;
    uint64_t segments;
    uint64_t computeBatches;
    uint64_t computeTasks;
    uint64_t chunkGroupIndices;
    uint64_t peerReorderRanges;
    uint64_t peerReorderSpans;
    uint64_t sendChunks;
    uint64_t recvChunks;
    uint64_t packRounds;
    uint64_t packReady;  // Local int32 completed rows per peer/CV, 128B slots.
    uint64_t gmm1Done;
    uint64_t inputScaleDone;
    uint64_t activationDone;
    uint64_t gmm2Done;        // Completed N tiles per M-group, BEFORE peer row reorder.
    uint64_t gmm2MatmulDone;  // Reserved, zero: paired direct Reorder needs no intermediate GM flag.
    uint64_t localFlags;
    uint64_t timer;
    uint64_t trace;
    uint64_t traceCapacity;
    uint64_t coarseTrace;
    uint64_t systemWorkspace;
    uint64_t ffts;
    uint32_t sharedExperts, sharedIntermediate;
    uint64_t sharedWeight1, sharedWeight2, sharedScale1, sharedScale2;
    uint64_t sharedGmm1, sharedActivation, sharedActivationScale, sharedOutput;
    uint64_t sharedGmm1Done, sharedActivationDone;
    uint64_t topkIds, routeHistogram, routeInfo, routeConfig, routeCounts, routeOffsets;
    uint64_t expertCountsOutput;
    uint32_t topkIdBytes;
    uint64_t sharedRoutedOutput;  // S1 FP32 routed sum; round only after adding shared result.
    uint64_t gmmIssued;           // per-AIC monotonic routed output submission count
    uint64_t expertRoutes;        // int32 [384, sourceRowCapacity], one owner per expert
};
static_assert(sizeof(Fragment) == 40);
static_assert(sizeof(RowSegment) == 40);
static_assert(sizeof(PeerReorderSpan) == 24);
static_assert(sizeof(PeerReorderRange) == 8);
static_assert(sizeof(ComputeBatch) == 8);
static_assert(sizeof(ComputeBatchTask) == 8);
static_assert(sizeof(Chunk) == 40);
static_assert(sizeof(PackRound) == kWorld * sizeof(Chunk) + sizeof(uint64_t));
}  // namespace MegaMoeUrma

extern "C" void mega_moe_urma_kernel_do(uint32_t block_dim, uint32_t ub_bytes, void* stream,
                                        const MegaMoeUrma::Arguments* arguments);
