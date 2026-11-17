#pragma once
#include <cstdint>
#include "ll_compile_attrs.h"

namespace LlStageTimer {
constexpr uint32_t kSchema = 3U;
constexpr uint64_t kMagic = 0x4c4c5431ULL;
constexpr uint32_t kSlots = 32U;
constexpr uint32_t kFields = 8U;
constexpr uint64_t kActive = uint64_t(1) << 62U;

// Schema 2 packs each stage into first/start_or_end/sum/count|active (4 words).
// Header remains 8 words; the tensor shape and 2048B/core allocation are unchanged.
constexpr uint32_t StageOffset(uint32_t stage) { return 8U + (stage - 1U) * 4U; }
// 每核独占2048B，避免不同核对同一cache line写回。第0行是头，余下为阶段记录。
enum Stage : uint32_t {
    Kernel = 1,
    Magic,
    WorkerInit,
    CountInit,
    CountSend,
    Sender,
    SendPrepare,
    SendPayload,
    SendMetadata,
    CountReceive,
    CountPrefix,
    CountPublish,
    Receiver,
    RecvPrepare,
    RecvWait,
    RecvConsume,
    RecvCleanup,
    CountFinish,
    CombinePrepare,
    CombineNotify,
    CombineWaitReady,
    CombineDirect,
    CombinePartial,
    CombineAggregate,
    CombineDrain,
    CombineFinalDrain,
    SendClassify,
    SendPack,
    SendIssue,
    RecvLaneWait,
    RecvGmToUb,
    RecvLoadWait,
    RecvMetadata,
    RecvOutputWait,
    RecvOutputWrite,
    RecvForward,
    SendInputGmToUb,
    SendInputWait,
    SendTail,
    SendLaneWait,
    SendMteWrite,
    SendRdmaStage,
    SendRdmaBodyPost,
    SendRdmaBodyWait,
    SendRdmaFlagPost,
    SendRdmaFlagWait,
    SendRdmaRestore,
    StageCount
};
constexpr const char* kNames[] = {"header",
                                  "kernel",
                                  "magic",
                                  "worker_init",
                                  "count_init",
                                  "count_send",
                                  "sender",
                                  "send_prepare",
                                  "send_payload",
                                  "send_metadata",
                                  "count_receive",
                                  "count_prefix",
                                  "count_publish",
                                  "receiver",
                                  "recv_prepare",
                                  "recv_wait",
                                  "recv_consume",
                                  "recv_cleanup",
                                  "count_finish",
                                  "combine_prepare",
                                  "combine_notify",
                                  "combine_wait_ready",
                                  "combine_direct",
                                  "combine_partial",
                                  "combine_aggregate",
                                  "combine_drain",
                                  "combine_final_drain",
                                  "send_classify",
                                  "send_pack",
                                  "send_issue",
                                  "recv_lane_wait", "recv_gm_to_ub", "recv_load_wait",
                                  "recv_metadata", "recv_output_wait", "recv_output_write", "recv_forward",
                                  "send_input_gm_to_ub", "send_input_wait", "send_tail", "send_lane_wait",
                                  "send_mte_write", "send_rdma_stage", "send_rdma_body_post",
                                  "send_rdma_body_wait", "send_rdma_flag_post", "send_rdma_flag_wait",
                                  "send_rdma_restore"};
static_assert(sizeof(kNames) / sizeof(kNames[0]) == StageCount, "timer names");
}  // namespace LlStageTimer
