/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include "common/types.h"
#include "runtime/shmem_instance.h"
#include "ops/dispatch_host_timing.h"
#include "ops/dispatch_device_timing.h"
#include "ops/dispatch_channel_metadata.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_shmem_unique_id", &GetShmemUniqueId);
    m.def("get_shmem_bootstrap", &GetShmemBootstrap);
    m.attr("fused_layout_notify_version") = 1;
    m.attr("fused_notify_pipeline_version") = 1;
    m.def("take_dispatch_device_timing", &ascend_deepep::ops::TakeDispatchDeviceTiming);
    m.def("take_combine_device_timing", &ascend_deepep::ops::TakeCombineDeviceTiming);
    m.def("last_dispatch_channel_metadata", [] { return ascend_deepep::ops::LastDispatchChannelMetadata(); });
    m.def("last_dispatch_channel_metadata_tile_tokens", [] {
        return ascend_deepep::ops::LastDispatchChannelMetadataTileTokens();
    });
    m.attr("dispatch_channel_metadata_version") = 2;
    m.def("last_dispatch_overlap_options", [] { return ascend_deepep::ops::LastDispatchOverlapOptions(); });
    m.attr("dispatch_overlap_version") = 1;
    m.def("last_dispatch_metadata_options", [] { return ascend_deepep::ops::LastDispatchMetadataOptions(); });
    m.attr("dispatch_metadata_output_version") = 1;
    m.attr("dispatch_device_timing_version") = 2;
    m.attr("combine_device_timing_version") = 2;
    m.attr("dispatch_device_timing_cycles_per_us") = ascend_deepep::kNpuSystemCyclesPerUs;
    m.attr("dispatch_device_timing_fields") = std::vector<std::string>{
        "version",        "rank",          "core",          "cached",      "payload_peers", "whole_channel",
        "src_threads",    "topk_threads",  "entry_cycles",  "init_done",   "begin_done",    "metadata_done",
        "payload_queued", "published",     "incoming_done", "quiet_done",  "sync_done",     "output_done",
        "handle_done",    "work_done",     "src_cycles",    "topk_cycles", "self_cycles",   "list_cycles",
        "list_windows",   "put_rows_calls",
        "descriptor_cycles", "wqe_build_cycles", "wqe_submit_cycles", "credit_cycles", "credit_blocked_cycles",
        "credit_checks", "credit_blocked", "credit_max_cycles", "sq_used_max", "sq_used_sum", "sq_empty_checks",
        "wqes", "flushes", "short_flushes", "spans", "state_flushes", "qp_flushes", "index_flushes",
        "metadata_wqes", "metadata_flushes", "submit_no_credit", "self_begin", "self_end", "operation",
        "incoming_begin", "topk_init_cycles", "source_cycles", "slots_cycles",
        "src_output_bytes", "topk_output_bytes", "weight_output_bytes",
        "cqe_requested", "credit_checkpoints", "credit_quiet_calls", "final_quiet_cycles", "final_quiet_calls", "sq_wraps"};
    m.attr("combine_device_timing_fields") = m.attr("dispatch_device_timing_fields");
    m.def("begin_dispatch_host_timing", &ascend_deepep::ops::BeginDispatchHostTiming);
    m.def("end_dispatch_host_timing",
          [] { ascend_deepep::ops::MarkDispatchHostTiming(ascend_deepep::ops::ApiReturn); });
    m.def("complete_dispatch_host_timing",
          [] { ascend_deepep::ops::MarkDispatchHostTiming(ascend_deepep::ops::Complete); });
    m.def("take_dispatch_host_timings", &ascend_deepep::ops::TakeDispatchHostTimings);
    m.attr("dispatch_host_timing_fields") =
        std::vector<std::string>{"api_entry",         "native_entry",   "notify_submit_begin", "notify_submitted",
                                 "count_wait_begin",  "total_observed", "counts_observed",     "output_ready",
                                 "comm_submit_begin", "comm_submitted", "native_return",       "api_return",
                                 "complete",          "fused",          "host_prep",           "deferred_plan",
                                 "mapped_counts"};
    // Keep the Python dtype contract tied to the native V1 dispatch path.
    m.attr("topk_idx_t") = pybind11::cast(c10::CppTypeToScalarType<ascend_deepep::topk_idx_t>::value);
}
