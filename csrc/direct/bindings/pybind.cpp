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
    m.def("last_dispatch_channel_metadata", [] { return ascend_deepep::ops::LastDispatchChannelMetadata(); });
    m.attr("dispatch_channel_metadata_version") = 1;
    m.attr("dispatch_device_timing_version") = 1;
    m.attr("dispatch_device_timing_cycles_per_us") = ascend_deepep::kNpuSystemCyclesPerUs;
    m.attr("dispatch_device_timing_fields") = std::vector<std::string>{
        "version",        "rank",          "core",          "cached",      "payload_peers", "whole_channel",
        "src_threads",    "topk_threads",  "entry_cycles",  "init_done",   "begin_done",    "metadata_done",
        "payload_queued", "published",     "incoming_done", "quiet_done",  "sync_done",     "output_done",
        "handle_done",    "work_done",     "src_cycles",    "topk_cycles", "self_cycles",   "list_cycles",
        "list_windows",   "put_rows_calls"};
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
