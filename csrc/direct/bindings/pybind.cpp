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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_shmem_unique_id", &GetShmemUniqueId);
    m.def("get_shmem_bootstrap", &GetShmemBootstrap);
    // Keep the Python dtype contract tied to the native V1 dispatch path.
    // Resolve through torch instead of pybind's ScalarType caster: torch_npu may
    // not register every c10::ScalarType at import time.
    auto torch = pybind11::module_::import("torch");
    m.attr("topk_idx_t") = sizeof(ascend_deepep::topk_idx_t) == 8 ? torch.attr("int64") : torch.attr("int32");
}
