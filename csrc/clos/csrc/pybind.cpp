/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include "ascend_deepep/kernels/types.h"
#include "csrc/shmem_instance.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_shmem_unique_id", &GetShmemUniqueId);
    m.def("get_shmem_bootstrap", &GetShmemBootstrap);
    // Keep the Python dtype contract tied to the native V1 dispatch path.
    m.attr("topk_idx_t") = pybind11::cast(c10::CppTypeToScalarType<ascend_deepep::topk_idx_t>::value);
}
