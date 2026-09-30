# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by zhu-mingzhe71 2026

set(ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_ASC_FLAGS
    "--npu-arch=dav-3510 -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false")
add_sources("${ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_ASC_FLAGS}")
