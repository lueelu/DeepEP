// Copyright (c) 2026, Lu Lu

// Real shmem.h exposes HNS internals only in the device compiler pass. Keep
// those declarations absent here even when the backend build macro is set.
#define ROCE_PROGRESS_TEST_HOST_PASS 1
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_roce_progress.hpp"

int main()
{
    assert(MegaMoeImpl::RoceProgressReady(1U, 1U, nullptr, 7U) == 0U);
    return 0;
}
