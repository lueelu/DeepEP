#pragma once
#include <cstdint>
#include "ll_group_policy.h"

namespace DispatchDedup {
struct LlCountAssignment {
    uint32_t first, end, remoteRank, alignedRecords, remoteRecords;
};
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
uint32_t CountRemotePeer(uint32_t rank, uint32_t world, uint32_t task)
{
    // Compact remote-rank enumeration skips this source's physical host.
    const uint32_t hostFirst = rank / kLlPhysicalRanksPerHost * kLlPhysicalRanksPerHost;
    return task < world - kLlPhysicalRanksPerHost
        ? (task < hostFirst ? task : task + kLlPhysicalRanksPerHost) : world;
}
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
LlCountAssignment CountAssignment(uint32_t rank, uint32_t world, uint32_t experts,
                                   uint32_t cores, uint32_t core, bool allCore, bool directRanks)
{
    const uint32_t countCores = allCore ? cores : cores / 2U;
    uint32_t first = 0U, end = 0U;
    if (allCore || core >= cores / 2U) {
        const uint32_t worker = allCore ? core : core - cores / 2U;
        const uint32_t base = experts / countCores, extra = experts % countCores;
        first = worker * base + (worker < extra ? worker : extra);
        end = first + base + (worker < extra ? 1U : 0U);
    }
    if (directRanks) {
        const uint32_t hostExperts = kLlPhysicalRanksPerHost * (experts / world);
        const uint32_t hostFirst = (rank / kLlPhysicalRanksPerHost) * hostExperts;
        const uint32_t hostEnd = hostFirst + hostExperts;
        first = first < hostFirst ? hostFirst : first;
        end = end > hostEnd ? hostEnd : end;
        if (end < first) end = first;
    }
    const uint32_t records = end > first ? (experts + countCores - 1U) / countCores : 0U;
    const uint32_t remote = directRanks ? CountRemotePeer(rank, world, core) : world;
    const uint32_t remoteRecords = remote < world ? (experts / world + 1U + 7U) / 8U * 8U : 0U;
    return {first, end, remote, (records + 7U) / 8U * 8U, remoteRecords};
}
}
