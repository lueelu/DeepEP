// Copyright (c) 2026, Lu Lu

// Verify the real timer implementation without NPU instructions or SHMEM.
#include <cassert>
#include "kernel_operator.h"
#include "../csrc_ep32/kernels/mega_moe/common/ascend_timer_v2_device.hpp"

int main()
{
    using namespace AscendTimer;
    using namespace RoceTest;
    for (int lane = 0; lane < 3; ++lane) {
        isAiv = lane != 0;
        taskRatio = isAiv ? 2 : 1;
        blockIndex = isAiv ? 4 : 2;
        subBlockIndex = lane == 2 ? 1 : 0;
        const int core = 6 + lane;
        pipelineBarriers = 0;
        std::vector<int64_t> buffer(TOTAL_BUFFER_SIZE, 0);
        AscendTimerDevice timer;
        timer.Init(reinterpret_cast<uint8_t*>(buffer.data()));
        auto at = [&](int slot, int end = 0) {
            return buffer[core * N_TIMING_COUNTER_PER_CORE_ALIGN + slot * 2 + end];
        };
        auto wave = [&](int slot) { return buffer[WAVE_ID_OFFSET + core * WAVE_ID_PER_CORE_ALIGN + slot]; };
        const auto envelope = isAiv ? AIV_MOE_STARTUP : AIC_MOE_STARTUP;
        const auto ready = isAiv ? AIV_COUNT_READY_WAIT : AIC_COUNT_READY_WAIT;
        int64_t firstEnd = 0;
        {
            AscendTimerStartupScope startup(&timer);
            {
                AscendTimerFixedScope init(&timer, isAiv ? AIV_DISPATCH_BUFFER_INIT : AIC_DISPATCH_BUFFER_INIT);
            }
            if (lane == 2) {
                AscendTimerFixedScope prepare(&timer, AIV_COUNT_TABLE_PREPARE);
                {
                    AscendTimerFixedScope counts(&timer, AIV_COUNT_METADATA_WAIT);
                }
                {
                    AscendTimerFixedScope gateways(&timer, AIV_GATEWAY_PAYLOAD_CHECK);
                    for (uint32_t source = 16; source < 18; ++source) {
                        AscendTimerGatewayWaitScope dependency(&timer, source, source - 16, 130);
                    }
                }
                {
                    AscendTimerFixedScope prefix(&timer, AIV_COUNT_PREFIX_BUILD);
                }
                {
                    AscendTimerFixedScope publish(&timer, AIV_COUNT_TABLE_PUBLISH);
                }
            } else {
                AscendTimerFixedScope countReady(&timer, ready);
            }
            assert(pipelineBarriers == 0);  // Observation never changes synchronization.
            timer.SetWave(3);
            timer.SetSlice(4, 256, 32);
            if (lane == 0) {
                timer.StartTile(AIC_GMM1, 0, 0, 32);
                firstEnd = at(envelope, 1);
                timer.BeginWait(AIC_GMM1_WAIT);
                timer.EndWait();
                timer.EndTile();
            } else {
                auto type = lane == 1 ? AIV_GMM1_PROLOGUE : AIV_DISPATCH;
                timer.StartSpan(type);
                firstEnd = at(envelope, 1);
                timer.EndSpan(type);
                // Later work cannot extend the one-shot startup envelope.
                timer.StartSpan(type);
                timer.EndSpan(type);
            }
        }
        timer.FlushDynamicCounts();
#ifdef DEEPEP_MEGAMOE_TIMER
        assert(at(envelope) > 0 && firstEnd > at(envelope) && at(envelope, 1) == firstEnd);
        assert(wave(envelope) == NO_WAVE);
        if (lane == 2) {
            assert(at(AIV_COUNT_TABLE_PREPARE) < at(AIV_COUNT_METADATA_WAIT));
            assert(at(AIV_COUNT_METADATA_WAIT, 1) < at(AIV_GATEWAY_PAYLOAD_CHECK));
            assert(at(AIV_GATEWAY_PAYLOAD_CHECK, 1) < at(AIV_COUNT_PREFIX_BUILD));
            assert(at(AIV_COUNT_PREFIX_BUILD, 1) < at(AIV_COUNT_TABLE_PUBLISH));
            assert(at(AIV_COUNT_TABLE_PUBLISH, 1) < at(AIV_COUNT_TABLE_PREPARE, 1));
            assert(at(AIV_COUNT_TABLE_PREPARE, 1) < firstEnd);
            assert(buffer[N_TIMING_COUNTER + core * DYNAMIC_ITER_PER_CORE_ALIGN + AIV_GATEWAY_PAYLOAD_WAIT] == 2);
            for (int slot = 0; slot < 2; ++slot) {
                int logical = FIXED_TIMING_COUNT + AIV_GATEWAY_PAYLOAD_WAIT * MAX_DYNAMIC_ITER + slot;
                int meta = TASK_METADATA_OFFSET +
                           ((core * DYNAMIC_TYPE_COUNT + AIV_GATEWAY_PAYLOAD_WAIT) * MAX_DYNAMIC_ITER + slot) * 2;
                assert(at(logical) >= at(AIV_GATEWAY_PAYLOAD_CHECK) &&
                       at(logical, 1) < at(AIV_GATEWAY_PAYLOAD_CHECK, 1));
                assert(buffer[meta] == ((int64_t(17 + slot) << 32) | (slot + 1)) && buffer[meta + 1] == 130);
                assert(wave(logical) == NO_WAVE);
            }
        } else {
            assert(at(ready) > at(envelope) && at(ready, 1) < firstEnd);
        }
        int type = lane == 0 ? AIC_GMM1 : (lane == 1 ? AIV_GMM1_PROLOGUE : AIV_DISPATCH);
        int logical = FIXED_TIMING_COUNT + type * MAX_DYNAMIC_ITER;
        assert(at(logical) >= firstEnd && at(logical, 1) > at(logical) && wave(logical) == 3);
        int meta = TASK_METADATA_OFFSET + (core * DYNAMIC_TYPE_COUNT + type) * MAX_DYNAMIC_ITER * 2;
        assert(buffer[meta] == ((int64_t(5) << 32) | 256));
#else
        for (auto value : buffer) {
            assert(value == 0);
        }
        assert(firstEnd == 0 && pipelineBarriers == 0);
        (void)wave;
#endif
        // No-work cores still close their envelope on scope exit.
        timer.Init(reinterpret_cast<uint8_t*>(buffer.data()));
        {
            AscendTimerStartupScope empty(&timer);
        }
#ifdef DEEPEP_MEGAMOE_TIMER
        assert(at(envelope) > firstEnd && at(envelope, 1) > at(envelope));
#endif
        auto saved = at(envelope, 1);
        timer.Init(nullptr);
        {
            AscendTimerStartupScope disabled(&timer);
            AscendTimerFixedScope fixed(&timer, ready);
            AscendTimerGatewayWaitScope dependency(&timer, 16, 0, 131);
        }
        assert(at(envelope, 1) == saved);
    }
    return 0;
}
