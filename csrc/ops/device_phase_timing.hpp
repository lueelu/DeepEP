// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <acl/acl.h>
#include <array>
#include <stdexcept>
#include <vector>

namespace deepep {
// Optional device phase timing. Disabled paths create no events.
// Call sites define the four event boundaries; no barrier is inserted here.
class DevicePhaseTiming {
    std::array<aclrtEvent, 4> events_{};
    bool enabled_;
    static void check(aclError result)
    {
        if (result != 0) throw std::runtime_error("Device phase timing event failed");
    }

public:
    explicit DevicePhaseTiming(bool enabled) : enabled_(enabled)
    {
        if (!enabled_) return;
        try {
            for (auto& event : events_) check(aclrtCreateEvent(&event));
        } catch (...) {
            for (auto event : events_)
                if (event) aclrtDestroyEvent(event);
            throw;
        }
    }
    ~DevicePhaseTiming()
    {
        for (auto event : events_)
            if (event) aclrtDestroyEvent(event);
    }
    void mark(unsigned index, aclrtStream stream)
    {
        if (enabled_) check(aclrtRecordEvent(events_[index], stream));
    }
    std::vector<float> finish(aclrtStream stream)
    {
        if (!enabled_) return {};
        check(aclrtSynchronizeStream(stream));
        std::vector<float> ms(4);
        for (unsigned i = 0; i < 3; ++i) check(aclrtEventElapsedTime(&ms[i], events_[i], events_[i + 1]));
        check(aclrtEventElapsedTime(&ms[3], events_[0], events_[3]));
        return ms;
    }
};
}  // namespace deepep
