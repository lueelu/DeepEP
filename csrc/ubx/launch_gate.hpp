// Copyright (c) 2026, Lu Lu
#pragma once
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace deepep::ubx {
// Hold the measured stream until its collective dependency and timed work are
// all queued. The release stream must not depend on the held stream.
class LaunchGate {
    aclrtNotify notify_ = nullptr;
    aclrtStream release_ = nullptr;
    aclrtStream target_;
    bool pending_ = false;
    static constexpr unsigned timeout_ms = 180000;

    static void check(int status, const char* operation)
    {
        if (status) throw std::runtime_error(operation);
    }
    [[noreturn]] static void fail(const char* operation)
    {
        std::fprintf(stderr, "UBX launch gate failed: %s; terminating worker\n", operation);
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);
    }

public:
    explicit LaunchGate(aclrtStream stream) : target_(stream)
    {
        check(aclrtCreateNotify(&notify_, 0), "create Notify");
        if (aclrtCreateStream(&release_)) {
            aclrtDestroyNotify(notify_);
            throw std::runtime_error("create Notify release stream");
        }
    }
    LaunchGate(const LaunchGate&) = delete;
    LaunchGate& operator=(const LaunchGate&) = delete;
    ~LaunchGate()
    {
        if (pending_) fail("incomplete submission");
        if (aclrtSynchronizeStreamWithTimeout(release_, timeout_ms)) fail("release completion");
        aclrtDestroyStream(release_);
        aclrtDestroyNotify(notify_);
    }
    template <class Before, class Work>
    void enqueue(Before&& before, Work&& work)
    {
        check(aclrtWaitAndResetNotify(notify_, target_, timeout_ms), "arm Notify");
        pending_ = true;
        try {
            before();
            work();
            check(aclrtRecordNotify(notify_, release_), "release Notify");
        } catch (const std::exception& e) {
            fail(e.what());
        } catch (...) {
            fail("submit collective or measured work");
        }
    }
    void complete()
    {
        if (aclrtSynchronizeStreamWithTimeout(target_, timeout_ms)) fail("Combine completion");
        pending_ = false;
    }
};
}  // namespace deepep::ubx
