#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <thread>
#include <acl/acl.h>

namespace ascend_deepep::trace {

class DispatchTrace {
public:
    static std::shared_ptr<DispatchTrace> Open();
    ~DispatchTrace();

    struct Ticket {
        uint64_t address = 0;
        unsigned index = 0;
    };
    Ticket Reserve(int rank, uint32_t instance, uint32_t cores);
    void Submit(Ticket ticket, int64_t launch_ns, int status);

private:
    enum class State { Free, Reserved, Submitted, Quarantined };
    struct Slot {
        std::atomic<State> state{State::Free};
        uint64_t gid{};
        int64_t launch_ns{};
        int rank{}, status{};
        uint32_t instance{}, cores{};
        bool logged_launch{};
    };
    static constexpr unsigned kSlots = 128;
    DispatchTrace() = default;
    void Init(const char* directory, int device);
    void Poll();
    bool Collect(unsigned index, bool closing);

    Slot slots_[kSlots];
    std::atomic<unsigned> cursor_{0};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<bool> stop_{false};
    std::thread writer_;
    std::ofstream log_;
    void* host_{};
    void* mapped_{};
    aclrtContext context_{};
    bool device_accessed_{};
    int64_t offset_ns_{};
    uint64_t completed_{}, incomplete_{};
};

}  // namespace ascend_deepep::trace
