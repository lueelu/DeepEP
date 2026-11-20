#include "runtime/dispatch_trace.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

#include "common/dispatch_trace_layout.h"
#include "common/launch.h"
#include "common/platform.h"

namespace ascend_deepep::trace {
namespace {
constexpr size_t kSlotBytes = platform::kMaxNumAiv * kCoreBytes;
std::atomic<uint64_t> next_gid{0};

template <typename Clock>
int64_t NowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

void Check(aclError status, const char* op)
{
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(op) + ": " + std::to_string(status));
}
}  // namespace

std::shared_ptr<DispatchTrace> DispatchTrace::Open()
{
    const char* directory = std::getenv("DEEPEP_DISPATCH_TRACE_DIR");
    if (!directory || !*directory) return {};
    try {
        static std::mutex mutex;
        static std::map<aclrtContext, std::weak_ptr<DispatchTrace>> sinks;
        std::lock_guard<std::mutex> lock(mutex);
        aclrtContext context{};
        int32_t device{};
        Check(aclrtGetCurrentContext(&context), "get context");
        Check(aclrtGetDevice(&device), "get device");
        if (auto alive = sinks[context].lock()) return alive;
        auto sink = std::shared_ptr<DispatchTrace>(new DispatchTrace);
        sink->context_ = context;
        sink->Init(directory, device);
        sinks[context] = sink;
        return sink;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DeepEP dispatch trace disabled: %s\n", e.what());
        return {};
    }
}

void DispatchTrace::Init(const char* directory, int device)
{
    std::filesystem::create_directories(directory);
    char hostname[256]{};
    if (gethostname(hostname, sizeof(hostname) - 1) != 0) throw std::runtime_error("gethostname failed");
    const std::string filename = std::string("dispatch-") + hostname + "-" + std::to_string(getpid()) + "-d" +
                                 std::to_string(device) + "-" + std::to_string(NowNs<std::chrono::system_clock>()) +
                                 ".jsonl";
    log_.open(std::filesystem::path(directory) / filename);
    if (!log_) throw std::runtime_error("cannot open " + filename);
    Check(aclrtMallocHost(&host_, kSlots * kSlotBytes), "allocate trace ring");
    std::memset(host_, 0, kSlots * kSlotBytes);
    Check(aclrtHostRegister(host_, kSlots * kSlotBytes, ACL_HOST_REGISTER_MAPPED, &mapped_), "map trace ring");

    aclrtStream stream{};
    Check(aclrtCreateStream(&stream), "create calibration stream");
    const auto before = NowNs<std::chrono::steady_clock>();
    device_accessed_ = true;
    dispatch_trace_clock_do(stream, static_cast<uint8_t*>(mapped_));
    const auto synced = aclrtSynchronizeStream(stream);
    const auto after = NowNs<std::chrono::steady_clock>();
    const auto destroyed = aclrtDestroyStream(stream);
    Check(synced, "calibrate device clock");
    Check(destroyed, "destroy calibration stream");
    auto* first = static_cast<uint64_t*>(host_);
    const auto tick = __atomic_exchange_n(first, uint64_t{0}, __ATOMIC_ACQ_REL);
    if (!tick) throw std::runtime_error("calibration kernel did not publish a tick");
    const auto anchor0 = NowNs<std::chrono::steady_clock>();
    const auto wall = NowNs<std::chrono::system_clock>();
    const auto anchor1 = NowNs<std::chrono::steady_clock>();
    offset_ns_ =
        wall - (anchor0 + (anchor1 - anchor0) / 2) + before + (after - before) / 2 - static_cast<int64_t>(tick);
    log_ << "{\"event\":\"calibration\",\"device\":" << device << ",\"pid\":" << getpid() << ",\"tick_hz\":" << kTickHz
         << ",\"tick\":" << tick << ",\"offset_ns\":" << offset_ns_ << ",\"window_ns\":" << after - before << "}\n"
         << std::flush;
    if (!log_) throw std::runtime_error("cannot write " + filename);
    writer_ = std::thread(&DispatchTrace::Poll, this);
}

DispatchTrace::Ticket DispatchTrace::Reserve(int rank, uint32_t instance, uint32_t cores)
{
    const auto index = cursor_.fetch_add(1, std::memory_order_relaxed) % kSlots;
    auto& slot = slots_[index];
    auto expected = State::Free;
    if (cores == 0 || cores > platform::kMaxNumAiv ||
        !slot.state.compare_exchange_strong(expected, State::Reserved, std::memory_order_acquire)) {
        ++dropped_;
        return {};
    }
    slot.gid = next_gid.fetch_add(1, std::memory_order_relaxed);
    slot.rank = rank;
    slot.instance = instance;
    slot.cores = cores;
    slot.logged_launch = false;
    return {reinterpret_cast<uint64_t>(mapped_) + index * kSlotBytes, index};
}

void DispatchTrace::Submit(Ticket ticket, int64_t launch_ns, int status)
{
    auto& slot = slots_[ticket.index];
    slot.launch_ns = launch_ns;
    slot.status = status;
    slot.state.store(State::Submitted, std::memory_order_release);
}

bool DispatchTrace::Collect(unsigned index, bool closing)
{
    auto& slot = slots_[index];
    if (slot.state.load(std::memory_order_acquire) != State::Submitted) return false;
    bool changed = false;
    if (!slot.logged_launch) {
        log_ << "{\"event\":\"" << (slot.status ? "launch_error" : "launch") << "\",\"gid\":" << slot.gid
             << ",\"kernel\":\"v1_dispatch_udma_kernel\",\"rank\":" << slot.rank
             << ",\"local_instance\":" << slot.instance << ",\"launch_ns\":" << slot.launch_ns
             << ",\"status\":" << slot.status << "}\n";
        slot.logged_launch = true;
        changed = true;
        if (slot.status) {
            // A runtime error may refer to earlier asynchronous work. Never reuse an ambiguous slot.
            slot.state.store(State::Quarantined, std::memory_order_release);
            ++incomplete_;
            return true;
        }
    }
    auto* base = static_cast<uint8_t*>(host_) + index * kSlotBytes;
    uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0;
    for (uint32_t core = 0; core < slot.cores; ++core) {
        auto* line = base + core * kCoreBytes;
        const auto end = __atomic_load_n(reinterpret_cast<uint64_t*>(line + kLineBytes), __ATOMIC_ACQUIRE);
        const auto start = __atomic_load_n(reinterpret_cast<uint64_t*>(line), __ATOMIC_ACQUIRE);
        if (!start || !end || end < start) {
            if (closing) {
                log_ << "{\"event\":\"incomplete\",\"gid\":" << slot.gid << "}\n";
                ++incomplete_;
            }
            return changed || closing;
        }
        first = std::min(first, start);
        last = std::max(last, end);
    }
    log_ << "{\"event\":\"end\",\"gid\":" << slot.gid << ",\"kernel\":\"v1_dispatch_udma_kernel\""
         << ",\"rank\":" << slot.rank << ",\"local_instance\":" << slot.instance << ",\"aiv\":" << slot.cores
         << ",\"launch_ns\":" << slot.launch_ns << ",\"start_tick\":" << first << ",\"end_tick\":" << last
         << ",\"start_ns\":" << offset_ns_ + static_cast<int64_t>(first)
         << ",\"end_ns\":" << offset_ns_ + static_cast<int64_t>(last) << ",\"elapsed_ns\":" << last - first << "}\n";
    // Every AIV has published its final tick and no longer writes this slot.
    for (uint32_t core = 0; core < slot.cores; ++core) {
        auto* line = base + core * kCoreBytes;
        __atomic_store_n(reinterpret_cast<uint64_t*>(line), uint64_t{0}, __ATOMIC_RELEASE);
        __atomic_store_n(reinterpret_cast<uint64_t*>(line + kLineBytes), uint64_t{0}, __ATOMIC_RELEASE);
    }
    ++completed_;
    slot.state.store(State::Free, std::memory_order_release);
    return true;
}

void DispatchTrace::Poll()
{
    bool reported_error = false;
    for (;;) {
        const bool closing = stop_.load(std::memory_order_acquire);
        bool changed = false;
        for (unsigned i = 0; i < kSlots; ++i) changed |= Collect(i, closing);
        if (changed) log_.flush();
        if (!log_ && !reported_error) {
            std::fprintf(stderr, "DeepEP dispatch trace write failed\n");
            reported_error = true;
        }
        if (closing) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    log_ << "{\"event\":\"summary\",\"completed\":" << completed_ << ",\"incomplete\":" << incomplete_
         << ",\"dropped\":" << dropped_.load() << "}\n"
         << std::flush;
}

DispatchTrace::~DispatchTrace()
{
    aclrtContext previous{};
    const bool have_previous = aclrtGetCurrentContext(&previous) == ACL_SUCCESS;
    const bool current = have_previous && previous == context_;
    const bool selected = current || aclrtSetCurrentContext(context_) == ACL_SUCCESS;
    const bool safe = !device_accessed_ || (selected && aclrtSynchronizeDevice() == ACL_SUCCESS);
    stop_.store(true, std::memory_order_release);
    if (writer_.joinable()) writer_.join();
    // If pending writes cannot be ruled out, leave the mapping to process teardown.
    if (safe && selected && (!mapped_ || aclrtHostUnregister(host_) == ACL_SUCCESS)) {
        if (host_) aclrtFreeHost(host_);
    } else if (host_) {
        std::fprintf(stderr, "DeepEP dispatch trace retained an undrained host mapping\n");
    }
    if (!current && selected && have_previous) aclrtSetCurrentContext(previous);
}
}  // namespace ascend_deepep::trace
