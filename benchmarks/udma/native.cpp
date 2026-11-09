#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "shmem.h"
#include "common/check.h"
#include "common/udma_schedule.h"
#include "runtime/shmem_instance.h"
#include "contract.h"

namespace py = pybind11;
using namespace udma_bench;

class Session {
public:
    Session(int rank, int world, uint64_t capacity, int cores, int peers, int visit, const std::vector<int64_t>& uid,
            bool huge_pages)
    {
        namespace u = ascend_deepep::udma;
        if (world < 2 || world > 256 || rank < 0 || rank >= world || capacity == 0 || capacity > (1ULL << 30) ||
            capacity % 32 || cores < 1 || cores > 64 || !u::ValidPeers(peers) || !u::ValidVisit(visit) ||
            u::Workers(world, peers) > uint32_t(std::min(cores, 32)))
            throw std::invalid_argument("invalid world/rank/capacity/core/peer/visit configuration");
        // Bounds above keep every product well below int64_t before narrowing.
        config_ = {
            capacity, 0, 0, 0, uint32_t(rank), uint32_t(world), uint32_t(cores), uint32_t(peers), uint32_t(visit),
            0,        0, 0, 0};
        bytes_ = RecvOffset(config_) + uint64_t(world) * capacity;
        // Match global Clos: MTE supplies synchronization, UDMA carries payload.
        shmem_ = std::make_unique<ShmemInstance>(rank, world, "", ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA,
                                                 huge_pages, uid, "udma-microbench", true);
        try {
            if (huge_pages) {
                shmem_->Init(2LL << 20, bytes_);
                workspace_ = shmem_->WorkspacePtr();
            } else {
                constexpr uint64_t page = 2ULL << 20;
                shmem_->Init((bytes_ + page + page - 1) / page * page);
                workspace_ = aclshmem_malloc(bytes_);
            }
            if (!workspace_) throw std::runtime_error("SHMEM workspace allocation failed");
            ASCEND_DEEPEP_CHECK_ACL(aclrtCreateStream(&stream_));
            ASCEND_DEEPEP_CHECK_ACL(aclrtMalloc(&times_, cores * 64, ACL_MEM_MALLOC_HUGE_FIRST));
            ASCEND_DEEPEP_CHECK_ACL(aclrtMalloc(&errors_, cores * kLanes * 8, ACL_MEM_MALLOC_HUGE_FIRST));
            shmem_->Barrier();
        } catch (...) {
            Release();
            throw;
        }
    }

    ~Session()
    {
        Release();
    }

    py::dict Sample(int mode, int source, int destination, int delta, uint64_t bytes, uint64_t chunk, uint64_t epoch,
                    bool corrupt)
    {
        if (!workspace_) throw std::runtime_error("session is closed");
        if (mode < 0 || mode > 2 || !bytes || bytes > config_.capacity || bytes % 32 || !chunk ||
            chunk > (1ULL << 20) || chunk % 32 || !epoch ||
            (mode == 0 && (source < 0 || destination < 0 || source >= int(config_.world) ||
                           destination >= int(config_.world) || source == destination)) ||
            (mode == 1 && (delta < 1 || delta >= int(config_.world))))
            throw std::invalid_argument("invalid case, byte/chunk alignment or epoch");
        Config c = config_;
        c.mode = mode;
        c.source = source;
        c.destination = destination;
        c.delta = delta;
        c.bytes = bytes;
        c.chunk = chunk;
        c.epoch = epoch;
        std::vector<uint64_t> times(c.cores * 8), errors(c.cores * kLanes);
        {
            py::gil_scoped_release release;
            // These operations and the post-transfer verification are not timed.
            ASCEND_DEEPEP_CHECK_ACL(aclrtMemset(errors_, errors.size() * 8, 0, errors.size() * 8));
            udma_bench_prepare(stream_, workspace_, c, shmem_->FftsAddr());
            udma_bench_transfer(stream_, workspace_, times_, c, shmem_->FftsAddr());
            ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeStream(stream_));
            if (corrupt) {
                for (uint32_t src = 0; src < c.world; ++src) {
                    if (!Edge(c, src, c.rank)) continue;
                    auto* word = static_cast<uint8_t*>(workspace_) + RecvOffset(c) + src * c.capacity;
                    uint64_t value;
                    ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpy(&value, 8, word, 8, ACL_MEMCPY_DEVICE_TO_HOST));
                    value ^= 1U;
                    ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpy(word, 8, &value, 8, ACL_MEMCPY_HOST_TO_DEVICE));
                    break;
                }
            }
            udma_bench_verify(stream_, workspace_, errors_, c, shmem_->FftsAddr());
            ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeStream(stream_));
            ASCEND_DEEPEP_CHECK_ACL(
                aclrtMemcpy(times.data(), times.size() * 8, times_, times.size() * 8, ACL_MEMCPY_DEVICE_TO_HOST));
            ASCEND_DEEPEP_CHECK_ACL(
                aclrtMemcpy(errors.data(), errors.size() * 8, errors_, errors.size() * 8, ACL_MEMCPY_DEVICE_TO_HOST));
            // Prevent the next sample overwriting a peer's still-verified receive slots.
            shmem_->Barrier();
        }
        py::dict result;
        result["ticks"] = times;
        result["mismatched_words"] = std::accumulate(errors.begin(), errors.end(), uint64_t{0});
        result["epoch"] = epoch;
        return result;
    }

    void Close()
    {
        if (!workspace_) return;
        py::gil_scoped_release release;
        ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeStream(stream_));
        shmem_->Barrier();
        FreeBuffers();
        shmem_->Destroy();
        shmem_.reset();
    }

private:
    void FreeBuffers() noexcept
    {
        if (stream_) aclrtDestroyStream(stream_);
        if (times_) aclrtFree(times_);
        if (errors_) aclrtFree(errors_);
        if (workspace_ && shmem_ && !shmem_->Use1gHugePages()) aclshmem_free(workspace_);
        stream_ = nullptr;
        times_ = errors_ = workspace_ = nullptr;
    }
    void Release() noexcept
    {
        // Explicit Close is the normal collective shutdown. On failure, torchrun
        // terminates the job; do not introduce another host barrier here.
        if (stream_) aclrtSynchronizeStream(stream_);
        FreeBuffers();
        shmem_.reset();
    }
    Config config_{};
    uint64_t bytes_ = 0;
    std::unique_ptr<ShmemInstance> shmem_;
    aclrtStream stream_ = nullptr;
    void *workspace_ = nullptr, *times_ = nullptr, *errors_ = nullptr;
};

PYBIND11_MODULE(_udma_microbench, m)
{
    m.def("unique_id", &GetShmemUniqueId);
    py::class_<Session>(m, "Session")
        .def(py::init<int, int, uint64_t, int, int, int, const std::vector<int64_t>&, bool>())
        .def("sample", &Session::Sample, py::arg("mode"), py::arg("source"), py::arg("destination"), py::arg("delta"),
             py::arg("bytes"), py::arg("chunk"), py::arg("epoch"), py::arg("corrupt") = false)
        .def("close", &Session::Close);
}
