// Copyright (c) 2026, Lu Lu
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include "session.hpp"
#include "launch.hpp"
#include "launch_gate.hpp"
#include "kernels/combine_layout.h"
#include "kernels/ep_memory_server_dedup_combine_tiling.h"

namespace py = pybind11;
namespace deepep::ubx {
namespace {
void check(int code, const char* operation)
{
    if (code) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
}
struct Buffer {
    uint8_t* data = nullptr;
    size_t bytes;
    explicit Buffer(size_t size) : bytes(std::max(size, size_t(32)))
    {
        check(aclrtMalloc(reinterpret_cast<void**>(&data), bytes, ACL_MEM_MALLOC_HUGE_FIRST), "UBX allocation");
    }
    Buffer(const Buffer&) = delete;
    ~Buffer()
    {
        if (data) aclrtFree(data);
    }
    void upload(const void* source, size_t size)
    {
        if (size > bytes) throw std::invalid_argument("UBX upload exceeds allocation");
        if (size) check(aclrtMemcpy(data, bytes, source, size, ACL_MEMCPY_HOST_TO_DEVICE), "UBX upload");
    }
};
struct Event {
    aclrtEvent value = nullptr;
    Event()
    {
        check(aclrtCreateEventWithFlag(&value, ACL_EVENT_TIME_LINE), "UBX timing event");
    }
    ~Event()
    {
        if (value) aclrtDestroyEvent(value);
    }
};
}  // namespace

// Prepared CPU metadata is uploaded once. A local gate keeps host submission
// gaps out of Event/kernel/Event while preserving the preceding HCCL dependency.
class Combine {
    std::shared_ptr<Session> session_;
    std::vector<std::unique_ptr<Buffer>> buffers_;
    CombineArgs args_{};
    Event start_, end_;
    size_t input_bytes_, output_bytes_;
    uint8_t* upload(const void* source, size_t size)
    {
        auto buffer = std::make_unique<Buffer>(size);
        buffer->upload(source, size);
        auto* result = buffer->data;
        buffers_.push_back(std::move(buffer));
        return result;
    }

public:
    Combine(std::shared_ptr<Session> session, unsigned bs, unsigned hidden, unsigned topk, unsigned experts,
            unsigned capacity, bool bf16, py::bytes input, py::bytes destination, py::bytes counts, py::bytes weights,
            py::bytes types, py::bytes relay, py::bytes mask)
        : session_(std::move(session))
    {
        session_->check_context();
        UbxCombine::Layout layout{};
        if (capacity < bs || capacity > 65535 ||
            !UbxCombine::BuildLayout(session_->world, bs, hidden, topk, experts, layout, session_->cores))
            throw std::invalid_argument("UBX shape exceeds the tested workspace/UB limits");
        const std::string x = input, d = destination, c = counts, w = weights, t = types, r = relay, m = mask;
        if (c.size() != size_t(experts) * 4 || w.size() != size_t(bs) * topk * 4 || m.size() != size_t(bs) * 4)
            throw std::invalid_argument("Invalid UBX counts, weights or mask shape");
        std::vector<int32_t> prefix(experts);
        std::memcpy(prefix.data(), c.data(), c.size());
        if (!std::is_sorted(prefix.begin(), prefix.end()) || prefix.front() < 0 ||
            uint64_t(prefix.back()) > layout.capacityRows ||
            prefix.back() > int(UbxCombine::kEpServerDedupCombineMaxTokenNum))
            throw std::invalid_argument("Invalid UBX expert/source prefix");
        const size_t rows = prefix.back();
        input_bytes_ = rows * hidden * 2;
        output_bytes_ = size_t(bs) * hidden * 2;
        if (x.size() != input_bytes_ || d.size() != rows * 12 || t.size() != rows * 4 || r.size() != rows * topk * 8)
            throw std::invalid_argument("Invalid UBX expanded row metadata shape");
        auto comm = session_->comm_args();
        UbxCombine::EpServerDedupCombineTilingData td{};
        td.bs = bs;
        td.h = hidden;
        td.topK = topk;
        td.moeExpertNum = experts;
        td.dtype = bf16 ? UbxDispatch::DISPATCH_DEDUP_DATA_TYPE_BFP16 : UbxDispatch::DISPATCH_DEDUP_DATA_TYPE_FP16;
        td.numMaxTokensPerRank = capacity;
        td.tilingKey = UbxDispatch::LlTilingKeyForWorld(session_->world);
        args_.comm = upload(&comm, sizeof(comm));
        args_.tiling = upload(&td, sizeof(td));
        args_.expert = upload(x.data(), x.size());
        args_.destination = upload(d.data(), d.size());
        args_.counts = upload(c.data(), c.size());
        args_.weights = upload(w.data(), w.size());
        args_.types = upload(t.data(), t.size());
        args_.relay = upload(r.data(), r.size());
        args_.mask = upload(m.data(), m.size());
        auto output = std::make_unique<Buffer>(output_bytes_);
        args_.output = output->data;
        buffers_.push_back(std::move(output));
        args_.cores = session_->cores;
    }
    double run(uintptr_t stream_address, const py::function& synchronize)
    {
        session_->check_context();
        if (!stream_address) throw std::invalid_argument("A current NPU stream is required");
        auto stream = reinterpret_cast<aclrtStream>(stream_address);
        LaunchGate gate(stream);
        gate.enqueue(
            [&] {
                if (synchronize().cast<uintptr_t>() != stream_address)
                    throw std::runtime_error("HCCL dependency must use the measured NPU stream");
            },
            [&] {
                py::gil_scoped_release release;
                check(aclrtRecordEvent(start_.value, stream), "UBX start event");
                ubx_combine_launch(stream, &args_);
                check(aclrtRecordEvent(end_.value, stream), "UBX end event");
            });
        py::gil_scoped_release release;
        gate.complete();
        float ms = 0;
        check(aclrtEventElapsedTime(&ms, start_.value, end_.value), "UBX elapsed event");
        if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid UBX Event duration");
        return double(ms) * 1000;
    }
    void set_input(py::bytes input)
    {
        session_->check_context();
        const std::string value = input;
        if (value.size() != input_bytes_) throw std::invalid_argument("UBX input size changed");
        if (input_bytes_)
            check(aclrtMemcpy(args_.expert, input_bytes_, value.data(), input_bytes_, ACL_MEMCPY_HOST_TO_DEVICE),
                  "UBX input update");
    }
    py::bytes output()
    {
        session_->check_context();
        std::string value(output_bytes_, '\0');
        check(aclrtMemcpy(value.data(), value.size(), args_.output, value.size(), ACL_MEMCPY_DEVICE_TO_HOST),
              "UBX output");
        return py::bytes(value);
    }
};
}  // namespace deepep::ubx

PYBIND11_MODULE(_ubx_C, m)
{
    using namespace deepep::ubx;
    m.attr("ranks_per_host") = UbxDispatch::kLlPhysicalRanksPerHost;
    m.def("routes", [](unsigned world, unsigned bs, unsigned topk, unsigned experts, unsigned seed) {
        if (!world || world > 256 || !bs || bs > 256 || !topk || topk > 32 || topk > experts || experts > 1024)
            throw std::invalid_argument("Invalid CPU route fixture dimensions");
        std::vector<int32_t> result(size_t(world) * bs * topk), ids(experts);
        for (unsigned rank = 0; rank < world; ++rank) {
            std::mt19937 rng(seed + rank * 1009U);
            std::iota(ids.begin(), ids.end(), 0);
            for (unsigned token = 0; token < bs; ++token) {
                std::shuffle(ids.begin(), ids.end(), rng);
                std::copy_n(ids.begin(), topk, result.begin() + (size_t(rank) * bs + token) * topk);
            }
        }
        return py::bytes(reinterpret_cast<const char*>(result.data()), result.size() * sizeof(int32_t));
    });
    py::class_<Session, std::shared_ptr<Session>>(m, "Session")
        .def(py::init<int, int, const std::string&, unsigned>())
        .def_readonly("cores", &Session::cores)
        .def("reset", &Session::reset)
        .def("close", &Session::close);
    py::class_<Combine>(m, "Combine")
        .def(py::init<std::shared_ptr<Session>, unsigned, unsigned, unsigned, unsigned, unsigned, bool, py::bytes,
                      py::bytes, py::bytes, py::bytes, py::bytes, py::bytes, py::bytes>())
        .def("run", &Combine::run, py::arg("stream"), py::arg("synchronize"))
        .def("set_input", &Combine::set_input)
        .def("output", &Combine::output);
}
