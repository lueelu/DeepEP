// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <array>
#include <limits>
#include <stdexcept>
#ifdef DEEPEP_BUILD_EP_GMM_FUSED
#include <memory>
#include <vector>
#endif

namespace deepep {
// The framework owns ACL initialization, device and context. Only this workspace
// and the SHMEM instance are owned here. Destruction is an explicit collective.
class Session {
public:
    Session(int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout, bool udma = false,
            bool megamoe = false);
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session() = default;  // Never start collectives from Python GC/interpreter exit.
    void barrier();
    void close();
    bool closed() const
    {
        return closed_;
    }
    uint64_t capacity() const
    {
        return bytes_;
    }
    static bool busy();
    // 仅借给可选 MegaMoE 扩展；内存仍由本 Session 独占并显式销毁。
    uintptr_t workspace_address() const;
#ifdef DEEPEP_BUILD_EP_GMM_FUSED
    static std::unique_ptr<Session> create_udma(int rank, int size, uint64_t heap_bytes, uint64_t workspace_bytes,
                                                const std::string& endpoint, const std::vector<uint8_t>& uid_bytes);
    static size_t udma_unique_id_size();
    static std::vector<uint8_t> udma_unique_id();
    uintptr_t udma_workspace_address() const;
#endif

    void* workspace() const
    {
        check_context();
        return workspace_;
    }
    int rank() const
    {
        return rank_;
    }
    int world() const
    {
        return world_;
    }
    bool peer_table() const
    {
        return peer_table_;
    }
    uint64_t next_generation()
    {
        if (generation_ == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("Dispatch generation exhausted; recreate Session");
        return ++generation_;
    }
    using DispatchControlLayout = std::array<uint64_t, 10>;
    bool dispatch_controls_match(const DispatchControlLayout& layout) const
    {
        return dispatch_controls_valid_ && dispatch_control_layout_ == layout;
    }
    void invalidate_dispatch_controls()
    {
        dispatch_controls_valid_ = false;
    }
    void remember_dispatch_controls(const DispatchControlLayout& layout)
    {
        dispatch_control_layout_ = layout;
        dispatch_controls_valid_ = true;
    }
    bool udma() const
    {
        return udma_;
    }

private:
#ifdef DEEPEP_BUILD_EP_GMM_FUSED
    Session() = default;
#endif
    void check_context() const;
    aclrtContext context_ = nullptr;
    void* workspace_ = nullptr;
    uint64_t bytes_ = 0;
    bool closed_ = true;
    bool poisoned_ = false;
    int rank_ = 0;
    int world_ = 0;
    bool udma_ = false;
    bool peer_table_ = false;
    uint64_t generation_ = 0;
    bool dispatch_controls_valid_ = false;
    DispatchControlLayout dispatch_control_layout_{};
};
}  // namespace deepep
