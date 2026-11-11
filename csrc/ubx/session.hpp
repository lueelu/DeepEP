// Copyright (c) 2026, Lu Lu
#pragma once
#include <acl/acl.h>
#include <cstdint>
#include <string>
#include "kernels/comm_args.h"

namespace deepep::ubx {
class Session {
public:
    Session(int rank, int world, const std::string& endpoint, unsigned timeout);
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session() = default;
    static bool busy();
    void check_context() const;
    void close();
    void reset();
    UbxDispatch::CommArgs comm_args() const;
    int rank, world;
    uint32_t cores;
    static constexpr uint64_t capacity = 8ULL << 30;

private:
    aclrtContext context_ = nullptr;
    void* workspace_ = nullptr;
    void* stage_ = nullptr;
    bool closed_ = true;
};
}  // namespace deepep::ubx
