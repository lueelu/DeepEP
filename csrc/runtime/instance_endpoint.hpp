// Copyright (c) 2026, DeepEP contributors
#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include "host/init/shmem_host_init.h"

namespace deepep {
// The SDK requires port zero plus a range for nonzero DEFAULT instances.
// Preserve the caller's exact endpoint, temporarily encoding it in that form.
// The process-wide InstanceLock must cover this object and SDK initialization.
class InstanceEndpoint {
public:
    explicit InstanceEndpoint(aclshmemx_init_attr_t& attr)
    {
        if (attr.instance_id == 0 || attr.ip_port[0] == '\0') return;
        std::string endpoint(attr.ip_port);
        const auto colon = endpoint.rfind(':');
        if (colon == std::string::npos) throw std::invalid_argument("Invalid SHMEM endpoint");
        const auto port = std::stoul(endpoint.substr(colon + 1));
        if (port <= attr.instance_id || port > 65535)
            throw std::invalid_argument("SHMEM endpoint port cannot encode this instance ID");
        const char* previous = std::getenv("SHMEM_INSTANCE_PORT_RANGE");
        present_ = previous != nullptr;
        if (previous) previous_ = previous;
        const std::string range = std::to_string(port - attr.instance_id) + ":" + std::to_string(port);
        if (setenv("SHMEM_INSTANCE_PORT_RANGE", range.c_str(), 1) != 0)
            throw std::runtime_error("Cannot configure SHMEM instance endpoint");
        endpoint.resize(colon + 1);
        endpoint += "0";
        std::memcpy(attr.ip_port, endpoint.c_str(), endpoint.size() + 1);
        active_ = true;
    }
    ~InstanceEndpoint()
    {
        if (!active_) return;
        if (present_)
            setenv("SHMEM_INSTANCE_PORT_RANGE", previous_.c_str(), 1);
        else
            unsetenv("SHMEM_INSTANCE_PORT_RANGE");
    }
    InstanceEndpoint(const InstanceEndpoint&) = delete;
    InstanceEndpoint& operator=(const InstanceEndpoint&) = delete;

private:
    bool active_ = false;
    bool present_ = false;
    std::string previous_;
};
}  // namespace deepep
