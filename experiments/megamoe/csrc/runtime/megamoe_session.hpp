// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#pragma once

#include <memory>
#include <string>
#include "runtime/session.hpp"

namespace deepep::megamoe {

// MegaMoE owns backend policy selection here.  The returned object keeps the
// stable public Session ABI used by Python; generic, Elastic and LL callers do
// not need to know about MegaMoE's QP policy.
std::unique_ptr<Session> create_runtime(int rank, int size, uint64_t bytes, const std::string& endpoint,
                                        unsigned timeout, const std::string& backend = "mte");

}  // namespace deepep::megamoe
