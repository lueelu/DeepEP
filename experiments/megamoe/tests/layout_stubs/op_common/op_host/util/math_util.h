// Copyright (c) 2026, Lu Lu
#pragma once
#include <type_traits>

// CPU substitutes for CANN integer arithmetic only; the layout under test is
// the production MegaMoE header, including its device address construction.
namespace Ops::Base {
template <typename T, typename U>
constexpr auto CeilDiv(T value, U divisor)
{
    using Result = std::common_type_t<T, U>;
    return (static_cast<Result>(value) + divisor - 1) / divisor;
}
// Match the device SDK's single-type signature; mixed signedness must fail.
template <typename T>
constexpr T CeilAlign(T value, T alignment)
{
    return CeilDiv(value, alignment) * alignment;
}
}  // namespace Ops::Base
