// Copyright (c) 2026, Lu Lu
#pragma once

// Host-side stand-ins for the CCE keywords, mirroring
// DeepEP-Ascend/deep_ep/common/compiled.hpp. Device builds keep the
// compiler-provided keywords, so every header spells
// `constexpr __aicore__ __forceinline__` directly instead of wrapping each
// function in a project-specific macro.
#if not defined(__CCE__)
#define __aicore__
#define __forceinline__ inline
#endif
