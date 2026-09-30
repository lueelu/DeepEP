// Copyright (c) 2026, Lu Lu
// Modified by nino888 2026

#pragma once
#include <cstddef>
#include <cstdint>
using aclrtContext = void*;
int aclrtGetCurrentContext(aclrtContext*);
const char* aclrtGetSocName();
int aclrtMemset(void*, size_t, int, size_t);
