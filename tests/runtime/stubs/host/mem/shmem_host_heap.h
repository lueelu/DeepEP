// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <cstddef>
void* aclshmem_malloc(size_t);
void* aclshmem_align(size_t, size_t);
void aclshmem_free(void*);
