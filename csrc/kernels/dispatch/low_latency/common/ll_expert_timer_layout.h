#pragma once
#include <cstdint>
namespace LlExpertTimer {
constexpr uint64_t kMagic = 0x4c4c4531ULL;
constexpr uint32_t kSchema = 1U;
constexpr uint32_t kHeaderRows = 8U; // 512B, keeps every core independently aligned.
constexpr uint32_t kFields = 8U;
enum Stage : uint32_t { Packets, Metadata, OutputWrite, Forward, StageCount };
constexpr const char* kNames[] = {"packets", "metadata", "output_write", "forward"};
static_assert(sizeof(kNames)/sizeof(kNames[0]) == StageCount, "expert timer names");
}
