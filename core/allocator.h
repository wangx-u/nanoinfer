#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "core/tensor.h"


namespace ni {


// 简化 Arena 分配器占位（后续可换成块分配 + 重用）
class TempBufferPool {
public:
explicit TempBufferPool(Device dev): dev_(dev) {}
Tensor get(size_t bytes);
private:
Device dev_;
};


} // namespace ni