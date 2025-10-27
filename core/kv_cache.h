#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include "core/tensor.h"


namespace ni {


struct KVPage {
Tensor k; // [heads, page, head_dim]
Tensor v; // [heads, page, head_dim]
};


struct KVHandle {
// 每层一个 page 列表
std::vector<std::vector<KVPage>> layers;
int page_size {16};
};


class KVCacheManager {
public:
KVCacheManager(int n_layers, int heads, int head_dim, int page_size, Device dev);
std::shared_ptr<KVHandle> create_handle();
void append_tokens(std::shared_ptr<KVHandle>& h, int layer, const Tensor& k, const Tensor& v);
private:
int n_layers_, heads_, head_dim_, page_size_;
Device dev_;
};


} // namespace ni