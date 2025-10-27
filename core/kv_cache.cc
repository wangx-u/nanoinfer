#include "core/kv_cache.h"
#include <stdexcept>


namespace ni {


KVCacheManager::KVCacheManager(int n_layers, int heads, int head_dim, int page_size, Device dev)
: n_layers_(n_layers), heads_(heads), head_dim_(head_dim), page_size_(page_size), dev_(dev) {}


std::shared_ptr<KVHandle> KVCacheManager::create_handle(){
auto h = std::make_shared<KVHandle>();
h->page_size = page_size_;
h->layers.resize(n_layers_);
return h;
}


void KVCacheManager::append_tokens(std::shared_ptr<KVHandle>& h, int layer, const Tensor& k, const Tensor& v){
// 占位逻辑：当当前页满了就新开一页（真实实现需逐 token 写入）
auto& pages = h->layers.at(layer);
if (pages.empty() || pages.back().k.dim(1) >= h->page_size) {
KVPage page{Tensor::empty({(int64_t)heads_, (int64_t)h->page_size, (int64_t)head_dim_}, DType::F16, dev_),
Tensor::empty({(int64_t)heads_, (int64_t)h->page_size, (int64_t)head_dim_}, DType::F16, dev_)};
pages.push_back(std::move(page));
}
// TODO: 将 k/v 拷贝到 page 的合适位置（简化略）
}


} // namespace ni