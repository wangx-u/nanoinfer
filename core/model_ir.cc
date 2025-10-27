#include "core/model_ir.h"
#include "backends/backend.h"
#include <random>


namespace ni {


LlamaBlock::LlamaBlock(const LlamaHyper& h, std::shared_ptr<IBackend> be)
: h_(h), be_(std::move(be)) {}


Tensor LlamaBlock::forward(const Tensor& x, void* /*kv_layer_ctx*/){
// 占位：直接返回 x（后续替换为 RMSNorm → Attn → MLP）
return x;
}


LlamaModel::LlamaModel(const LlamaHyper& h, std::shared_ptr<IBackend> be)
: h_(h), be_(std::move(be))
{
blocks_.reserve(h_.n_layers);
for(int i=0;i<h_.n_layers;++i) blocks_.emplace_back(h_, be_);
}


bool LlamaModel::load_dummy(){
w_emb_ = Tensor::zeros({(int64_t)h_.vocab_size, (int64_t)h_.d_model}, DType::F16, Device::cpu());
w_out_ = Tensor::zeros({(int64_t)h_.d_model, (int64_t)h_.vocab_size}, DType::F16, Device::cpu());
return true;
}


Tensor LlamaModel::forward_prefill(const Tensor& tok_emb){
Tensor x = tok_emb; // [B, T, D]
for(auto& blk: blocks_) x = blk.forward(x, nullptr);
return x; // 占位：返回最后一层隐状态
}


} // namespace ni