#include "models/llama_config.h"
#include <random>


namespace ni {


LlamaEngine::LlamaEngine(std::shared_ptr<IBackend> be): be_(std::move(be)) {
h_.n_layers=2; h_.d_model=128; h_.n_heads=4; h_.head_dim=32; h_.vocab_size=32000;
}


bool LlamaEngine::load(const std::string& /*model_dir*/){
model_ = std::make_unique<LlamaModel>(h_, be_);
return model_->load_dummy();
}


std::vector<int> LlamaEngine::generate(const std::vector<int>& prompt_ids, const GenerateParams& p){
// 占位：直接回显 prompt + 若干固定 token（用于端到端打通）
auto out = prompt_ids;
for(int i=0;i<p.max_new_tokens;++i) out.push_back(42); // dummy token
return out;
}


} // namespace ni