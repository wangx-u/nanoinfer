#pragma once
#include <string>
#include <vector>
#include <memory>
#include "core/tensor.h"


namespace ni {


struct LlamaHyper {
int n_layers{2};
int d_model{512};
int n_heads{8};
int head_dim{64};
int vocab_size{32000};
};


class IBackend; // 前置声明


class LlamaBlock {
public:
LlamaBlock(const LlamaHyper& h, std::shared_ptr<IBackend> be);
Tensor forward(const Tensor& x, /*kv*/ void* kv_layer_ctx);
private:
LlamaHyper h_;
std::shared_ptr<IBackend> be_;
// 权重占位：qkv, o, ffn 等
Tensor w_qkv_, w_o_, w_ff1_, w_ff2_;
Tensor rms_att_w_, rms_mlp_w_;
};

class LlamaModel {
  public:
  LlamaModel(const LlamaHyper& h, std::shared_ptr<IBackend> be);
  bool load_dummy(); // 占位权重
  Tensor forward_prefill(const Tensor& tok_emb);
  private:
  LlamaHyper h_;
  std::vector<LlamaBlock> blocks_;
  std::shared_ptr<IBackend> be_;
  Tensor w_emb_; // token embedding
  Tensor w_out_; // lm head
  };
  
  
  } // namespace ni