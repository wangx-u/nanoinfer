#pragma once
#include <string>
#include "core/model_ir.h"
#include "backends/backend.h"


namespace ni {


class LlamaEngineImpl; // pimpl


class LlamaEngine : public Engine {
public:
explicit LlamaEngine(std::shared_ptr<IBackend> be);
bool load(const std::string& model_dir) override;
std::vector<int> generate(const std::vector<int>& prompt_ids, const GenerateParams& p) override;
private:
std::shared_ptr<IBackend> be_;
LlamaHyper h_;
std::unique_ptr<LlamaModel> model_;
};


} // namespace ni