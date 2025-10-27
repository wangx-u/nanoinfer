#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace ni {


struct GenerateParams {
int max_new_tokens = 64;
float temperature = 1.0f;
float top_p = 1.0f;
int top_k = 0; // 0 = disabled
int seed = 1337;
};


class Engine {
public:
virtual ~Engine() = default;
virtual bool load(const std::string& model_dir) = 0; // weights + config
virtual std::vector<int> generate(const std::vector<int>& prompt_ids,
const GenerateParams& p) = 0;
};


std::unique_ptr<Engine> CreateEngineCPU();
std::unique_ptr<Engine> CreateEngineCUDA(int device = 0);


} // namespace ni