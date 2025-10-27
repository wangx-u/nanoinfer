#pragma once
#include <cstdint>
#include <queue>
#include <vector>
#include <optional>
#include <functional>
#include "core/kv_cache.h"


namespace ni {


struct Request {
int id;
std::vector<int> input_ids;
int max_new_tokens{32};
float temperature{1.0f};
};


struct Batch {
std::vector<Request> reqs;
bool prefill{true};
};


class Scheduler {
public:
void submit(Request r) { q_.push(std::move(r)); }
std::optional<Batch> form_batch(size_t max_batch, int max_wait_ms);
bool empty() const { return q_.empty(); }
private:
std::queue<Request> q_;
};


} // namespace ni