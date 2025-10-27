#include "core/scheduler.h"
#include <chrono>


namespace ni {


std::optional<Batch> Scheduler::form_batch(size_t max_batch, int /*max_wait_ms*/){
if (q_.empty()) return std::nullopt;
Batch b; b.prefill = true; // 简化：一律 prefill 批
while(!q_.empty() && b.reqs.size()<max_batch){ b.reqs.push_back(std::move(q_.front())); q_.pop(); }
return b;
}


} // namespace ni