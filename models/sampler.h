#pragma once
#include <vector>
#include <random>


namespace ni {


inline int sample_greedy(const std::vector<float>& logits){
int idx=0; float best=-1e30f;
for(int i=0;i<(int)logits.size();++i) if (logits[i]>best){ best=logits[i]; idx=i; }
return idx;
}


} // namespace ni