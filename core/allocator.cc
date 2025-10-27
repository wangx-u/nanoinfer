#include "core/allocator.h"


namespace ni {


Tensor TempBufferPool::get(size_t bytes){
size_t elems = (bytes + 1) / sizeof(float);
return Tensor::empty({(int64_t)elems}, DType::F32, dev_);
}


} // namespace ni