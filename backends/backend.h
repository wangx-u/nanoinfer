#pragma once
#include <memory>
#include "core/tensor.h"


namespace ni {


struct OpContext { /* streams/events 占位 */ };


class IBackend {
public:
virtual ~IBackend() = default;
virtual Device device() const = 0;


// 基础算子（占位）
virtual Tensor matmul(const Tensor& A, const Tensor& B) = 0;
virtual void rmsnorm(Tensor& y, const Tensor& x, const Tensor& w, float eps) = 0;
virtual void rope_inplace(Tensor& q, Tensor& k, float theta, int head_dim) = 0;
virtual void softmax_inplace(Tensor& x) = 0;
};


std::shared_ptr<IBackend> MakeCPUBackend();
std::shared_ptr<IBackend> MakeCUDABackend(int device=0);


} // namespace ni