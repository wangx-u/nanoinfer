#include "backends/backend.h"
#ifdef NI_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>


namespace ni {


class CUDABackend final : public IBackend {
public:
explicit CUDABackend(int dev): dev_(dev) {
cudaSetDevice(dev_);
cublasCreate(&handle_);
cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH);
}
~CUDABackend(){ cublasDestroy(handle_); }
Device device() const override { return Device::cuda(dev_); }


Tensor matmul(const Tensor& A, const Tensor& B) override {
// 占位：未实现（返回 device 上的空张量）
return Tensor::empty({A.desc().shape[0], B.desc().shape[1]}, DType::F16, Device::cuda(dev_));
}
void rmsnorm(Tensor&, const Tensor&, const Tensor&, float) override {}
void rope_inplace(Tensor&, Tensor&, float, int) override {}
void softmax_inplace(Tensor&) override {}
private:
int dev_ {0};
cublasHandle_t handle_ {nullptr};
};


std::shared_ptr<IBackend> MakeCUDABackend(int device){ return std::make_shared<CUDABackend>(device); }


} // namespace ni
#endif