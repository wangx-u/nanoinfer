#include "backends/backend.h"
#include <stdexcept>
#include <cstring>


namespace ni {


class CPUBackend final : public IBackend {
public:
Device device() const override { return Device::cpu(); }
Tensor matmul(const Tensor& A, const Tensor& B) override {
// 朴素 GEMM：A[M,K] * B[K,N] = C[M,N]
const int64_t M=A.desc().shape[0], K=A.desc().shape[1], N=B.desc().shape[1];
Tensor C = Tensor::zeros({M,N}, DType::F32, Device::cpu());
const float* a=(const float*)A.data();
const float* b=(const float*)B.data();
float* c=(float*)C.data();
for(int64_t m=0;m<M;++m)
for(int64_t n=0;n<N;++n){
float acc=0.f;
for(int64_t k=0;k<K;++k) acc += a[m*K+k]*b[k*N+n];
c[m*N+n]=acc;
}
return C;
}
void rmsnorm(Tensor& y, const Tensor& x, const Tensor& w, float eps) override {
// 简化实现：y=x（占位）
std::memcpy(y.data(), x.data(), x.nbytes());
}
void rope_inplace(Tensor&, Tensor&, float, int) override { /* 占位 */ }
void softmax_inplace(Tensor&) override { /* 占位 */ }
};


std::shared_ptr<IBackend> MakeCPUBackend(){ return std::make_shared<CPUBackend>(); }


} // namespace ni