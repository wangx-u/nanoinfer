#include <cuda_fp16.h>
extern "C" __global__ void ni_rmsnorm(const half* x, half* y, const half* w, int D, float eps){
// 占位 kernel
}