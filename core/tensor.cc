#include "core/tensor.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef NI_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace ni
{

  size_t dtype_size(DType d)
  {
    switch (d)
    {
    case DType::F32:
      return 4;
    case DType::F16:
      return 2;
    case DType::BF16:
      return 2;
    case DType::I32:
      return 4;
    case DType::I8:
      return 1;
    case DType::U8:
      return 1;
    }
    return 0;
  }

  Tensor::Tensor(const TensorDesc &d) : desc_(d)
  {
    const size_t bytes = nbytes();
    if (desc_.device.type == DeviceType::CPU)
    {
      data_ = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
      if (!data_)
        throw std::bad_alloc();
      std::memset(data_, 0, bytes);
    }
    else
    {
#ifdef NI_WITH_CUDA
      cudaSetDevice(desc_.device.index);
      cudaMalloc(&data_, bytes);
      cudaMemset(data_, 0, bytes);
#else
      throw std::runtime_error("CUDA not enabled");
#endif
    }
  }

  Tensor::~Tensor()
  {
    if (!data_)
      return;
    const size_t bytes = nbytes();
    (void)bytes;
    if (desc_.device.type == DeviceType::CPU)
    {
      std::free(data_);
    }
    else
    {
#ifdef NI_WITH_CUDA
      cudaSetDevice(desc_.device.index);
      cudaFree(data_);
#endif
    }
  }

  size_t Tensor::nbytes() const
  {
    size_t elems = 1;
    for (auto s : desc_.shape)
      elems *= (size_t)s;
    return elems * dtype_size(desc_.dtype);
  }

  Tensor Tensor::zeros(const std::vector<int64_t> &shape, DType dt, Device dev)
  {
    Tensor t(TensorDesc{shape, dt, dev, true});
    if (dev.type == DeviceType::CPU)
      std::memset(t.data(), 0, t.nbytes());
#ifdef NI_WITH_CUDA
    else
      cudaMemset(t.data(), 0, t.nbytes());
#endif
    return t;
  }

  Tensor Tensor::empty(const std::vector<int64_t> &shape, DType dt, Device dev)
  {
    return Tensor(TensorDesc{shape, dt, dev, true});
  }

  Tensor Tensor::to(Device dev) const
  {
    Tensor out(TensorDesc{desc_.shape, desc_.dtype, dev, desc_.contiguous});
    const size_t bytes = nbytes();
    if (desc_.device.type == DeviceType::CPU && dev.type == DeviceType::CPU)
    {
      std::memcpy(out.data(), data_, bytes);
    }
#ifdef NI_WITH_CUDA
    else if (desc_.device.type == DeviceType::CPU && dev.type == DeviceType::CUDA)
    {
      cudaMemcpy(out.data(), data_, bytes, cudaMemcpyHostToDevice);
    }
    else if (desc_.device.type == DeviceType::CUDA && dev.type == DeviceType::CPU)
    {
      cudaMemcpy(out.data(), data_, bytes, cudaMemcpyDeviceToHost);
    }
    else
    {
      cudaMemcpy(out.data(), data_, bytes, cudaMemcpyDeviceToDevice);
    }
#else
    else
      throw std::runtime_error("CUDA not enabled");
#endif
    return out;
  }

} // namespace ni