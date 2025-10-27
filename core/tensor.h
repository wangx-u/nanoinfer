#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>


namespace ni {


enum class DeviceType { CPU, CUDA };


enum class DType { F32, F16, BF16, I32, I8, U8 };


struct Device {
DeviceType type {DeviceType::CPU};
int index {0};
static Device cpu() { return {DeviceType::CPU, 0}; }
static Device cuda(int i=0) { return {DeviceType::CUDA, i}; }
};


struct TensorDesc {
std::vector<int64_t> shape;
DType dtype {DType::F16};
Device device {Device::cpu()};
bool contiguous {true};
};

class Tensor {
  public:
  Tensor() = default;
  explicit Tensor(const TensorDesc& d);
  ~Tensor();
  
  
  void* data() { return data_; }
  const void* data() const { return data_; }
  
  
  const TensorDesc& desc() const { return desc_; }
  size_t nbytes() const;
  int64_t dim(size_t i) const { return desc_.shape.at(i); }
  size_t ndim() const { return desc_.shape.size(); }
  
  
  // 简化 API
  static Tensor zeros(const std::vector<int64_t>& shape, DType dt, Device dev);
  static Tensor empty(const std::vector<int64_t>& shape, DType dt, Device dev);
  
  
  // 设备拷贝（占位）
  Tensor to(Device dev) const;
  
  
  private:
  TensorDesc desc_{};
  void* data_ {nullptr};
  };
  
  
  size_t dtype_size(DType d);
  
  
  } // namespace ni