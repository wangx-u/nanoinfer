cmake_minimum_required(VERSION 3.22)
project(nanoinfer LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 选项
option(NI_WITH_CUDA "Build CUDA backend" ON)
option(NI_WITH_CPU "Build CPU backend" ON)


if(NI_WITH_CUDA)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
add_definitions(-DNI_WITH_CUDA)
endif()


# 头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


# 源码列表
set(NI_CORE
core/tensor.cc
core/allocator.cc
core/kv_cache.cc
core/scheduler.cc
core/model_ir.cc
io/safetensors_loader.cc
io/tokenizer_bpe.cc
models/llama_config.cc
)


if(NI_WITH_CPU)
list(APPEND NI_CORE backends/cpu/backend_cpu.cc backends/cpu/ops_cpu.cc)
add_definitions(-DNI_WITH_CPU)
endif()


if(NI_WITH_CUDA)
list(APPEND NI_CORE backends/cuda/backend_cuda.cc
backends/cuda/kernels/rmsnorm.cu
backends/cuda/kernels/rope.cu
backends/cuda/kernels/attn.cu)
endif()


add_library(nanoinfer STATIC ${NI_CORE})


if(NI_WITH_CUDA)
  target_link_libraries(nanoinfer PRIVATE CUDA::cudart CUDA::cublas CUDA::cuda_driver CUDA::curand)
  set_target_properties(nanoinfer PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()


add_executable(nanoinfer_cli cli/nanoinfer.cc)
target_link_libraries(nanoinfer_cli PRIVATE nanoinfer)


# 更严格的编译选项
if (MSVC)
  target_compile_options(nanoinfer PRIVATE /W4)
  target_compile_options(nanoinfer_cli PRIVATE /W4)
else()
  target_compile_options(nanoinfer PRIVATE -Wall -Wextra -Wno-psabi)
  target_compile_options(nanoinfer_cli PRIVATE -Wall -Wextra -Wno-psabi)
endif()