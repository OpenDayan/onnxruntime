#include "runtime/device_context.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/device_buffer.h"
#include "runtime/kernel_catalog.h"

namespace rvnpu {

namespace {
void RequireSuccess(ssize_t result, const std::string &stage) {
  if (result != NPU_SUCCESS) {
    throw std::runtime_error(stage +
                             " failed: runtime code=" + std::to_string(result));
  }
}

std::vector<uint8_t> ReadFile(const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open kernel ELF: " + path);
  }
  const auto end = stream.tellg();
  if (end <= 0) {
    throw std::runtime_error("kernel ELF is empty: " + path);
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char *>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("cannot read kernel ELF: " + path);
  }
  return bytes;
}
} // namespace

DeviceContext::DeviceContext(std::string kernel_elf)
    : kernel_elf_(std::move(kernel_elf)) {
  if (kernel_elf_.empty() || !std::filesystem::is_regular_file(kernel_elf_)) {
    throw std::invalid_argument(
        "kernel_elf must name an existing regular file: " + kernel_elf_);
  }

  RequireSuccess(npu_runtime_init(&runtime_), "npu_runtime_init");
  initialized_ = true;
  try {
    const auto image = ReadFile(kernel_elf_);
    RequireSuccess(npu_runtime_load_kernelset_from_memory(
                       &runtime_, image.data(), image.size()),
                   "npu_runtime_load_kernelset_from_memory");

    NPUKernelDesc *tx_init = RequireKernel(kernels::kTxInit);
    int kernel_result = -1;
    RequireSuccess(
        npu_runtime_kernel_start(&runtime_, tx_init, &kernel_result, 0),
        "__tx_init launch");
    if (kernel_result != 0) {
      throw std::runtime_error("__tx_init returned " +
                               std::to_string(kernel_result));
    }
    matmul_kernel_ = RequireKernel(kernels::kMatMulFp32);
  } catch (...) {
    npu_runtime_deinit(&runtime_);
    initialized_ = false;
    throw;
  }
}

DeviceContext::~DeviceContext() {
  if (initialized_) {
    npu_runtime_deinit(&runtime_);
  }
}

NPUKernelDesc *DeviceContext::RequireKernel(const char *name) {
  NPUKernelDesc *descriptor = nullptr;
  RequireSuccess(npu_runtime_get_kernel_desc(&runtime_, name, &descriptor),
                 std::string("kernel lookup ") + name);
  if (descriptor == nullptr || descriptor->entry == 0) {
    throw std::runtime_error(std::string("kernel has no entry: ") + name);
  }
  return descriptor;
}

void DeviceContext::RunMatMul(const float *a, const float *b, float *y,
                              const MatMulShape &shape) {
  if (a == nullptr || b == nullptr || y == nullptr) {
    throw std::invalid_argument("RVNPU MatMul received a null tensor pointer");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const size_t a_bytes = shape.a_elements * sizeof(float);
  const size_t b_bytes = shape.b_elements * sizeof(float);
  const size_t y_bytes = shape.y_elements * sizeof(float);

  DeviceBuffer device_a(runtime_, a_bytes);
  DeviceBuffer device_b(runtime_, b_bytes);
  DeviceBuffer device_y(runtime_, y_bytes);
  device_a.CopyFromHost(a, a_bytes);
  device_b.CopyFromHost(b, b_bytes);

  int kernel_result = -1;
  const uint64_t m = static_cast<uint64_t>(static_cast<uint32_t>(shape.m));
  const uint64_t n = static_cast<uint64_t>(static_cast<uint32_t>(shape.n));
  const uint64_t k = static_cast<uint64_t>(static_cast<uint32_t>(shape.k));
  const uint64_t one = 1;
  RequireSuccess(
      npu_runtime_kernel_start(&runtime_, matmul_kernel_, &kernel_result, 12,
                               device_a.address(), device_b.address(),
                               device_y.address(), m, n, k, k, one, n, one, n,
                               one),
      std::string("RVNPU MatMul launch kernel=") + kernels::kMatMulFp32 +
          " M=" + std::to_string(shape.m) + " N=" + std::to_string(shape.n) +
          " K=" + std::to_string(shape.k));
  if (kernel_result != 0) {
    throw std::runtime_error(std::string("RVNPU MatMul device kernel=") +
                             kernels::kMatMulFp32 + " returned " +
                             std::to_string(kernel_result));
  }
  device_y.CopyToHost(y, y_bytes);

  std::cout << "RVNPU MatMul M=" << shape.m << " N=" << shape.n
            << " K=" << shape.k << " device kernel=" << kernels::kMatMulFp32
            << std::endl;
}

} // namespace rvnpu
