#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "NPURuntime.h"
#include "runtime/tensor_utils.h"

namespace rvnpu {

class DeviceContext {
public:
  explicit DeviceContext(std::string kernel_elf);
  ~DeviceContext();

  DeviceContext(const DeviceContext &) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;

  void RunMatMul(const float *a, const float *b, float *y,
                 const MatMulShape &shape);
  const std::string &kernel_elf() const noexcept { return kernel_elf_; }

private:
  NPUKernelDesc *RequireKernel(const char *name);

  std::string kernel_elf_;
  NPURuntimeState runtime_{};
  bool initialized_{false};
  NPUKernelDesc *matmul_kernel_{};
  std::mutex mutex_;
};

using DeviceContextPtr = std::shared_ptr<DeviceContext>;

} // namespace rvnpu
