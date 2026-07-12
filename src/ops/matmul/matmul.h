#pragma once

#include <memory>

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace rvnpu {

class MatMulKernel final : public OrtKernelImpl {
public:
  MatMulKernel(const OrtKernelInfo *info, DeviceContext *context);

  static OrtStatus *Create(const OrtKernelInfo *info, void *state,
                           OrtKernelImpl *&output) noexcept;
  static OrtStatus *ORT_API_CALL
  ComputeImpl(OrtKernelImpl *kernel, OrtKernelContext *context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *kernel) noexcept;

private:
  Ort::ConstKernelInfo info_;
  DeviceContext *context_;
};

} // namespace rvnpu
