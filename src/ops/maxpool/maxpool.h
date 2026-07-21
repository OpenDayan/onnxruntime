#pragma once

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace rvnpu {

MaxPoolAttributes ReadMaxPoolAttributes(const Ort::ConstNode &node);

class MaxPoolKernel final : public OrtKernelImpl {
public:
  MaxPoolKernel(const OrtKernelInfo *info, DeviceContext *context);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                              OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  MaxPoolAttributes attributes_;
  DeviceContext *context_{};
};

} // namespace rvnpu
