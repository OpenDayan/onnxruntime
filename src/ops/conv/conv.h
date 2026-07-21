#pragma once

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace rvnpu {

ConvAttributes ReadConvAttributes(const Ort::ConstNode &node);

class ConvKernel final : public OrtKernelImpl {
public:
  ConvKernel(const OrtKernelInfo *info, DeviceContext *context);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                              OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  ConvAttributes attributes_;
  DeviceContext *context_{};
};

} // namespace rvnpu
