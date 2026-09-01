#pragma once

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace rvnpu {

ArgMaxAttributes ReadArgMaxAttributes(const Ort::ConstNode &node);

class ArgMaxKernel final : public OrtKernelImpl {
public:
  ArgMaxKernel(const OrtKernelInfo *, DeviceContext *);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                              OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  ArgMaxAttributes attributes_;
  DeviceContext *context_{};
};

} // namespace rvnpu
