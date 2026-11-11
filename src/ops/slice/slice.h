#pragma once

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace dayan {

class SliceKernel final : public OrtKernelImpl {
public:
  SliceKernel(DeviceContext *);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                             OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  DeviceContext *context_{};
};

} // namespace dayan
