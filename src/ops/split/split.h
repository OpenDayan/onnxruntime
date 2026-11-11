#pragma once

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace dayan {

int64_t ReadSplitAxis(const Ort::ConstNode &node);

class SplitKernel final : public OrtKernelImpl {
public:
  SplitKernel(const OrtKernelInfo *, DeviceContext *);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                             OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  int64_t axis_{};
  DeviceContext *context_{};
};

} // namespace dayan
