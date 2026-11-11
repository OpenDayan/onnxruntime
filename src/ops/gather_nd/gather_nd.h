#pragma once

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace dayan {

int64_t ReadGatherNdBatchDims(const Ort::ConstNode &node);

class GatherNdKernel final : public OrtKernelImpl {
public:
  GatherNdKernel(const OrtKernelInfo *, DeviceContext *);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                             OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  int64_t batch_dims_{};
  DeviceContext *context_{};
};

} // namespace dayan
