#pragma once

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace dayan {

struct ReduceMaxAttributes {
  std::vector<int64_t> axes;
  int64_t keepdims{1};
};

ReduceMaxAttributes ReadReduceMaxAttributes(const Ort::ConstNode &node);

class ReduceMaxKernel final : public OrtKernelImpl {
public:
  ReduceMaxKernel(const OrtKernelInfo *, DeviceContext *);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                             OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  ReduceMaxAttributes attributes_;
  DeviceContext *context_{};
};

} // namespace dayan
