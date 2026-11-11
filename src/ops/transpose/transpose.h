#pragma once

#include <vector>

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace dayan {

std::vector<int64_t> ReadTransposePerm(const Ort::ConstNode &node);

class TransposeKernel final : public OrtKernelImpl {
public:
  TransposeKernel(const OrtKernelInfo *, DeviceContext *);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                             OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  std::vector<int64_t> perm_;
  DeviceContext *context_{};
};

} // namespace dayan
