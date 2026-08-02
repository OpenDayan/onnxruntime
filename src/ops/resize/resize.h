#pragma once

#include <string>

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace rvnpu {

struct ResizeAttributes {
  std::string coordinate_transformation_mode;
  std::string mode;
  std::string nearest_mode;
};

ResizeAttributes ReadResizeAttributes(const Ort::ConstNode &node);
bool IsSupportedResizeAttributes(const ResizeAttributes &attributes);

class ResizeKernel final : public OrtKernelImpl {
public:
  ResizeKernel(const OrtKernelInfo *, DeviceContext *);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                             OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  ResizeAttributes attributes_;
  DeviceContext *context_{};
};

} // namespace rvnpu
