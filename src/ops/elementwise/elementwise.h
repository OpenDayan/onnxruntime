#pragma once

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace rvnpu {

class AddKernel final : public OrtKernelImpl {
public:
  AddKernel(const OrtKernelInfo *info, DeviceContext *context);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                              OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  const OrtKernelInfo *info_{};
  DeviceContext *context_{};
};

class MulKernel final : public OrtKernelImpl {
public:
  MulKernel(const OrtKernelInfo *info, DeviceContext *context);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                              OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  const OrtKernelInfo *info_{};
  DeviceContext *context_{};
};

class SigmoidKernel final : public OrtKernelImpl {
public:
  SigmoidKernel(const OrtKernelInfo *info, DeviceContext *context);
  static OrtStatus *Create(const OrtKernelInfo *, void *,
                           OrtKernelImpl *&) noexcept;

private:
  static OrtStatus *ORT_API_CALL ComputeImpl(OrtKernelImpl *,
                                              OrtKernelContext *) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *) noexcept;
  const OrtKernelInfo *info_{};
  DeviceContext *context_{};
};

} // namespace rvnpu
