#include "plugin/kernel_registry.h"

#include <array>

#include "ops/matmul/matmul.h"

namespace rvnpu {

const OrtDataType *TensorType(ONNXTensorElementDataType type) {
  const OrtDataType *result = nullptr;
  Ort::ThrowOnError(Ort::GetEpApi().GetTensorDataType(type, &result));
  return result;
}

static constexpr std::array<BuildKernelCreateInfoFn, 1> kBuilders = {
    BuildKernelCreateInfo<class RVNPU_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        MatMul)>,
};

size_t GetKernelCount() { return kBuilders.size(); }

OrtStatus *CreateKernelRegistry(const char *ep_name, void *kernel_state,
                                OrtKernelRegistry **registry) {
  *registry = nullptr;
  RVNPU_EXCEPTION_GUARD_BEGIN
  Ort::KernelRegistry result;
  for (const auto builder : kBuilders) {
    KernelCreateInfo info;
    RVNPU_RETURN_IF_ERROR(builder(ep_name, kernel_state, &info));
    RVNPU_RETURN_IF_ERROR(
        result.AddKernel(info.definition, info.create, info.state));
  }
  *registry = result.release();
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

} // namespace rvnpu
