#include "plugin/kernel_registry.h"

#include <array>

#include "ops/elementwise/elementwise.h"
#include "ops/matmul/matmul.h"
#include "ops/nms/nms.h"
#include "ops/conv/conv.h"
#include "ops/maxpool/maxpool.h"
#include "ops/topk/topk.h"

namespace rvnpu {

const OrtDataType *TensorType(ONNXTensorElementDataType type) {
  const OrtDataType *result = nullptr;
  Ort::ThrowOnError(Ort::GetEpApi().GetTensorDataType(type, &result));
  return result;
}

static constexpr std::array<BuildKernelCreateInfoFn, 8> kBuilders = {
    BuildKernelCreateInfo<class RVNPU_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        MatMul)>,
    BuildKernelCreateInfo<class RVNPU_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Add)>,
    BuildKernelCreateInfo<class RVNPU_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Mul)>,
    BuildKernelCreateInfo<class RVNPU_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Sigmoid)>,
    BuildKernelCreateInfo<class RVNPU_KERNEL_CLASS_NAME(kOnnxDomain, 11, 11,
                                                        Conv)>,
    BuildKernelCreateInfo<class RVNPU_KERNEL_CLASS_NAME(kOnnxDomain, 12, 12,
                                                        MaxPool)>,
    BuildKernelCreateInfo<class RVNPU_KERNEL_CLASS_NAME(kOnnxDomain, 11, 11,
                                                        TopK)>,
    BuildKernelCreateInfo<class RVNPU_KERNEL_CLASS_NAME(
        kOnnxDomain, 11, 11, NonMaxSuppression)>,
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
