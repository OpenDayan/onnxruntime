/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * onnxruntime is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include "plugin/kernel_registry.h"

#include <array>

#include "ops/argmax/argmax.h"
#include "ops/concat/concat.h"
#include "ops/conv/conv.h"
#include "ops/elementwise/elementwise.h"
#include "ops/gather_nd/gather_nd.h"
#include "ops/matmul/matmul.h"
#include "ops/maxpool/maxpool.h"
#include "ops/nms/nms.h"
#include "ops/reduce_max/reduce_max.h"
#include "ops/resize/resize.h"
#include "ops/slice/slice.h"
#include "ops/softmax/softmax.h"
#include "ops/split/split.h"
#include "ops/topk/topk.h"
#include "ops/transpose/transpose.h"

namespace dayan {

const OrtDataType *TensorType(ONNXTensorElementDataType type) {
  const OrtDataType *result = nullptr;
  Ort::ThrowOnError(Ort::GetEpApi().GetTensorDataType(type, &result));
  return result;
}

static constexpr std::array<BuildKernelCreateInfoFn, 19> kBuilders = {
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        MatMul)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Add)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Mul)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Sub)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Div)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Concat)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Split)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Slice)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Resize)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Transpose)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Softmax)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        ReduceMax)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        ArgMax)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        Sigmoid)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 11, 11,
                                                        Conv)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 12, 12,
                                                        MaxPool)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 11, 11,
                                                        TopK)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(
        kOnnxDomain, 11, 11, NonMaxSuppression)>,
    BuildKernelCreateInfo<class DAYAN_KERNEL_CLASS_NAME(kOnnxDomain, 13, 13,
                                                        GatherND)>,
};

size_t GetKernelCount() { return kBuilders.size(); }

OrtStatus *CreateKernelRegistry(const char *ep_name, void *kernel_state,
                                OrtKernelRegistry **registry) {
  *registry = nullptr;
  DAYAN_EXCEPTION_GUARD_BEGIN
  Ort::KernelRegistry result;
  for (const auto builder : kBuilders) {
    KernelCreateInfo info;
    DAYAN_RETURN_IF_ERROR(builder(ep_name, kernel_state, &info));
    DAYAN_RETURN_IF_ERROR(
        result.AddKernel(info.definition, info.create, info.state));
  }
  *registry = result.release();
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

} // namespace dayan
