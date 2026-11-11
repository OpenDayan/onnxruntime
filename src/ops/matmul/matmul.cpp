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

#include "ops/matmul/matmul.h"

#include <memory>
#include <vector>

#include "runtime/tensor_utils.h"

namespace dayan {

DAYAN_DEFINE_KERNEL(MatMul, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    MatMulKernel)

MatMulKernel::MatMulKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, info_(info), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *MatMulKernel::Create(const OrtKernelInfo *info, void *state,
                                OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan MatMul has no DeviceContext");
  }
  output = new MatMulKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL MatMulKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<MatMulKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue a = kernel_context.GetInput(0);
  const Ort::ConstValue b = kernel_context.GetInput(1);
  const auto a_info = a.GetTensorTypeAndShapeInfo();
  const auto b_info = b.GetTensorTypeAndShapeInfo();
  if (a_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      b_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Dayan MatMul requires FP32 tensors");
  }
  const auto shape = ValidateMatMulShapes(a_info.GetShape(), b_info.GetShape());
  const std::vector<int64_t> output_shape{shape.m, shape.n};
  Ort::UnownedValue output = kernel_context.GetOutput(0, output_shape);
  self.context_->RunMatMul(a.GetTensorData<float>(), b.GetTensorData<float>(),
                           output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL MatMulKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<MatMulKernel *>(kernel);
}

} // namespace dayan
