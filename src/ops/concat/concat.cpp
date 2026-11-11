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

#include "ops/concat/concat.h"

#include <stdexcept>
#include <vector>

namespace dayan {
namespace {

int64_t KernelAxis(const OrtKernelInfo *info) {
  return Ort::ConstKernelInfo{info}.GetAttribute<int64_t>("axis");
}

void RequireFloat(const Ort::ConstValue &value) {
  if (value.GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::invalid_argument("Dayan Concat requires FP32 tensors");
}

} // namespace

int64_t ReadConcatAxis(const Ort::ConstNode &node) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName("axis", attribute);
  if (!status.IsOK() || attribute == nullptr)
    throw std::invalid_argument("Dayan Concat requires axis");
  int64_t axis = 0;
  const Ort::Status read_status = attribute.GetValue<int64_t>(axis);
  if (!read_status.IsOK())
    throw std::invalid_argument("Dayan Concat axis is invalid");
  return axis;
}

DAYAN_DEFINE_KERNEL(Concat, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    ConcatKernel)

ConcatKernel::ConcatKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, axis_(KernelAxis(info)), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *ConcatKernel::Create(const OrtKernelInfo *info, void *state,
                                OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan Concat has no DeviceContext");
  output = new ConcatKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL ConcatKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<ConcatKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<const float *> inputs;
  input_shapes.reserve(kernel_context.GetInputCount());
  inputs.reserve(kernel_context.GetInputCount());
  for (size_t index = 0; index < kernel_context.GetInputCount(); ++index) {
    const Ort::ConstValue input = kernel_context.GetInput(index);
    RequireFloat(input);
    input_shapes.push_back(input.GetTensorTypeAndShapeInfo().GetShape());
    inputs.push_back(input.GetTensorData<float>());
  }
  const auto shape = ValidateConcatShapes(input_shapes, self.axis_);
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunConcat(inputs, output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL ConcatKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<ConcatKernel *>(kernel);
}

} // namespace dayan
