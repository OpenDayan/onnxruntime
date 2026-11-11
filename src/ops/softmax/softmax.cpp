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

#include "ops/softmax/softmax.h"

#include <stdexcept>

namespace dayan {
namespace {

int64_t KernelAxis(const OrtKernelInfo *info) {
  try {
    return Ort::ConstKernelInfo{info}.GetAttribute<int64_t>("axis");
  } catch (const Ort::Exception &) {
    return -1;
  }
}

} // namespace

int64_t ReadSoftmaxAxis(const Ort::ConstNode &node) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName("axis", attribute);
  if (!status.IsOK())
    throw std::invalid_argument("cannot read Dayan Softmax axis");
  if (attribute == nullptr)
    return -1;
  int64_t axis = -1;
  const Ort::Status read_status = attribute.GetValue<int64_t>(axis);
  if (!read_status.IsOK())
    throw std::invalid_argument("Dayan Softmax axis is invalid");
  return axis;
}

DAYAN_DEFINE_KERNEL(Softmax, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    SoftmaxKernel)

SoftmaxKernel::SoftmaxKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, axis_(KernelAxis(info)), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *SoftmaxKernel::Create(const OrtKernelInfo *info, void *state,
                                 OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan Softmax has no DeviceContext");
  output = new SoftmaxKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL SoftmaxKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<SoftmaxKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const auto info = input.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::invalid_argument("Dayan Softmax requires FP32 input");
  const auto shape = ValidateSoftmaxAxis1Shape(info.GetShape(), self.axis_);
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunSoftmax(input.GetTensorData<float>(),
                            output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL SoftmaxKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<SoftmaxKernel *>(kernel);
}

} // namespace dayan
