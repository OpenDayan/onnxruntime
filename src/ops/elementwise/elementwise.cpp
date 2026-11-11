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

#include "ops/elementwise/elementwise.h"

#include "runtime/tensor_utils.h"

namespace dayan {

namespace {

OrtStatus *MissingContext(const char *name) {
  return Ort::GetApi().CreateStatus(
      ORT_EP_FAIL,
      (std::string("Dayan ") + name + " has no DeviceContext").c_str());
}

void ValidateFloat(const Ort::ConstValue &value, const char *name) {
  if (value.GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::invalid_argument(std::string("Dayan ") + name +
                                " requires FP32 tensors");
  }
}

} // namespace

DAYAN_DEFINE_KERNEL(Add, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    AddKernel)
DAYAN_DEFINE_KERNEL(Mul, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    MulKernel)
DAYAN_DEFINE_KERNEL(Sub, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    SubKernel)
DAYAN_DEFINE_KERNEL(Div, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    DivKernel)
DAYAN_DEFINE_KERNEL(Sigmoid, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    SigmoidKernel)

AddKernel::AddKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, info_(info), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *AddKernel::Create(const OrtKernelInfo *info, void *state,
                             OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext("Add");
  output = new AddKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL AddKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<AddKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue lhs = kernel_context.GetInput(0);
  const Ort::ConstValue rhs = kernel_context.GetInput(1);
  ValidateFloat(lhs, "Add");
  ValidateFloat(rhs, "Add");
  const auto shape = ValidateBinaryElementwiseShapes(
      lhs.GetTensorTypeAndShapeInfo().GetShape(),
      rhs.GetTensorTypeAndShapeInfo().GetShape(), "Add");
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunAdd(lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
                        output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL AddKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<AddKernel *>(kernel);
}

MulKernel::MulKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, info_(info), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *MulKernel::Create(const OrtKernelInfo *info, void *state,
                             OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext("Mul");
  output = new MulKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL MulKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<MulKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue lhs = kernel_context.GetInput(0);
  const Ort::ConstValue rhs = kernel_context.GetInput(1);
  ValidateFloat(lhs, "Mul");
  ValidateFloat(rhs, "Mul");
  const auto shape = ValidateBinaryElementwiseShapes(
      lhs.GetTensorTypeAndShapeInfo().GetShape(),
      rhs.GetTensorTypeAndShapeInfo().GetShape(), "Mul");
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunMul(lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
                        output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL MulKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<MulKernel *>(kernel);
}

SubKernel::SubKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, info_(info), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *SubKernel::Create(const OrtKernelInfo *info, void *state,
                             OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext("Sub");
  output = new SubKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL SubKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<SubKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue lhs = kernel_context.GetInput(0);
  const Ort::ConstValue rhs = kernel_context.GetInput(1);
  ValidateFloat(lhs, "Sub");
  ValidateFloat(rhs, "Sub");
  const auto shape = ValidateBinaryElementwiseShapes(
      lhs.GetTensorTypeAndShapeInfo().GetShape(),
      rhs.GetTensorTypeAndShapeInfo().GetShape(), "Sub");
  if (shape.swap_inputs)
    throw std::invalid_argument("Dayan Sub requires lhs to match output shape");
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunSub(lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
                        output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL SubKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<SubKernel *>(kernel);
}

DivKernel::DivKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, info_(info), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *DivKernel::Create(const OrtKernelInfo *info, void *state,
                             OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext("Div");
  output = new DivKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL DivKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<DivKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue lhs = kernel_context.GetInput(0);
  const Ort::ConstValue rhs = kernel_context.GetInput(1);
  ValidateFloat(lhs, "Div");
  ValidateFloat(rhs, "Div");
  const auto shape = ValidateBinaryElementwiseShapes(
      lhs.GetTensorTypeAndShapeInfo().GetShape(),
      rhs.GetTensorTypeAndShapeInfo().GetShape(), "Div");
  if (shape.swap_inputs)
    throw std::invalid_argument("Dayan Div requires lhs to match output shape");
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunDiv(lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
                        output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL DivKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<DivKernel *>(kernel);
}

SigmoidKernel::SigmoidKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, info_(info), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *SigmoidKernel::Create(const OrtKernelInfo *info, void *state,
                                 OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext("Sigmoid");
  output = new SigmoidKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL SigmoidKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<SigmoidKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  ValidateFloat(input, "Sigmoid");
  const auto shape = ValidateUnaryTensorShape(
      input.GetTensorTypeAndShapeInfo().GetShape(), "Sigmoid");
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunSigmoid(input.GetTensorData<float>(),
                            output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL SigmoidKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<SigmoidKernel *>(kernel);
}

} // namespace dayan
