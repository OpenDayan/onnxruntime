#include "ops/elementwise/elementwise.h"

#include <vector>

#include "runtime/tensor_utils.h"

namespace rvnpu {

namespace {

OrtStatus *MissingContext(const char *name) {
  return Ort::GetApi().CreateStatus(
      ORT_EP_FAIL, (std::string("RVNPU ") + name + " has no DeviceContext").c_str());
}

void ValidateFloat(const Ort::ConstValue &value, const char *name) {
  if (value.GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::invalid_argument(std::string("RVNPU ") + name +
                                " requires FP32 tensors");
  }
}

} // namespace

RVNPU_DEFINE_KERNEL(Add, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    AddKernel)
RVNPU_DEFINE_KERNEL(Mul, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    MulKernel)
RVNPU_DEFINE_KERNEL(Sigmoid, kOnnxDomain, 13,
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
  RVNPU_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext("Add");
  output = new AddKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL AddKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
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
  RVNPU_EXCEPTION_GUARD_END
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
  RVNPU_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext("Mul");
  output = new MulKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL MulKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
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
  RVNPU_EXCEPTION_GUARD_END
}

void ORT_API_CALL MulKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<MulKernel *>(kernel);
}

SigmoidKernel::SigmoidKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, info_(info), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *SigmoidKernel::Create(const OrtKernelInfo *info, void *state,
                                 OrtKernelImpl *&output) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext("Sigmoid");
  output = new SigmoidKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL SigmoidKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
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
  RVNPU_EXCEPTION_GUARD_END
}

void ORT_API_CALL SigmoidKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<SigmoidKernel *>(kernel);
}

} // namespace rvnpu
