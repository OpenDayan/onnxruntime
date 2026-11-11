#include "ops/transpose/transpose.h"

#include <stdexcept>

namespace dayan {
namespace {

std::vector<int64_t> KernelPerm(const OrtKernelInfo *info) {
  try {
    return Ort::ConstKernelInfo{info}.GetAttributes<int64_t>("perm");
  } catch (const Ort::Exception &) {
    return {};
  }
}

} // namespace

std::vector<int64_t> ReadTransposePerm(const Ort::ConstNode &node) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName("perm", attribute);
  if (!status.IsOK() || attribute == nullptr)
    return {};
  std::vector<int64_t> result;
  const Ort::Status read_status = attribute.GetValueArray<int64_t>(result);
  if (!read_status.IsOK())
    throw std::invalid_argument("Dayan Transpose perm is invalid");
  return result;
}

DAYAN_DEFINE_KERNEL(
    Transpose, kOnnxDomain, 13,
    (Ort::KernelDefBuilder().AddTypeConstraint(
        "T", std::vector<const OrtDataType *>{
                 TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
                 TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)})),
    TransposeKernel)

TransposeKernel::TransposeKernel(const OrtKernelInfo *info,
                                 DeviceContext *context)
    : OrtKernelImpl{}, perm_(KernelPerm(info)), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *TransposeKernel::Create(const OrtKernelInfo *info, void *state,
                                   OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan Transpose has no DeviceContext");
  output = new TransposeKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL TransposeKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<TransposeKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const auto input_info = input.GetTensorTypeAndShapeInfo();
  const auto input_type = input_info.GetElementType();
  if (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::invalid_argument(
        "Dayan Transpose requires FP32 or INT64 input");
  }
  const auto shape = ValidateTransposeShape(input_info.GetShape(), self.perm_);
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    self.context_->RunTranspose(input.GetTensorData<float>(),
                                output.GetTensorMutableData<float>(), shape);
  } else {
    self.context_->RunTranspose(input.GetTensorData<int64_t>(),
                                output.GetTensorMutableData<int64_t>(), shape);
  }
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL TransposeKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<TransposeKernel *>(kernel);
}

} // namespace dayan
