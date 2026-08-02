#include "ops/slice/slice.h"

#include <stdexcept>

namespace rvnpu {
namespace {

int64_t ScalarInt64(const Ort::KernelContext &context, size_t index,
                    int64_t default_value, bool required) {
  if (index >= context.GetInputCount()) {
    if (required)
      throw std::invalid_argument("RVNPU Slice is missing an input");
    return default_value;
  }
  const Ort::ConstValue value = context.GetInput(index);
  if (value == nullptr) {
    if (required)
      throw std::invalid_argument("RVNPU Slice is missing an input");
    return default_value;
  }
  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      info.GetElementCount() != 1)
    throw std::invalid_argument("RVNPU Slice controls must be scalar INT64");
  return *value.GetTensorData<int64_t>();
}

} // namespace

RVNPU_DEFINE_KERNEL(
    Slice, kOnnxDomain, 13,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T",
                            TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("Tind",
                            TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))),
    SliceKernel)

SliceKernel::SliceKernel(DeviceContext *context)
    : OrtKernelImpl{}, context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *SliceKernel::Create(const OrtKernelInfo *, void *state,
                               OrtKernelImpl *&output) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "RVNPU Slice has no DeviceContext");
  output = new SliceKernel(static_cast<DeviceContext *>(state));
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL SliceKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<SliceKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const auto input_info = input.GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::invalid_argument("RVNPU Slice requires FP32 input");
  const int64_t start = ScalarInt64(kernel_context, 1, 0, true);
  const int64_t end = ScalarInt64(kernel_context, 2, 0, true);
  const int64_t axis = ScalarInt64(kernel_context, 3, 0, false);
  const int64_t step = ScalarInt64(kernel_context, 4, 1, false);
  const auto shape =
      ValidateSliceShape(input_info.GetShape(), start, end, axis, step);
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunSlice(input.GetTensorData<float>(),
                          output.GetTensorMutableData<float>(), shape);
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

void ORT_API_CALL SliceKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<SliceKernel *>(kernel);
}

} // namespace rvnpu
