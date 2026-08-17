#include "ops/reduce_max/reduce_max.h"

#include <stdexcept>

namespace rvnpu {
namespace {

ReduceMaxAttributes ReadAttributes(const OrtKernelInfo *info) {
  ReduceMaxAttributes result;
  try {
    result.axes = Ort::ConstKernelInfo{info}.GetAttributes<int64_t>("axes");
  } catch (const Ort::Exception &) {
    result.axes.clear();
  }
  try {
    result.keepdims =
        Ort::ConstKernelInfo{info}.GetAttribute<int64_t>("keepdims");
  } catch (const Ort::Exception &) {
    result.keepdims = 1;
  }
  return result;
}

} // namespace

ReduceMaxAttributes ReadReduceMaxAttributes(const Ort::ConstNode &node) {
  ReduceMaxAttributes result;
  Ort::ConstOpAttr axes;
  Ort::ConstOpAttr keepdims;
  if (!node.GetAttributeByName("axes", axes).IsOK() || axes == nullptr ||
      !axes.GetValueArray<int64_t>(result.axes).IsOK())
    throw std::invalid_argument("RVNPU ReduceMax axes are invalid");
  if (!node.GetAttributeByName("keepdims", keepdims).IsOK())
    throw std::invalid_argument("cannot read RVNPU ReduceMax keepdims");
  if (keepdims != nullptr &&
      !keepdims.GetValue<int64_t>(result.keepdims).IsOK())
    throw std::invalid_argument("RVNPU ReduceMax keepdims is invalid");
  return result;
}

RVNPU_DEFINE_KERNEL(ReduceMax, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    ReduceMaxKernel)

ReduceMaxKernel::ReduceMaxKernel(const OrtKernelInfo *info,
                                 DeviceContext *context)
    : OrtKernelImpl{}, attributes_(ReadAttributes(info)), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *ReduceMaxKernel::Create(const OrtKernelInfo *info, void *state,
                                   OrtKernelImpl *&output) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "RVNPU ReduceMax has no DeviceContext");
  output = new ReduceMaxKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL ReduceMaxKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<ReduceMaxKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const auto info = input.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::invalid_argument("RVNPU ReduceMax requires FP32 input");
  const auto shape = ValidateReduceMaxShape(
      info.GetShape(), self.attributes_.axes, self.attributes_.keepdims);
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunReduceMax(input.GetTensorData<float>(),
                              output.GetTensorMutableData<float>(), shape);
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

void ORT_API_CALL ReduceMaxKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<ReduceMaxKernel *>(kernel);
}

} // namespace rvnpu
