#include "ops/split/split.h"

#include <stdexcept>
#include <vector>

namespace dayan {
namespace {

int64_t KernelAxis(const OrtKernelInfo *info) {
  try {
    return Ort::ConstKernelInfo{info}.GetAttribute<int64_t>("axis");
  } catch (const Ort::Exception &) {
    return 0;
  }
}

} // namespace

int64_t ReadSplitAxis(const Ort::ConstNode &node) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName("axis", attribute);
  if (!status.IsOK())
    throw std::invalid_argument("cannot read Dayan Split axis");
  if (attribute == nullptr)
    return 0;
  int64_t axis = 0;
  const Ort::Status read_status = attribute.GetValue<int64_t>(axis);
  if (!read_status.IsOK())
    throw std::invalid_argument("Dayan Split axis is invalid");
  return axis;
}

DAYAN_DEFINE_KERNEL(Split, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    SplitKernel)

SplitKernel::SplitKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, axis_(KernelAxis(info)), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *SplitKernel::Create(const OrtKernelInfo *info, void *state,
                               OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan Split has no DeviceContext");
  output = new SplitKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL SplitKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<SplitKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const auto input_info = input.GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::invalid_argument("Dayan Split requires FP32 input");
  if (kernel_context.GetInputCount() < 2)
    throw std::invalid_argument("Dayan Split requires explicit split sizes");
  const Ort::ConstValue split = kernel_context.GetInput(1);
  const auto split_info = split.GetTensorTypeAndShapeInfo();
  if (split_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    throw std::invalid_argument("Dayan Split sizes must be INT64");
  const size_t count = split_info.GetElementCount();
  const int64_t *split_data = split.GetTensorData<int64_t>();
  std::vector<int64_t> splits(split_data, split_data + count);
  const auto shape =
      ValidateSplitShape(input_info.GetShape(), splits, self.axis_,
                         kernel_context.GetOutputCount());
  std::vector<float *> outputs;
  outputs.reserve(shape.output_shapes.size());
  for (size_t index = 0; index < shape.output_shapes.size(); ++index) {
    outputs.push_back(
        kernel_context.GetOutput(index, shape.output_shapes[index])
            .GetTensorMutableData<float>());
  }
  self.context_->RunSplit(input.GetTensorData<float>(), outputs, shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL SplitKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<SplitKernel *>(kernel);
}

} // namespace dayan
