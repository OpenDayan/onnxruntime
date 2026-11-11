#include "ops/nms/nms.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace dayan {

namespace {

int64_t NodeCenterPointBox(const Ort::ConstNode &node) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status =
      node.GetAttributeByName("center_point_box", attribute);
  if (!status.IsOK())
    throw std::runtime_error("cannot read NMS center_point_box");
  if (attribute == nullptr)
    return 0;
  int64_t result = 0;
  const Ort::Status read_status = attribute.GetValue<int64_t>(result);
  if (!read_status.IsOK())
    throw std::runtime_error("invalid NMS center_point_box");
  return result;
}

NmsAttributes ReadNmsAttributes(const OrtKernelInfo *info) {
  NmsAttributes result;
  try {
    result.center_point_box =
        Ort::ConstKernelInfo{info}.GetAttribute<int64_t>("center_point_box");
  } catch (const Ort::Exception &) {
    result.center_point_box = 0;
  }
  return result;
}

template <typename T>
T ScalarInput(const Ort::KernelContext &context, size_t index,
              T default_value, ONNXTensorElementDataType expected_type) {
  if (index >= context.GetInputCount())
    return default_value;
  const Ort::ConstValue value = context.GetInput(index);
  if (value == nullptr)
    return default_value;
  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != expected_type || info.GetElementCount() != 1)
    throw std::invalid_argument("Dayan NMS scalar input has invalid type");
  return *value.GetTensorData<T>();
}

} // namespace

NmsAttributes ReadNmsAttributes(const Ort::ConstNode &node) {
  NmsAttributes result;
  result.center_point_box = NodeCenterPointBox(node);
  return result;
}

DAYAN_DEFINE_KERNEL(
    NonMaxSuppression, kOnnxDomain, 11,
    (Ort::KernelDefBuilder()),
    NonMaxSuppressionKernel)

NonMaxSuppressionKernel::NonMaxSuppressionKernel(const OrtKernelInfo *info,
                                                 DeviceContext *context)
    : OrtKernelImpl{}, attributes_(ReadNmsAttributes(info)),
      context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *NonMaxSuppressionKernel::Create(const OrtKernelInfo *info,
                                           void *state,
                                           OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan NMS has no DeviceContext");
  }
  output =
      new NonMaxSuppressionKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL NonMaxSuppressionKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<NonMaxSuppressionKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue boxes = kernel_context.GetInput(0);
  const Ort::ConstValue scores = kernel_context.GetInput(1);
  const auto boxes_info = boxes.GetTensorTypeAndShapeInfo();
  const auto scores_info = scores.GetTensorTypeAndShapeInfo();
  if (boxes_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      scores_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::invalid_argument("Dayan NMS requires FP32 boxes and scores");
  }
  const int64_t max_output = ScalarInput<int64_t>(
      kernel_context, 2, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  const float iou_threshold = ScalarInput<float>(
      kernel_context, 3, 0.0f, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  const float score_threshold = ScalarInput<float>(
      kernel_context, 4, 0.0f, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  const auto shape =
      ValidateNmsShapes(boxes_info.GetShape(), scores_info.GetShape(),
                        max_output, self.attributes_);
  const std::vector<int64_t> selected = self.context_->RunNms(
      boxes.GetTensorData<float>(), scores.GetTensorData<float>(),
      iou_threshold, score_threshold, shape);
  const std::vector<int64_t> output_shape{
      static_cast<int64_t>(selected.size() / 3), 3};
  Ort::UnownedValue output = kernel_context.GetOutput(0, output_shape);
  if (!selected.empty()) {
    std::copy(selected.begin(), selected.end(),
              output.GetTensorMutableData<int64_t>());
  }
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL
NonMaxSuppressionKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<NonMaxSuppressionKernel *>(kernel);
}

} // namespace dayan
