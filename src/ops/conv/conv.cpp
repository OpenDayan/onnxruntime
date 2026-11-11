#include "ops/conv/conv.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace dayan {

namespace {

template <typename T>
T NodeAttribute(const Ort::ConstNode &node, const char *name,
                T default_value) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName(name, attribute);
  if (!status.IsOK()) {
    throw std::runtime_error(std::string("cannot read attribute ") + name +
                             ": " + status.GetErrorMessage());
  }
  if (attribute == nullptr)
    return default_value;
  T result;
  const Ort::Status read_status = attribute.GetValue<T>(result);
  if (!read_status.IsOK()) {
    throw ConvModelError(std::string("invalid attribute ") + name);
  }
  return result;
}

template <typename T>
std::vector<T> NodeAttributeArray(const Ort::ConstNode &node,
                                  const char *name,
                                  std::vector<T> default_value,
                                  bool *provided = nullptr) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName(name, attribute);
  if (!status.IsOK()) {
    throw std::runtime_error(std::string("cannot read attribute ") + name +
                             ": " + status.GetErrorMessage());
  }
  if (provided)
    *provided = attribute != nullptr;
  if (attribute == nullptr)
    return default_value;
  std::vector<T> result;
  const Ort::Status read_status = attribute.GetValueArray<T>(result);
  if (!read_status.IsOK()) {
    throw ConvModelError(std::string("invalid attribute ") + name);
  }
  return result;
}

template <typename T>
T KernelAttribute(const OrtKernelInfo *info, const char *name,
                  T default_value) {
  try {
    return Ort::ConstKernelInfo{info}.GetAttribute<T>(name);
  } catch (const Ort::Exception &) {
    return default_value;
  }
}

template <typename T>
std::vector<T> KernelAttributeArray(const OrtKernelInfo *info,
                                    const char *name,
                                    std::vector<T> default_value,
                                    bool *provided = nullptr) {
  try {
    auto result = Ort::ConstKernelInfo{info}.GetAttributes<T>(name);
    if (provided)
      *provided = true;
    return result;
  } catch (const Ort::Exception &) {
    if (provided)
      *provided = false;
    return default_value;
  }
}

ConvAttributes ReadConvAttributes(const OrtKernelInfo *info) {
  ConvAttributes result;
  result.dilations =
      KernelAttributeArray<int64_t>(info, "dilations", {1, 1});
  result.kernel_shape =
      KernelAttributeArray<int64_t>(info, "kernel_shape", {});
  result.pads = KernelAttributeArray<int64_t>(
      info, "pads", {0, 0, 0, 0}, &result.pads_provided);
  result.strides = KernelAttributeArray<int64_t>(info, "strides", {1, 1});
  result.group = KernelAttribute<int64_t>(info, "group", 1);
  result.auto_pad =
      KernelAttribute<std::string>(info, "auto_pad", "NOTSET");
  ValidateConvAttributeValues(result);
  return result;
}

void ValidateFloat(const Ort::ConstValue &value, const char *operator_name) {
  if (value.GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::invalid_argument(std::string("Dayan ") + operator_name +
                                " requires FP32 tensors");
  }
}

OrtStatus *MissingContext(const char *operator_name) {
  const std::string message =
      std::string("Dayan ") + operator_name + " has no DeviceContext";
  return Ort::GetApi().CreateStatus(ORT_EP_FAIL, message.c_str());
}

} // namespace

ConvAttributes ReadConvAttributes(const Ort::ConstNode &node) {
  ConvAttributes result;
  result.dilations =
      NodeAttributeArray<int64_t>(node, "dilations", {1, 1});
  result.kernel_shape =
      NodeAttributeArray<int64_t>(node, "kernel_shape", {});
  result.pads = NodeAttributeArray<int64_t>(
      node, "pads", {0, 0, 0, 0}, &result.pads_provided);
  result.strides = NodeAttributeArray<int64_t>(node, "strides", {1, 1});
  result.group = NodeAttribute<int64_t>(node, "group", 1);
  result.auto_pad =
      NodeAttribute<std::string>(node, "auto_pad", "NOTSET");
  ValidateConvAttributeValues(result);
  return result;
}

DAYAN_DEFINE_KERNEL(Conv, kOnnxDomain, 11,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    ConvKernel)

ConvKernel::ConvKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, attributes_(ReadConvAttributes(info)),
      context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *ConvKernel::Create(const OrtKernelInfo *info, void *state,
                              OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext("Conv");
  output = new ConvKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL ConvKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<ConvKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const Ort::ConstValue weight = kernel_context.GetInput(1);
  ValidateFloat(input, "Conv");
  ValidateFloat(weight, "Conv");
  const Ort::ConstValue bias =
      kernel_context.GetInputCount() == 3 ? kernel_context.GetInput(2)
                                          : Ort::ConstValue{nullptr};
  if (bias != nullptr)
    ValidateFloat(bias, "Conv");
  const auto input_shape = input.GetTensorTypeAndShapeInfo().GetShape();
  const auto weight_shape = weight.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> bias_shape;
  if (bias != nullptr)
    bias_shape = bias.GetTensorTypeAndShapeInfo().GetShape();
  const auto shape = ValidateConvShapes(
      input_shape, weight_shape, bias == nullptr ? nullptr : &bias_shape,
      self.attributes_);
  Ort::UnownedValue output =
      kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunConv(
      input.GetTensorData<float>(), weight.GetTensorData<float>(),
      bias == nullptr ? nullptr : bias.GetTensorData<float>(),
      output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL ConvKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<ConvKernel *>(kernel);
}

} // namespace dayan
