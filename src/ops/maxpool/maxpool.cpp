#include "ops/maxpool/maxpool.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace rvnpu {

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
  if (!read_status.IsOK())
    throw std::runtime_error(std::string("invalid attribute ") + name);
  return result;
}

template <typename T>
std::vector<T> NodeAttributeArray(const Ort::ConstNode &node,
                                  const char *name,
                                  std::vector<T> default_value) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName(name, attribute);
  if (!status.IsOK()) {
    throw std::runtime_error(std::string("cannot read attribute ") + name +
                             ": " + status.GetErrorMessage());
  }
  if (attribute == nullptr)
    return default_value;
  std::vector<T> result;
  const Ort::Status read_status = attribute.GetValueArray<T>(result);
  if (!read_status.IsOK())
    throw std::runtime_error(std::string("invalid attribute ") + name);
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
                                    std::vector<T> default_value) {
  try {
    return Ort::ConstKernelInfo{info}.GetAttributes<T>(name);
  } catch (const Ort::Exception &) {
    return default_value;
  }
}

MaxPoolAttributes ReadMaxPoolAttributes(const OrtKernelInfo *info) {
  MaxPoolAttributes result;
  result.dilations =
      KernelAttributeArray<int64_t>(info, "dilations", {1, 1});
  result.kernel_shape =
      KernelAttributeArray<int64_t>(info, "kernel_shape", {});
  result.pads = KernelAttributeArray<int64_t>(info, "pads", {0, 0, 0, 0});
  result.strides = KernelAttributeArray<int64_t>(info, "strides", {1, 1});
  result.ceil_mode = KernelAttribute<int64_t>(info, "ceil_mode", 0);
  result.storage_order = KernelAttribute<int64_t>(info, "storage_order", 0);
  result.auto_pad =
      KernelAttribute<std::string>(info, "auto_pad", "NOTSET");
  return result;
}

void ValidateFloat(const Ort::ConstValue &value) {
  if (value.GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::invalid_argument("RVNPU MaxPool requires FP32 tensors");
  }
}

OrtStatus *MissingContext() {
  return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                    "RVNPU MaxPool has no DeviceContext");
}

} // namespace

MaxPoolAttributes ReadMaxPoolAttributes(const Ort::ConstNode &node) {
  MaxPoolAttributes result;
  result.dilations =
      NodeAttributeArray<int64_t>(node, "dilations", {1, 1});
  result.kernel_shape =
      NodeAttributeArray<int64_t>(node, "kernel_shape", {});
  result.pads = NodeAttributeArray<int64_t>(node, "pads", {0, 0, 0, 0});
  result.strides = NodeAttributeArray<int64_t>(node, "strides", {1, 1});
  result.ceil_mode = NodeAttribute<int64_t>(node, "ceil_mode", 0);
  result.storage_order = NodeAttribute<int64_t>(node, "storage_order", 0);
  result.auto_pad =
      NodeAttribute<std::string>(node, "auto_pad", "NOTSET");
  return result;
}

RVNPU_DEFINE_KERNEL(MaxPool, kOnnxDomain, 12,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    MaxPoolKernel)

MaxPoolKernel::MaxPoolKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, attributes_(ReadMaxPoolAttributes(info)),
      context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *MaxPoolKernel::Create(const OrtKernelInfo *info, void *state,
                                 OrtKernelImpl *&output) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return MissingContext();
  output = new MaxPoolKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL MaxPoolKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<MaxPoolKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  ValidateFloat(input);
  const auto shape = ValidateMaxPoolShape(
      input.GetTensorTypeAndShapeInfo().GetShape(), self.attributes_,
      kernel_context.GetOutputCount());
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunMaxPool(input.GetTensorData<float>(),
                            output.GetTensorMutableData<float>(), shape);
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

void ORT_API_CALL MaxPoolKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<MaxPoolKernel *>(kernel);
}

} // namespace rvnpu
