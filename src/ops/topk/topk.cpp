#include "ops/topk/topk.h"

#include <stdexcept>
#include <string>

namespace rvnpu {

namespace {

int64_t NodeIntAttribute(const Ort::ConstNode &node, const char *name,
                         int64_t default_value) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName(name, attribute);
  if (!status.IsOK())
    throw std::runtime_error(std::string("cannot read TopK attribute ") + name);
  if (attribute == nullptr)
    return default_value;
  int64_t result = 0;
  const Ort::Status read_status = attribute.GetValue<int64_t>(result);
  if (!read_status.IsOK())
    throw std::runtime_error(std::string("invalid TopK attribute ") + name);
  return result;
}

int64_t KernelIntAttribute(const OrtKernelInfo *info, const char *name,
                           int64_t default_value) {
  try {
    return Ort::ConstKernelInfo{info}.GetAttribute<int64_t>(name);
  } catch (const Ort::Exception &) {
    return default_value;
  }
}

TopKAttributes ReadTopKAttributes(const OrtKernelInfo *info) {
  TopKAttributes result;
  result.axis = KernelIntAttribute(info, "axis", -1);
  result.largest = KernelIntAttribute(info, "largest", 1);
  result.sorted = KernelIntAttribute(info, "sorted", 1);
  return result;
}

} // namespace

TopKAttributes ReadTopKAttributes(const Ort::ConstNode &node) {
  TopKAttributes result;
  result.axis = NodeIntAttribute(node, "axis", -1);
  result.largest = NodeIntAttribute(node, "largest", 1);
  result.sorted = NodeIntAttribute(node, "sorted", 1);
  return result;
}

RVNPU_DEFINE_KERNEL(TopK, kOnnxDomain, 11,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    TopKKernel)

TopKKernel::TopKKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, attributes_(ReadTopKAttributes(info)),
      context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *TopKKernel::Create(const OrtKernelInfo *info, void *state,
                              OrtKernelImpl *&output) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  if (state == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "RVNPU TopK has no DeviceContext");
  }
  output = new TopKKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL TopKKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<TopKKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const Ort::ConstValue k_value = kernel_context.GetInput(1);
  const auto input_info = input.GetTensorTypeAndShapeInfo();
  const auto k_info = k_value.GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      k_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      k_info.GetElementCount() != 1) {
    throw std::invalid_argument(
        "RVNPU TopK requires FP32 input and one INT64 K value");
  }
  const auto shape = ValidateTopKShape(
      input_info.GetShape(), *k_value.GetTensorData<int64_t>(),
      self.attributes_);
  Ort::UnownedValue values =
      kernel_context.GetOutput(0, shape.output_shape);
  Ort::UnownedValue indices =
      kernel_context.GetOutput(1, shape.output_shape);
  self.context_->RunTopK(input.GetTensorData<float>(),
                         values.GetTensorMutableData<float>(),
                         indices.GetTensorMutableData<int64_t>(), shape);
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

void ORT_API_CALL TopKKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<TopKKernel *>(kernel);
}

} // namespace rvnpu
