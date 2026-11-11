#include "ops/argmax/argmax.h"

#include <stdexcept>
#include <string>

namespace dayan {
namespace {

int64_t NodeIntAttribute(const Ort::ConstNode &node, const char *name,
                         int64_t default_value) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName(name, attribute);
  if (!status.IsOK())
    throw std::runtime_error(std::string("cannot read ArgMax attribute ") +
                             name);
  if (attribute == nullptr)
    return default_value;
  int64_t result = 0;
  if (!attribute.GetValue<int64_t>(result).IsOK())
    throw std::invalid_argument(std::string("invalid ArgMax attribute ") +
                                name);
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

ArgMaxAttributes ReadAttributes(const OrtKernelInfo *info) {
  ArgMaxAttributes result;
  result.axis = KernelIntAttribute(info, "axis", 0);
  result.keepdims = KernelIntAttribute(info, "keepdims", 1);
  result.select_last_index =
      KernelIntAttribute(info, "select_last_index", 0);
  return result;
}

} // namespace

ArgMaxAttributes ReadArgMaxAttributes(const Ort::ConstNode &node) {
  ArgMaxAttributes result;
  result.axis = NodeIntAttribute(node, "axis", 0);
  result.keepdims = NodeIntAttribute(node, "keepdims", 1);
  result.select_last_index =
      NodeIntAttribute(node, "select_last_index", 0);
  return result;
}

DAYAN_DEFINE_KERNEL(ArgMax, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    ArgMaxKernel)

ArgMaxKernel::ArgMaxKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, attributes_(ReadAttributes(info)), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *ArgMaxKernel::Create(const OrtKernelInfo *info, void *state,
                                OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan ArgMax has no DeviceContext");
  output = new ArgMaxKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL ArgMaxKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<ArgMaxKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const auto info = input.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::invalid_argument("Dayan ArgMax requires FP32 input");
  const auto shape = ValidateArgMaxShape(info.GetShape(), self.attributes_);
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunArgMax(input.GetTensorData<float>(),
                           output.GetTensorMutableData<int64_t>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL ArgMaxKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<ArgMaxKernel *>(kernel);
}

} // namespace dayan
