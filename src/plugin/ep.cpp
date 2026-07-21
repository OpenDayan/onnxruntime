#include "plugin/ep.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "plugin/ep_factory.h"
#include "plugin/kernel_registry.h"
#include "ops/nms/nms.h"
#include "ops/conv/conv.h"
#include "ops/maxpool/maxpool.h"
#include "ops/topk/topk.h"
#include "runtime/tensor_utils.h"

namespace rvnpu {

RvnpuEp::RvnpuEp(RvnpuEpFactory &factory, Config config,
                 const OrtLogger &logger)
    : OrtEp{}, factory_(factory), api_(factory.api()),
      ep_api_(factory.ep_api()), name_(factory.name()),
      config_(std::move(config)), logger_(logger),
      device_context_(std::make_shared<DeviceContext>(config_.kernel_elf)) {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  Compile = nullptr;
  ReleaseNodeComputeInfos = nullptr;
  CreateProfiler = nullptr;
  Ort::ThrowOnError(CreateKernelRegistry(name_.c_str(), device_context_.get(),
                                         &kernel_registry_));
  RVNPU_IGNORE_STATUS(api_.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_INFO,
                                             "RVNPUExecutionProvider created",
                                             __FILE__, __LINE__, __FUNCTION__));
}

RvnpuEp::~RvnpuEp() {
  if (kernel_registry_ != nullptr) {
    ep_api_.ReleaseKernelRegistry(kernel_registry_);
  }
}

const char *ORT_API_CALL RvnpuEp::GetNameImpl(const OrtEp *ep) noexcept {
  return static_cast<const RvnpuEp *>(ep)->name_.c_str();
}

bool RvnpuEp::SupportsMatMul(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "MatMul" ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 13) {
    return false;
  }
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || !IsFloatTensor(inputs[0]) ||
      !IsFloatTensor(inputs[1])) {
    return false;
  }
  const auto a = TensorShape(inputs[0]);
  const auto b = TensorShape(inputs[1]);
  if (!a || !b || a->size() != 2 || b->size() != 2 ||
      std::any_of(a->begin(), a->end(), [](int64_t d) { return d <= 0; }) ||
      std::any_of(b->begin(), b->end(), [](int64_t d) { return d <= 0; })) {
    return false;
  }
  try {
    static_cast<void>(ValidateMatMulShapes(*a, *b));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsBinaryElementwise(const Ort::ConstNode &node,
                                        const char *operator_type) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != operator_type ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 13) {
    return false;
  }
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || !IsFloatTensor(inputs[0]) ||
      !IsFloatTensor(inputs[1])) {
    return false;
  }
  const auto lhs = TensorShape(inputs[0]);
  const auto rhs = TensorShape(inputs[1]);
  if (!lhs || !rhs) {
    return false;
  }
  try {
    static_cast<void>(
        ValidateBinaryElementwiseShapes(*lhs, *rhs, operator_type));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsSigmoid(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "Sigmoid" ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 13) {
    return false;
  }
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || !IsFloatTensor(inputs[0])) {
    return false;
  }
  const auto shape = TensorShape(inputs[0]);
  if (!shape) {
    return false;
  }
  try {
    static_cast<void>(ValidateUnaryTensorShape(*shape, "Sigmoid"));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsConv(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "Conv" ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 11) {
    return false;
  }
  const auto inputs = node.GetInputs();
  if ((inputs.size() != 2 && inputs.size() != 3) ||
      !IsFloatTensor(inputs[0]) || !IsFloatTensor(inputs[1]) ||
      (inputs.size() == 3 && !IsFloatTensor(inputs[2]))) {
    return false;
  }
  const auto input = TensorShape(inputs[0]);
  const auto weight = TensorShape(inputs[1]);
  const auto bias = inputs.size() == 3 ? TensorShape(inputs[2])
                                       : std::optional<std::vector<int64_t>>{};
  if (!input || !weight || (inputs.size() == 3 && !bias)) {
    return false;
  }
  try {
    const auto attributes = ReadConvAttributes(node);
    static_cast<void>(ValidateConvShapes(
        *input, *weight, bias ? &*bias : nullptr, attributes));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsMaxPool(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "MaxPool" ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 12) {
    return false;
  }
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 ||
      !IsFloatTensor(inputs[0])) {
    return false;
  }
  const auto input = TensorShape(inputs[0]);
  if (!input) {
    return false;
  }
  try {
    const auto attributes = ReadMaxPoolAttributes(node);
    static_cast<void>(ValidateMaxPoolShape(*input, attributes, outputs.size()));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsTopK(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "TopK" ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 11) {
    return false;
  }
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 2 ||
      !IsFloatTensor(inputs[0]) ||
      !IsTensorType(inputs[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      !inputs[1].IsConstantInitializer()) {
    return false;
  }
  const auto input_shape = TensorShape(inputs[0]);
  if (!input_shape) {
    return false;
  }
  try {
    Ort::ConstValue k_value{nullptr};
    const Ort::Status status = inputs[1].GetInitializer(k_value);
    if (!status.IsOK() || k_value == nullptr) {
      return false;
    }
    const auto k_info = k_value.GetTensorTypeAndShapeInfo();
    if (k_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        k_info.GetElementCount() != 1) {
      return false;
    }
    const auto attributes = ReadTopKAttributes(node);
    static_cast<void>(ValidateTopKShape(
        *input_shape, *k_value.GetTensorData<int64_t>(), attributes));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsNms(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "NonMaxSuppression" ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 11) {
    return false;
  }
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() < 3 || inputs.size() > 5 || outputs.size() != 1 ||
      !IsFloatTensor(inputs[0]) || !IsFloatTensor(inputs[1]) ||
      !IsTensorType(inputs[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      !inputs[2].IsConstantInitializer()) {
    return false;
  }
  for (size_t index = 3; index < inputs.size(); ++index) {
    if (inputs[index] == nullptr)
      continue;
    if (!IsFloatTensor(inputs[index]))
      return false;
    const auto shape = TensorShape(inputs[index]);
    if (!shape || (!shape->empty() &&
                   !(shape->size() == 1 && (*shape)[0] == 1))) {
      return false;
    }
  }
  const auto boxes = TensorShape(inputs[0]);
  const auto scores = TensorShape(inputs[1]);
  if (!boxes || !scores) {
    return false;
  }
  try {
    Ort::ConstValue max_output_value{nullptr};
    const Ort::Status status = inputs[2].GetInitializer(max_output_value);
    if (!status.IsOK() || max_output_value == nullptr)
      return false;
    const auto max_output_info =
        max_output_value.GetTensorTypeAndShapeInfo();
    if (max_output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        max_output_info.GetElementCount() != 1) {
      return false;
    }
    const auto attributes = ReadNmsAttributes(node);
    static_cast<void>(ValidateNmsShapes(
        *boxes, *scores, *max_output_value.GetTensorData<int64_t>(),
        attributes));
    return true;
  } catch (...) {
    return false;
  }
}

OrtStatus *ORT_API_CALL
RvnpuEp::GetCapabilityImpl(OrtEp *ep_ptr, const OrtGraph *graph_ptr,
                           OrtEpGraphSupportInfo *support_info) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  auto &ep = *static_cast<RvnpuEp *>(ep_ptr);
  Ort::ConstGraph graph{graph_ptr};
  for (const auto &node : graph.GetNodes()) {
    const bool supported =
        SupportsMatMul(node) || SupportsBinaryElementwise(node, "Add") ||
        SupportsBinaryElementwise(node, "Mul") || SupportsSigmoid(node) ||
        SupportsConv(node) || SupportsMaxPool(node) || SupportsTopK(node) ||
        SupportsNms(node);
    if (!supported) {
      continue;
    }
    const OrtKernelDef *definition = nullptr;
    RVNPU_RETURN_IF_ERROR(ep.ep_api_.EpGraphSupportInfo_LookUpKernel(
        support_info, node, &definition));
    if (definition != nullptr) {
      RVNPU_RETURN_IF_ERROR(
          ep.ep_api_.EpGraphSupportInfo_AddSingleNode(support_info, node));
    }
  }
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL RvnpuEp::GetKernelRegistryImpl(
    OrtEp *ep_ptr, const OrtKernelRegistry **registry) noexcept {
  *registry = static_cast<RvnpuEp *>(ep_ptr)->kernel_registry_;
  return nullptr;
}

} // namespace rvnpu
