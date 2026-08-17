#include "plugin/ep.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "ops/concat/concat.h"
#include "ops/conv/conv.h"
#include "ops/gather_nd/gather_nd.h"
#include "ops/maxpool/maxpool.h"
#include "ops/nms/nms.h"
#include "ops/resize/resize.h"
#include "ops/reduce_max/reduce_max.h"
#include "ops/softmax/softmax.h"
#include "ops/split/split.h"
#include "ops/topk/topk.h"
#include "ops/transpose/transpose.h"
#include "plugin/ep_factory.h"
#include "plugin/kernel_registry.h"
#include "runtime/tensor_utils.h"

namespace rvnpu {

namespace {

std::optional<int64_t> ReadInt64Scalar(const Ort::ConstValue &value) {
  if (value == nullptr)
    return std::nullopt;
  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      info.GetElementCount() != 1)
    return std::nullopt;
  return *value.GetTensorData<int64_t>();
}

std::optional<int64_t> ReadConstantInt64Scalar(
    const Ort::ConstValueInfo &input) {
  Ort::ConstValue initializer{nullptr};
  if (input.GetInitializer(initializer).IsOK() && initializer != nullptr)
    return ReadInt64Scalar(initializer);

  const auto producer = input.GetProducerNode();
  if (producer.node == nullptr ||
      producer.node.GetOperatorType() != "Constant")
    return std::nullopt;
  Ort::ConstOpAttr attribute;
  if (!producer.node.GetAttributeByName("value", attribute).IsOK() ||
      attribute == nullptr)
    return std::nullopt;
  Ort::Value value{nullptr};
  if (!attribute.GetTensorAttributeAsOrtValue(value).IsOK() || value == nullptr)
    return std::nullopt;
  return ReadInt64Scalar(value.GetConst());
}

bool IsNmsShapeCompatible(const std::vector<int64_t> &boxes,
                          const std::vector<int64_t> &scores) {
  if (boxes.empty() || scores.empty())
    return true;
  if (boxes.size() != 3 || scores.size() != 3)
    return false;
  if ((boxes[0] > 0 && boxes[0] != 1) ||
      (scores[0] > 0 && scores[0] != 1) ||
      (boxes[2] > 0 && boxes[2] != 4))
    return false;
  return boxes[1] <= 0 || scores[2] <= 0 || boxes[1] == scores[2];
}

} // namespace

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

bool RvnpuEp::IsOperatorEnabled(const std::string &operator_type) const {
  return config_.enabled_ops.empty() ||
         config_.enabled_ops.count(operator_type) != 0;
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
  if (inputs.size() != 2) {
    return false;
  }
  const bool is_float = IsFloatTensor(inputs[0]) && IsFloatTensor(inputs[1]);
  const std::string op = operator_type;
  if (!is_float)
    return false;
  const auto lhs = TensorShape(inputs[0]);
  const auto rhs = TensorShape(inputs[1]);
  if (!lhs || !rhs || lhs->size() > 4 || rhs->size() > 4) {
    return false;
  }
  const bool has_dynamic_dimension =
      std::any_of(lhs->begin(), lhs->end(),
                  [](int64_t value) { return value <= 0; }) ||
      std::any_of(rhs->begin(), rhs->end(),
                  [](int64_t value) { return value <= 0; });
  if (has_dynamic_dimension)
    return true;
  try {
    const auto shape =
        ValidateBinaryElementwiseShapes(*lhs, *rhs, operator_type);
    if ((op == "Sub" || op == "Div") && shape.swap_inputs) {
      return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsConcat(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "Concat" ||
      !(domain.empty() || domain == "ai.onnx") || node.GetSinceVersion() != 13)
    return false;
  const auto inputs = node.GetInputs();
  if (inputs.size() < 2)
    return false;
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(inputs.size());
  for (const auto input : inputs) {
    if (!IsFloatTensor(input))
      return false;
    const auto shape = TensorShape(input);
    if (!shape)
      return false;
    if (std::any_of(shape->begin(), shape->end(),
                    [](int64_t dimension) { return dimension == 0; }))
      return false;
    shapes.push_back(*shape);
  }
  try {
    const size_t axis =
        NormalizeAxis(ReadConcatAxis(node), shapes[0].size(), "Concat");
    const bool dynamic = std::any_of(
        shapes.begin(), shapes.end(), [](const std::vector<int64_t> &shape) {
          return std::any_of(shape.begin(), shape.end(),
                             [](int64_t dimension) { return dimension <= 0; });
        });
    if (dynamic) {
      if (shapes[0].empty() || shapes[0].size() > 4)
        return false;
      for (const auto &shape : shapes) {
        if (shape.size() != shapes[0].size())
          return false;
        for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
          if (dimension != axis && shape[dimension] > 0 &&
              shapes[0][dimension] > 0 &&
              shape[dimension] != shapes[0][dimension])
            return false;
        }
      }
      return true;
    }
    static_cast<void>(ValidateConcatShapes(shapes, static_cast<int64_t>(axis)));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsSplit(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "Split" ||
      !(domain.empty() || domain == "ai.onnx") || node.GetSinceVersion() != 13)
    return false;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() < 2 || !IsFloatTensor(inputs[0]) ||
      !IsTensorType(inputs[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
    return false;
  const auto input_shape = TensorShape(inputs[0]);
  if (!input_shape || input_shape->empty() || input_shape->size() > 4)
    return false;
  try {
    const size_t axis =
        NormalizeAxis(ReadSplitAxis(node), input_shape->size(), "Split");
    const int64_t axis_size = input_shape->at(axis);
    return axis_size > 0 &&
           axis_size % static_cast<int64_t>(outputs.size()) == 0;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsSlice(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "Slice" ||
      !(domain.empty() || domain == "ai.onnx") || node.GetSinceVersion() != 13)
    return false;
  const auto inputs = node.GetInputs();
  if (inputs.size() < 3 || inputs.size() > 5 || !IsFloatTensor(inputs[0]))
    return false;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index] != nullptr &&
        !IsTensorType(inputs[index], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
      return false;
  }
  const auto input_shape = TensorShape(inputs[0]);
  if (!input_shape || input_shape->empty() || input_shape->size() > 4)
    return false;
  const auto outputs = node.GetOutputs();
  if (outputs.size() != 1)
    return false;
  const auto output_shape = TensorShape(outputs[0]);
  if (output_shape &&
      std::any_of(output_shape->begin(), output_shape->end(),
                  [](int64_t dimension) { return dimension == 0; }))
    return false;
  return true;
}

bool RvnpuEp::SupportsResize(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "Resize" ||
      !(domain.empty() || domain == "ai.onnx") || node.GetSinceVersion() != 13)
    return false;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() < 3 || outputs.size() != 1 || !IsFloatTensor(inputs[0]) ||
      inputs[2] == nullptr || !IsFloatTensor(inputs[2]))
    return false;
  const auto input_shape = TensorShape(inputs[0]);
  const auto output_shape = TensorShape(outputs[0]);
  if (!input_shape || !output_shape)
    return false;
  try {
    if (!IsSupportedResizeAttributes(ReadResizeAttributes(node)))
      return false;
    static_cast<void>(ValidateResizeShape(*input_shape, *output_shape));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsTranspose(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "Transpose" ||
      !(domain.empty() || domain == "ai.onnx") || node.GetSinceVersion() != 13)
    return false;
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || !IsFloatTensor(inputs[0]))
    return false;
  const auto input_shape = TensorShape(inputs[0]);
  if (!input_shape)
    return false;
  try {
    const auto perm = ReadTransposePerm(node);
    const bool rank3 = input_shape->size() == 3 &&
                       perm == std::vector<int64_t>({0, 2, 1});
    const bool rank4 = input_shape->size() == 4 &&
                       perm == std::vector<int64_t>({0, 2, 1, 3});
    if (!rank3 && !rank4)
      return false;
    if (std::all_of(input_shape->begin(), input_shape->end(),
                    [](int64_t dimension) { return dimension > 0; }))
      static_cast<void>(ValidateTransposeShape(*input_shape, perm));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsSoftmax(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "Softmax" ||
      !(domain.empty() || domain == "ai.onnx") || node.GetSinceVersion() != 13)
    return false;
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || !IsFloatTensor(inputs[0]))
    return false;
  const auto input_shape = TensorShape(inputs[0]);
  if (!input_shape)
    return false;
  try {
    if (input_shape->size() != 4 ||
        NormalizeAxis(ReadSoftmaxAxis(node), input_shape->size(), "Softmax") !=
            1 ||
        (*input_shape)[1] != 16)
      return false;
    if (std::all_of(input_shape->begin(), input_shape->end(),
                    [](int64_t dimension) { return dimension > 0; }))
      static_cast<void>(ValidateSoftmaxAxis1Shape(*input_shape, 1));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsReduceMax(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "ReduceMax" ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 13)
    return false;
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || !IsFloatTensor(inputs[0]))
    return false;
  const auto input_shape = TensorShape(inputs[0]);
  if (!input_shape)
    return false;
  try {
    const auto attributes = ReadReduceMaxAttributes(node);
    static_cast<void>(ValidateReduceMaxShape(
        *input_shape, attributes.axes, attributes.keepdims));
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
  if ((inputs.size() != 2 && inputs.size() != 3) || !IsFloatTensor(inputs[0]) ||
      !IsFloatTensor(inputs[1]) ||
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
    static_cast<void>(ValidateConvShapes(*input, *weight,
                                         bias ? &*bias : nullptr, attributes));
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
  if (inputs.size() != 1 || outputs.size() != 1 || !IsFloatTensor(inputs[0])) {
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
  if (inputs.size() != 2 || outputs.size() != 2 || !IsFloatTensor(inputs[0]) ||
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
      !IsTensorType(inputs[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }
  for (size_t index = 3; index < inputs.size(); ++index) {
    if (inputs[index] == nullptr)
      continue;
    if (!IsFloatTensor(inputs[index]))
      return false;
    const auto shape = TensorShape(inputs[index]);
    if (!shape || (!shape->empty() &&
                   !(shape->size() == 1 && (*shape)[0] <= 1))) {
      return false;
    }
  }
  const auto boxes = TensorShape(inputs[0]);
  const auto scores = TensorShape(inputs[1]);
  if (!boxes || !scores || !IsNmsShapeCompatible(*boxes, *scores)) {
    return false;
  }
  try {
    const auto max_output = ReadConstantInt64Scalar(inputs[2]);
    if (!max_output || *max_output <= 0)
      return false;
    const auto attributes = ReadNmsAttributes(node);
    if (attributes.center_point_box != 0 && attributes.center_point_box != 1)
      return false;
    const bool static_shapes =
        !boxes->empty() && !scores->empty() &&
        std::all_of(boxes->begin(), boxes->end(),
                    [](int64_t dimension) { return dimension > 0; }) &&
        std::all_of(scores->begin(), scores->end(),
                    [](int64_t dimension) { return dimension > 0; });
    if (static_shapes)
      static_cast<void>(
          ValidateNmsShapes(*boxes, *scores, *max_output, attributes));
    return true;
  } catch (...) {
    return false;
  }
}

bool RvnpuEp::SupportsGatherNd(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "GatherND" ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 13) {
    return false;
  }
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 ||
      !IsFloatTensor(inputs[0]) ||
      !IsTensorType(inputs[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }
  const auto data = TensorShape(inputs[0]);
  const auto indices = TensorShape(inputs[1]);
  if (!data || !indices || data->empty() || data->size() > 3 ||
      indices->size() != 2 || (*indices)[1] <= 0 ||
      (*indices)[1] > static_cast<int64_t>(data->size()) ||
      ReadGatherNdBatchDims(node) != 0 ||
      std::any_of(data->begin(), data->end(),
                  [](int64_t dimension) { return dimension <= 0; })) {
    return false;
  }
  if ((*indices)[0] < 0)
    return true;
  try {
    static_cast<void>(ValidateGatherNdShape(*data, *indices, 0));
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
  const auto nodes = graph.GetNodes();
  for (const auto &node : nodes) {
    if (!ep.IsOperatorEnabled(node.GetOperatorType()))
      continue;
    const bool supported =
        SupportsMatMul(node) || SupportsBinaryElementwise(node, "Add") ||
        SupportsBinaryElementwise(node, "Mul") ||
        SupportsBinaryElementwise(node, "Sub") ||
        SupportsBinaryElementwise(node, "Div") || SupportsConcat(node) ||
        SupportsSplit(node) || SupportsSlice(node) || SupportsResize(node) ||
        SupportsTranspose(node) || SupportsSoftmax(node) ||
        SupportsReduceMax(node) ||
        SupportsSigmoid(node) || SupportsConv(node) || SupportsMaxPool(node) ||
        SupportsTopK(node) || SupportsNms(node) || SupportsGatherNd(node);
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
