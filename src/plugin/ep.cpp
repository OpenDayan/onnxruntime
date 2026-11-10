#include "plugin/ep.h"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include "ops/argmax/argmax.h"
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

class SiluNodeComputeInfo final : public OrtNodeComputeInfo {
public:
  explicit SiluNodeComputeInfo(DeviceContextPtr context)
      : context_(std::move(context)) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

private:
  static OrtStatus *ORT_API_CALL
  CreateStateImpl(OrtNodeComputeInfo *compute_info, OrtNodeComputeContext *,
                  void **compute_state) noexcept {
    RVNPU_EXCEPTION_GUARD_BEGIN
    if (compute_state == nullptr) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "SiLU compute state is null");
    }
    const auto &self = *static_cast<SiluNodeComputeInfo *>(compute_info);
    *compute_state = new DeviceContextPtr(self.context_);
    return nullptr;
    RVNPU_EXCEPTION_GUARD_END
  }

  static OrtStatus *ORT_API_CALL ComputeImpl(
      OrtNodeComputeInfo *, void *compute_state,
      OrtKernelContext *kernel_context) noexcept {
    RVNPU_EXCEPTION_GUARD_BEGIN
    if (compute_state == nullptr) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        "SiLU compute state is missing");
    }
    auto &context = **static_cast<DeviceContextPtr *>(compute_state);
    Ort::KernelContext ort_context(kernel_context);
    const Ort::ConstValue input = ort_context.GetInput(0);
    const auto info = input.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::invalid_argument("RVNPU fused SiLU requires FP32 input");
    }
    const auto shape =
        ValidateUnaryTensorShape(info.GetShape(), "fused SiLU");
    Ort::UnownedValue output =
        ort_context.GetOutput(0, shape.output_shape);
    context.RunSilu(input.GetTensorData<float>(),
                    output.GetTensorMutableData<float>(), shape);
    return nullptr;
    RVNPU_EXCEPTION_GUARD_END
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo *,
                                             void *compute_state) noexcept {
    delete static_cast<DeviceContextPtr *>(compute_state);
  }

  DeviceContextPtr context_;
};

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
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
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
  return config_.disabled_ops.count(operator_type) == 0 &&
         (config_.enabled_ops.empty() ||
          config_.enabled_ops.count(operator_type) != 0);
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
  if (inputs.size() < 3 || inputs.size() > 5)
    return false;
  const bool supported_data_type =
      IsFloatTensor(inputs[0]) ||
      IsTensorType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  if (!supported_data_type)
    return false;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index] != nullptr &&
        !IsTensorType(inputs[index], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
      return false;
  }
  if (inputs.size() == 5) {
    const auto step = ReadConstantInt64Scalar(inputs[4]);
    if (!step || (*step != 1 && *step != -1))
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
  if (inputs.size() != 1)
    return false;
  const bool is_float = IsFloatTensor(inputs[0]);
  const bool is_int64 =
      IsTensorType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  if (!is_float && !is_int64)
    return false;
  const auto input_shape = TensorShape(inputs[0]);
  if (!input_shape)
    return false;
  try {
    const auto perm = ReadTransposePerm(node);
    const bool rank2 = is_int64 && input_shape->size() == 2 &&
                       perm == std::vector<int64_t>({1, 0});
    const bool rank3 = is_float && input_shape->size() == 3 &&
                       perm == std::vector<int64_t>({0, 2, 1});
    const bool rank4 = is_float && input_shape->size() == 4 &&
                       perm == std::vector<int64_t>({0, 2, 1, 3});
    if (!rank2 && !rank3 && !rank4)
      return false;
    if (rank2 && std::any_of(input_shape->begin(), input_shape->end(),
                             [](int64_t dimension) { return dimension < 0; })) {
      return true;
    }
    if (rank3 &&
        !(((*input_shape)[1] > 0 && (*input_shape)[1] <= 128 &&
           (*input_shape)[2] > 0 && (*input_shape)[2] % 16 == 0) ||
          ((*input_shape)[0] > 0 && (*input_shape)[1] > 0 &&
           (*input_shape)[2] == 1 && (*input_shape)[0] <= 4 &&
           (*input_shape)[1] <= 16384)))
      return false;
    if (std::all_of(input_shape->begin(), input_shape->end(),
                    [](int64_t dimension) { return dimension >= 0; }))
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

bool RvnpuEp::SupportsArgMax(const Ort::ConstNode &node) {
  const auto domain = node.GetDomain();
  if (node.GetOperatorType() != "ArgMax" ||
      !(domain.empty() || domain == "ai.onnx") ||
      node.GetSinceVersion() != 13)
    return false;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 ||
      !IsFloatTensor(inputs[0]))
    return false;
  const auto input_shape = TensorShape(inputs[0]);
  if (!input_shape)
    return false;
  try {
    static_cast<void>(
        ValidateArgMaxShape(*input_shape, ReadArgMaxAttributes(node)));
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

std::optional<Ort::ConstNode>
RvnpuEp::MatchSiluPattern(const Ort::ConstNode &mul) {
  if (!SupportsBinaryElementwise(mul, "Mul"))
    return std::nullopt;
  const auto mul_inputs = mul.GetInputs();
  if (mul_inputs.size() != 2)
    return std::nullopt;

  for (size_t sigmoid_input_index = 0; sigmoid_input_index < 2;
       ++sigmoid_input_index) {
    const auto producer = mul_inputs[sigmoid_input_index].GetProducerNode();
    if (producer.node == nullptr || !SupportsSigmoid(producer.node))
      continue;
    const auto sigmoid_inputs = producer.node.GetInputs();
    const auto sigmoid_outputs = producer.node.GetOutputs();
    if (sigmoid_inputs.size() != 1 || sigmoid_outputs.size() != 1 ||
        sigmoid_inputs[0].GetName() !=
            mul_inputs[1 - sigmoid_input_index].GetName() ||
        sigmoid_outputs[0].GetName() !=
            mul_inputs[sigmoid_input_index].GetName() ||
        sigmoid_outputs[0].IsGraphOutput()) {
      continue;
    }
    const auto consumers = sigmoid_outputs[0].GetConsumers();
    if (consumers.size() != 1 || consumers[0].node == nullptr ||
        consumers[0].node.GetId() != mul.GetId()) {
      continue;
    }
    return producer.node;
  }
  return std::nullopt;
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
    const auto shape = ValidateConvShapes(*input, *weight,
                                         bias ? &*bias : nullptr, attributes);
    DeviceContext::ValidateConvResources(shape);
    // Static Conv semantics are enabled within the device/resource envelope.
    return true;
  } catch (const ConvModelError &error) {
    throw Ort::Exception(error.what(), ORT_INVALID_ARGUMENT);
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
  if (inputs.size() != 2 || outputs.size() != 1)
    return false;
  const bool supported_data_type =
      IsFloatTensor(inputs[0]) ||
      IsTensorType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  if (!supported_data_type ||
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
    static_cast<void>(ValidateGatherNdShape(
        *data, *indices, 0,
        IsTensorType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            ? sizeof(int64_t)
            : sizeof(float)));
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
  std::unordered_set<size_t> fused_node_ids;
  if (ep.IsOperatorEnabled("Silu")) {
    for (const auto &node : nodes) {
      const auto sigmoid = MatchSiluPattern(node);
      if (!sigmoid)
        continue;
      // A node explicitly excluded from this EP must not be claimed through
      // fusion either. Keep the existing enabled_ops=Silu shorthand intact.
      if (ep.config_.disabled_ops.count("Sigmoid") ||
          ep.config_.disabled_ops.count("Mul"))
        continue;
      std::array<Ort::ConstNode, 2> fusion_nodes{*sigmoid, node};
      OrtNodeFusionOptions fusion_options{};
      fusion_options.ort_version_supported = ORT_API_VERSION;
      RVNPU_RETURN_IF_ERROR(ep.ep_api_.EpGraphSupportInfo_AddNodesToFuse(
          support_info,
          reinterpret_cast<const OrtNode *const *>(fusion_nodes.data()),
          fusion_nodes.size(), &fusion_options));
      fused_node_ids.insert(sigmoid->GetId());
      fused_node_ids.insert(node.GetId());
    }
  }
  for (const auto &node : nodes) {
    if (fused_node_ids.count(node.GetId()) != 0)
      continue;
    if (!ep.IsOperatorEnabled(node.GetOperatorType()))
      continue;
    const bool supported =
        SupportsMatMul(node) || SupportsBinaryElementwise(node, "Add") ||
        SupportsBinaryElementwise(node, "Mul") ||
        SupportsBinaryElementwise(node, "Sub") ||
        SupportsBinaryElementwise(node, "Div") || SupportsConcat(node) ||
        SupportsSplit(node) || SupportsSlice(node) || SupportsResize(node) ||
        SupportsTranspose(node) || SupportsSoftmax(node) ||
        SupportsReduceMax(node) || SupportsArgMax(node) ||
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

OrtStatus *ORT_API_CALL RvnpuEp::CompileImpl(
    OrtEp *ep_ptr, const OrtGraph **graph_ptrs, const OrtNode **fused_nodes,
    size_t count, OrtNodeComputeInfo **node_compute_infos,
    OrtNode **ep_context_nodes) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  if (graph_ptrs == nullptr || fused_nodes == nullptr ||
      node_compute_infos == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "RVNPU Compile received null arrays");
  }
  auto &ep = *static_cast<RvnpuEp *>(ep_ptr);
  for (size_t index = 0; index < count; ++index) {
    Ort::ConstGraph graph{graph_ptrs[index]};
    const auto nodes = graph.GetNodes();
    const size_t sigmoid_count =
        std::count_if(nodes.begin(), nodes.end(), [](const auto &node) {
          return node.GetOperatorType() == "Sigmoid";
        });
    const size_t mul_count =
        std::count_if(nodes.begin(), nodes.end(), [](const auto &node) {
          return node.GetOperatorType() == "Mul";
        });
    Ort::ConstNode fused_node{fused_nodes[index]};
    if (nodes.size() != 2 || sigmoid_count != 1 || mul_count != 1 ||
        fused_node.GetInputs().size() != 1 ||
        fused_node.GetOutputs().size() != 1 ||
        fused_node.GetEpName() != ep.name_) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, "RVNPU Compile expected a Sigmoid+Mul SiLU subgraph");
    }
    node_compute_infos[index] =
        new SiluNodeComputeInfo(ep.device_context_);
    if (ep_context_nodes != nullptr)
      ep_context_nodes[index] = nullptr;
  }
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

void ORT_API_CALL RvnpuEp::ReleaseNodeComputeInfosImpl(
    OrtEp *, OrtNodeComputeInfo **node_compute_infos,
    size_t count) noexcept {
  if (node_compute_infos == nullptr)
    return;
  for (size_t index = 0; index < count; ++index)
    delete static_cast<SiluNodeComputeInfo *>(node_compute_infos[index]);
}

OrtStatus *ORT_API_CALL RvnpuEp::GetKernelRegistryImpl(
    OrtEp *ep_ptr, const OrtKernelRegistry **registry) noexcept {
  *registry = static_cast<RvnpuEp *>(ep_ptr)->kernel_registry_;
  return nullptr;
}

} // namespace rvnpu
