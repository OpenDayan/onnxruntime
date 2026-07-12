#include "plugin/ep.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "plugin/ep_factory.h"
#include "plugin/kernel_registry.h"
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

OrtStatus *ORT_API_CALL
RvnpuEp::GetCapabilityImpl(OrtEp *ep_ptr, const OrtGraph *graph_ptr,
                           OrtEpGraphSupportInfo *support_info) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  auto &ep = *static_cast<RvnpuEp *>(ep_ptr);
  Ort::ConstGraph graph{graph_ptr};
  for (const auto &node : graph.GetNodes()) {
    if (!SupportsMatMul(node)) {
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
