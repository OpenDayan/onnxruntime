#pragma once

#include <memory>
#include <string>
#include <unordered_set>

#include "plugin/plugin_utils.h"
#include "runtime/device_context.h"

namespace rvnpu {

class RvnpuEpFactory;

class RvnpuEp final : public OrtEp {
public:
  struct Config {
    std::string kernel_elf;
    std::string device_mode;
    std::unordered_set<std::string> enabled_ops;
    uint32_t device_id{};
  };

  RvnpuEp(RvnpuEpFactory &factory, Config config, const OrtLogger &logger);
  ~RvnpuEp();

private:
  static const char *ORT_API_CALL GetNameImpl(const OrtEp *) noexcept;
  static OrtStatus *ORT_API_CALL GetCapabilityImpl(
      OrtEp *, const OrtGraph *, OrtEpGraphSupportInfo *) noexcept;
  static OrtStatus *ORT_API_CALL
  GetKernelRegistryImpl(OrtEp *, const OrtKernelRegistry **) noexcept;

  static bool SupportsMatMul(const Ort::ConstNode &node);
  static bool SupportsBinaryElementwise(const Ort::ConstNode &node,
                                        const char *operator_type);
  static bool SupportsConcat(const Ort::ConstNode &node);
  static bool SupportsSplit(const Ort::ConstNode &node);
  static bool SupportsSlice(const Ort::ConstNode &node);
  static bool SupportsResize(const Ort::ConstNode &node);
  static bool SupportsTranspose(const Ort::ConstNode &node);
  static bool SupportsSoftmax(const Ort::ConstNode &node);
  static bool SupportsReduceMax(const Ort::ConstNode &node);
  static bool SupportsSigmoid(const Ort::ConstNode &node);
  static bool SupportsConv(const Ort::ConstNode &node);
  static bool SupportsMaxPool(const Ort::ConstNode &node);
  static bool SupportsTopK(const Ort::ConstNode &node);
  static bool SupportsNms(const Ort::ConstNode &node);
  static bool SupportsGatherNd(const Ort::ConstNode &node);
  bool IsOperatorEnabled(const std::string &operator_type) const;

  RvnpuEpFactory &factory_;
  const OrtApi &api_;
  const OrtEpApi &ep_api_;
  const std::string name_;
  Config config_;
  const OrtLogger &logger_;
  DeviceContextPtr device_context_;
  OrtKernelRegistry *kernel_registry_{};
};

} // namespace rvnpu
