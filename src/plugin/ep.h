#pragma once

#include <memory>
#include <string>

#include "plugin/plugin_utils.h"
#include "runtime/device_context.h"

namespace rvnpu {

class RvnpuEpFactory;

class RvnpuEp final : public OrtEp {
public:
  struct Config {
    std::string kernel_elf;
    std::string device_mode;
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
