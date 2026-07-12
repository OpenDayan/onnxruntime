#pragma once

#include <memory>
#include <string>

#include "plugin/memory.h"

namespace rvnpu {

class RvnpuEp;

class RvnpuEpFactory final : public OrtEpFactory {
public:
  RvnpuEpFactory(const OrtApi &api, const OrtEpApi &ep_api,
                 const OrtLogger &logger, bool allow_virtual_devices);
  ~RvnpuEpFactory();

  const OrtApi &api() const noexcept { return api_; }
  const OrtEpApi &ep_api() const noexcept { return ep_api_; }
  const std::string &name() const noexcept { return name_; }

private:
  static const char *ORT_API_CALL GetNameImpl(const OrtEpFactory *) noexcept;
  static const char *ORT_API_CALL GetVendorImpl(const OrtEpFactory *) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory *) noexcept;
  static const char *ORT_API_CALL GetVersionImpl(const OrtEpFactory *) noexcept;

  static OrtStatus *ORT_API_CALL
  GetSupportedDevicesImpl(OrtEpFactory *, const OrtHardwareDevice *const *,
                          size_t, OrtEpDevice **, size_t, size_t *) noexcept;
  static OrtStatus *ORT_API_CALL CreateEpImpl(OrtEpFactory *,
                                              const OrtHardwareDevice *const *,
                                              const OrtKeyValuePairs *const *,
                                              size_t, const OrtSessionOptions *,
                                              const OrtLogger *,
                                              OrtEp **) noexcept;
  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory *, OrtEp *) noexcept;

  static OrtStatus *ORT_API_CALL CreateAllocatorImpl(OrtEpFactory *,
                                                     const OrtMemoryInfo *,
                                                     const OrtKeyValuePairs *,
                                                     OrtAllocator **) noexcept;
  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory *,
                                                OrtAllocator *) noexcept;
  static OrtStatus *ORT_API_CALL
  CreateDataTransferImpl(OrtEpFactory *, OrtDataTransferImpl **) noexcept;
  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory *) noexcept;
  static OrtStatus *ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory *, const OrtMemoryDevice *, const OrtKeyValuePairs *,
      OrtSyncStreamImpl **) noexcept;

  const OrtApi &api_;
  const OrtEpApi &ep_api_;
  const OrtLogger &logger_;
  const bool allow_virtual_devices_;
  const std::string name_{"RVNPUExecutionProvider"};
  const std::string vendor_{"RVNPU"};
  const uint32_t vendor_id_{0x5256};
  const std::string version_{"0.1.0"};

  Ort::MemoryInfo default_memory_info_{nullptr};
  Ort::MemoryInfo readonly_memory_info_{nullptr};
  std::unique_ptr<HostStagingDataTransfer> data_transfer_;
  OrtHardwareDevice *virtual_device_{};
};

} // namespace rvnpu
