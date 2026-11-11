/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * onnxruntime is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include "plugin/ep_factory.h"

#include <charconv>
#include <memory>
#include <string>
#include <unordered_set>

#include "plugin/ep.h"

namespace dayan {

namespace {
constexpr const char *kVirtualMetadataKey = "is_virtual";
constexpr const char *kOptionPrefix = "ep.dayanexecutionprovider.";

uint32_t ParseDeviceId(const std::string &text) {
  uint32_t value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw std::invalid_argument(
        "Dayan option device_id must be an unsigned integer");
  }
  return value;
}

std::unordered_set<std::string> ParseOperatorSet(const std::string &text,
                                                const char *option) {
  std::unordered_set<std::string> result;
  if (!text.empty() && text.back() == ',')
    throw std::invalid_argument(std::string("Dayan option ") + option +
                                " contains an empty name");
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    const std::string name = text.substr(begin, end - begin);
    if (name.empty())
      throw std::invalid_argument(std::string("Dayan option ") + option +
                                  " contains an empty name");
    result.insert(name);
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return result;
}
} // namespace

DayanEpFactory::DayanEpFactory(const OrtApi &api, const OrtEpApi &ep_api,
                               const OrtLogger &logger,
                               bool allow_virtual_devices)
    : OrtEpFactory{}, api_(api), ep_api_(ep_api), logger_(logger),
      allow_virtual_devices_(allow_virtual_devices) {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  default_memory_info_ = Ort::MemoryInfo{"Dayan host staging",
                                         OrtMemoryInfoDeviceType_NPU,
                                         vendor_id_,
                                         0,
                                         OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                         64,
                                         OrtAllocatorType::OrtDeviceAllocator};
  readonly_memory_info_ =
      Ort::MemoryInfo{"Dayan host staging readonly",
                      OrtMemoryInfoDeviceType_NPU,
                      vendor_id_,
                      0,
                      OrtDeviceMemoryType_HOST_ACCESSIBLE,
                      64,
                      OrtAllocatorType::OrtReadOnlyAllocator};
  data_transfer_ = std::make_unique<HostStagingDataTransfer>(
      api_, ep_api_, ep_api_.MemoryInfo_GetMemoryDevice(default_memory_info_));
}

DayanEpFactory::~DayanEpFactory() {
  if (virtual_device_ != nullptr) {
    ep_api_.ReleaseHardwareDevice(virtual_device_);
  }
}

const char *ORT_API_CALL
DayanEpFactory::GetNameImpl(const OrtEpFactory *factory) noexcept {
  return static_cast<const DayanEpFactory *>(factory)->name_.c_str();
}

const char *ORT_API_CALL
DayanEpFactory::GetVendorImpl(const OrtEpFactory *factory) noexcept {
  return static_cast<const DayanEpFactory *>(factory)->vendor_.c_str();
}

uint32_t ORT_API_CALL
DayanEpFactory::GetVendorIdImpl(const OrtEpFactory *factory) noexcept {
  return static_cast<const DayanEpFactory *>(factory)->vendor_id_;
}

const char *ORT_API_CALL
DayanEpFactory::GetVersionImpl(const OrtEpFactory *factory) noexcept {
  return static_cast<const DayanEpFactory *>(factory)->version_.c_str();
}

OrtStatus *ORT_API_CALL DayanEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory *factory_ptr, const OrtHardwareDevice *const *, size_t,
    OrtEpDevice **ep_devices, size_t max_ep_devices, size_t *count) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &factory = *static_cast<DayanEpFactory *>(factory_ptr);
  *count = 0;
  if (!factory.allow_virtual_devices_ || max_ep_devices == 0) {
    return nullptr;
  }

  if (factory.virtual_device_ == nullptr) {
    Ort::KeyValuePairs hardware_metadata;
    hardware_metadata.Add(kVirtualMetadataKey, "1");
    DAYAN_RETURN_IF_ERROR(factory.ep_api_.CreateHardwareDevice(
        OrtHardwareDeviceType_NPU, factory.vendor_id_, 0,
        factory.vendor_.c_str(), hardware_metadata, &factory.virtual_device_));
  }

  Ort::KeyValuePairs ep_metadata;
  Ort::KeyValuePairs ep_options;
  ep_options.Add("device_id", "0");
  ep_options.Add("device_mode", "emulator");

  OrtEpDevice *device = nullptr;
  DAYAN_RETURN_IF_ERROR(factory.ep_api_.CreateEpDevice(
      &factory, factory.virtual_device_, ep_metadata, ep_options, &device));
  DAYAN_RETURN_IF_ERROR(factory.ep_api_.EpDevice_AddAllocatorInfo(
      device, factory.default_memory_info_));
  DAYAN_RETURN_IF_ERROR(factory.ep_api_.EpDevice_AddAllocatorInfo(
      device, factory.readonly_memory_info_));
  ep_devices[0] = device;
  *count = 1;
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL DayanEpFactory::CreateEpImpl(
    OrtEpFactory *factory_ptr, const OrtHardwareDevice *const *,
    const OrtKeyValuePairs *const *, size_t num_devices,
    const OrtSessionOptions *session_options, const OrtLogger *logger,
    OrtEp **ep) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &factory = *static_cast<DayanEpFactory *>(factory_ptr);
  *ep = nullptr;
  if (num_devices != 1) {
    return factory.api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "Dayan EP requires exactly one device");
  }
  if (session_options == nullptr || logger == nullptr) {
    return factory.api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Dayan EP received null session options or logger");
  }

  DayanEp::Config config;
  config.kernel_elf = GetSessionOption(
      *session_options, std::string(kOptionPrefix) + "kernel_elf");
  config.device_mode = GetSessionOption(
      *session_options, std::string(kOptionPrefix) + "device_mode", "emulator");
  config.device_id = ParseDeviceId(GetSessionOption(
      *session_options, std::string(kOptionPrefix) + "device_id", "0"));
  config.enabled_ops = ParseOperatorSet(GetSessionOption(
      *session_options, std::string(kOptionPrefix) + "enabled_ops", ""), "enabled_ops");
  config.disabled_ops = ParseOperatorSet(GetSessionOption(
      *session_options, std::string(kOptionPrefix) + "disabled_ops", ""), "disabled_ops");
  if (config.kernel_elf.empty()) {
    return factory.api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "Dayan EP option kernel_elf is required");
  }
  if (config.device_mode != "emulator") {
    return factory.api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Dayan EP currently supports device_mode=emulator only");
  }
  if (config.device_id != 0) {
    return factory.api_.CreateStatus(
        ORT_INVALID_ARGUMENT, "Dayan EP currently supports device_id=0 only");
  }

  *ep = new DayanEp(factory, std::move(config), *logger);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL DayanEpFactory::ReleaseEpImpl(OrtEpFactory *,
                                                OrtEp *ep) noexcept {
  delete static_cast<DayanEp *>(ep);
}

OrtStatus *ORT_API_CALL DayanEpFactory::CreateAllocatorImpl(
    OrtEpFactory *factory_ptr, const OrtMemoryInfo *memory_info,
    const OrtKeyValuePairs *, OrtAllocator **allocator) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &factory = *static_cast<DayanEpFactory *>(factory_ptr);
  *allocator = nullptr;
  if (memory_info == nullptr ||
      !(memory_info == factory.default_memory_info_ ||
        memory_info == factory.readonly_memory_info_)) {
    return factory.api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Dayan EP received unknown allocator memory info");
  }
  *allocator = new HostStagingAllocator(memory_info);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL DayanEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory *, OrtAllocator *allocator) noexcept {
  delete static_cast<HostStagingAllocator *>(allocator);
}

OrtStatus *ORT_API_CALL DayanEpFactory::CreateDataTransferImpl(
    OrtEpFactory *factory_ptr, OrtDataTransferImpl **transfer) noexcept {
  auto &factory = *static_cast<DayanEpFactory *>(factory_ptr);
  *transfer = factory.data_transfer_.get();
  return nullptr;
}

bool ORT_API_CALL
DayanEpFactory::IsStreamAwareImpl(const OrtEpFactory *) noexcept {
  return false;
}

OrtStatus *ORT_API_CALL DayanEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory *, const OrtMemoryDevice *, const OrtKeyValuePairs *,
    OrtSyncStreamImpl **stream) noexcept {
  *stream = nullptr;
  return nullptr;
}

} // namespace dayan
