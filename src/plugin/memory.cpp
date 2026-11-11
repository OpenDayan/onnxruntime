#include "plugin/memory.h"

#include <cstring>
#include <memory>

namespace dayan {

HostStagingAllocator::HostStagingAllocator(const OrtMemoryInfo *memory_info)
    : OrtAllocator{}, memory_info_(memory_info) {
  version = ORT_API_VERSION;
  Alloc = Allocate;
  Free = Deallocate;
  Info = InfoImpl;
  Reserve = Allocate;
  GetStats = nullptr;
  AllocOnStream = nullptr;
  Shrink = nullptr;
}

void *ORT_API_CALL HostStagingAllocator::Allocate(OrtAllocator *, size_t size) {
  return std::malloc(size == 0 ? 1 : size);
}

void ORT_API_CALL HostStagingAllocator::Deallocate(OrtAllocator *,
                                                   void *pointer) {
  std::free(pointer);
}

const OrtMemoryInfo *ORT_API_CALL
HostStagingAllocator::InfoImpl(const OrtAllocator *allocator) {
  return static_cast<const HostStagingAllocator *>(allocator)->memory_info_;
}

HostStagingDataTransfer::HostStagingDataTransfer(const OrtApi &api,
                                                 const OrtEpApi &ep_api,
                                                 const OrtMemoryDevice *device)
    : OrtDataTransferImpl{}, api_(api), ep_api_(ep_api), device_(device) {
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
  Release = ReleaseImpl;
}

bool ORT_API_CALL HostStagingDataTransfer::CanCopyImpl(
    const OrtDataTransferImpl *transfer, const OrtMemoryDevice *source,
    const OrtMemoryDevice *destination) noexcept {
  const auto &self = *static_cast<const HostStagingDataTransfer *>(transfer);
  const bool source_is_ours =
      self.ep_api_.MemoryDevice_AreEqual(source, self.device_);
  const bool destination_is_ours =
      self.ep_api_.MemoryDevice_AreEqual(destination, self.device_);
  if (source_is_ours && destination_is_ours) {
    return true;
  }
  if (source_is_ours) {
    return self.ep_api_.MemoryDevice_GetDeviceType(destination) ==
               OrtMemoryInfoDeviceType_CPU ||
           self.ep_api_.MemoryDevice_GetMemoryType(destination) ==
               OrtDeviceMemoryType_HOST_ACCESSIBLE;
  }
  if (destination_is_ours) {
    return self.ep_api_.MemoryDevice_GetDeviceType(source) ==
               OrtMemoryInfoDeviceType_CPU ||
           self.ep_api_.MemoryDevice_GetMemoryType(source) ==
               OrtDeviceMemoryType_HOST_ACCESSIBLE;
  }
  return false;
}

OrtStatus *ORT_API_CALL HostStagingDataTransfer::CopyTensorsImpl(
    OrtDataTransferImpl *transfer, const OrtValue **sources,
    OrtValue **destinations, OrtSyncStream **, size_t count) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<HostStagingDataTransfer *>(transfer);
  for (size_t index = 0; index < count; ++index) {
    const void *source = nullptr;
    void *destination = nullptr;
    size_t bytes = 0;
    DAYAN_RETURN_IF_ERROR(self.api_.GetTensorData(sources[index], &source));
    DAYAN_RETURN_IF_ERROR(
        self.api_.GetTensorMutableData(destinations[index], &destination));
    DAYAN_RETURN_IF_ERROR(
        self.api_.GetTensorSizeInBytes(sources[index], &bytes));
    size_t destination_bytes = 0;
    DAYAN_RETURN_IF_ERROR(self.api_.GetTensorSizeInBytes(destinations[index],
                                                         &destination_bytes));
    if (bytes != destination_bytes) {
      return self.api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                    "Dayan staging copy tensor sizes differ");
    }
    if (source != destination && bytes != 0) {
      std::memcpy(destination, source, bytes);
    }
  }
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL
HostStagingDataTransfer::ReleaseImpl(OrtDataTransferImpl *) noexcept {
  // The factory owns the shared transfer object.
}

} // namespace dayan
