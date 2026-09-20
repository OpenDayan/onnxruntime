#include "runtime/device_buffer.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "NPUDriver.h"
#include "NPURuntime.h"

namespace rvnpu {

namespace {
void RequireSuccess(ssize_t result, const char *stage) {
  if (result != NPU_SUCCESS) {
    throw std::runtime_error(std::string(stage) +
                             " failed: runtime code=" + std::to_string(result));
  }
}
} // namespace

DeviceBuffer::DeviceBuffer(const NPURuntimeState &runtime, size_t bytes)
    : runtime_(&runtime), size_(bytes) {
  if (bytes == 0) {
    throw std::invalid_argument("device allocation size must be positive");
  }
  // The hw-lab allocator does not align allocations. Keep the logical size for
  // copies, but reserve an aligned block so a following int64_t output buffer
  // cannot start at a misaligned address.
  const size_t allocation_bytes = (bytes + 7U) & ~static_cast<size_t>(7U);
  RequireSuccess(npu_runtime_malloc(runtime_, &address_, allocation_bytes),
                 "npu_runtime_malloc");
  if (address_ == 0) {
    throw std::runtime_error("npu_runtime_malloc returned address zero");
  }
}

DeviceBuffer::~DeviceBuffer() { Release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : runtime_(std::exchange(other.runtime_, nullptr)),
      address_(std::exchange(other.address_, 0)),
      size_(std::exchange(other.size_, 0)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
  if (this != &other) {
    Release();
    runtime_ = std::exchange(other.runtime_, nullptr);
    address_ = std::exchange(other.address_, 0);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

void DeviceBuffer::CopyFromHost(const void *source, size_t bytes) {
  if (source == nullptr || bytes > size_) {
    throw std::invalid_argument("invalid host-to-device copy");
  }
  const ssize_t result =
      npu_memcpy_to_dev(&runtime_->dev_state, address_, source, bytes);
  if (result < 0) {
    throw std::runtime_error("host-to-device copy failed: driver code=" +
                             std::to_string(result));
  }
}

void DeviceBuffer::CopyToHost(void *destination, size_t bytes) const {
  if (destination == nullptr || bytes > size_) {
    throw std::invalid_argument("invalid device-to-host copy");
  }
  const ssize_t result =
      npu_memcpy_from_dev(&runtime_->dev_state, destination, address_, bytes);
  if (result < 0) {
    throw std::runtime_error("device-to-host copy failed: driver code=" +
                             std::to_string(result));
  }
}

void DeviceBuffer::Release() noexcept {
  if (runtime_ != nullptr && address_ != 0) {
    static_cast<void>(npu_runtime_free(runtime_, address_));
  }
  runtime_ = nullptr;
  address_ = 0;
  size_ = 0;
}

} // namespace rvnpu
