#pragma once

#include <cstddef>
#include <cstdint>

struct NPURuntimeState;

namespace rvnpu {

class DeviceBuffer {
public:
  DeviceBuffer(const NPURuntimeState &runtime, size_t bytes);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;

  uint64_t address() const noexcept { return address_; }
  size_t size() const noexcept { return size_; }

  void CopyFromHost(const void *source, size_t bytes);
  void CopyToHost(void *destination, size_t bytes) const;

private:
  void Release() noexcept;

  const NPURuntimeState *runtime_{};
  uint64_t address_{};
  size_t size_{};
};

} // namespace rvnpu
