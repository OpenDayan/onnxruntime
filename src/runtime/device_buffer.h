#pragma once

#include <cstddef>
#include <cstdint>

#include "npu_runtime.h"

namespace rvnpu {

class DayanDeviceBuffer final {
public:
  DayanDeviceBuffer() = default;
  DayanDeviceBuffer(class DayanRuntime &runtime, size_t bytes,
                    size_t alignment = 64);
  ~DayanDeviceBuffer();
  DayanDeviceBuffer(const DayanDeviceBuffer &) = delete;
  DayanDeviceBuffer &operator=(const DayanDeviceBuffer &) = delete;

  npu_buffer_t *Get() const noexcept { return buffer_; }
  size_t size() const noexcept { return size_; }

private:
  npu_context_t *context_ = nullptr;
  npu_buffer_t *buffer_ = nullptr;
  size_t size_ = 0;
};

} // namespace rvnpu
