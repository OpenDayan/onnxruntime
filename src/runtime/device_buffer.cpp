#include "runtime/device_buffer.h"

#include <stdexcept>
#include <string>

#include "runtime/device_context.h"

namespace rvnpu {

DayanDeviceBuffer::DayanDeviceBuffer(DayanRuntime &runtime, size_t bytes,
                                     size_t alignment)
    : context_(runtime.context()) {
  if (bytes == 0)
    throw std::invalid_argument("Dayan buffer size must be nonzero");
  if (alignment == 0 || ((alignment & (alignment - 1)) != 0))
    throw std::invalid_argument("Dayan buffer alignment must be a power of two");

  npu_buffer_desc_t description;
  npu_buffer_desc_init(&description, static_cast<uint64_t>(bytes));
  description.alignment = static_cast<uint32_t>(alignment);
  const int status = npu_buffer_create(context_, &description, &buffer_);
  if (status != 0) {
    throw std::runtime_error("npu_buffer_create failed: runtime code=" +
                             std::to_string(status));
  }
  size_ = bytes;
}

DayanDeviceBuffer::~DayanDeviceBuffer() {
  if (buffer_ != nullptr)
    static_cast<void>(npu_buffer_destroy(buffer_));
}

} // namespace rvnpu
