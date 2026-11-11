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

#pragma once

#include <cstddef>
#include <cstdint>

#include "npu_runtime.h"

namespace dayan {

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

} // namespace dayan
