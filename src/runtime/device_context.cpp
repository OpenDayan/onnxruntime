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

#include "runtime/device_context.h"

#include <cstring>
#include <limits>

#include "runtime/device_buffer.h"

namespace dayan {

namespace {

constexpr uint32_t kTimeoutMs = 18000000;

void RequireSuccess(int status, const std::string &stage) {
  if (status != 0) {
    throw std::runtime_error(stage + " failed: runtime code=" +
                             std::to_string(status));
  }
}

void PreserveError(int candidate, int *status) {
  if (*status == 0 && candidate < 0)
    *status = candidate;
}

} // namespace

void DayanRuntime::EnsureInitialized() {
  std::call_once(initialized_, [this]() { Initialize(); });
}

void DayanRuntime::Initialize() {
  four_core_ = false;
  npu_runtime_options_t options;
  npu_runtime_options_init(&options);
  if (const char *socket = std::getenv("DAYAN_SERVER_SOCKET")) {
    if (*socket)
      options.server_socket = socket;
  }
  RequireSuccess(npu_runtime_open(&runtime_, &options), "npu_runtime_open");
  RequireSuccess(npu_runtime_get_uram_capacity(runtime_, &uram_capacity_),
                 "npu_runtime_get_uram_capacity");
  npu_runtime_topology_t topology{};
  if (npu_runtime_get_topology(runtime_, &topology) == 0)
    four_core_ = topology.cores_per_cluster >= 4 &&
                 topology.enabled_cluster_mask != 0 &&
                 (topology.enabled_core_mask & 0xfU) == 0xfU;
  try {
    RequireSuccess(npu_context_create(runtime_, nullptr, &context_),
                   "npu_context_create");
    RequireSuccess(npu_stream_create(context_, nullptr, &stream_),
                   "npu_stream_create");
  } catch (...) {
    if (context_ != nullptr)
      static_cast<void>(npu_context_destroy(context_));
    static_cast<void>(npu_runtime_close(runtime_));
    stream_ = nullptr;
    context_ = nullptr;
    runtime_ = nullptr;
    context_ = nullptr;
    throw;
  }
}

DayanRuntime::~DayanRuntime() {
  if (stream_ != nullptr)
    static_cast<void>(npu_stream_destroy(stream_));
  if (context_ != nullptr)
    static_cast<void>(npu_context_destroy(context_));
  if (runtime_ != nullptr)
    static_cast<void>(npu_runtime_close(runtime_));
}

DeviceContext::DeviceContext(std::string kernel_elf,
                             bool cpu_bf16_conv_reference)
    : kernel_elf_(std::move(kernel_elf)),
      cpu_bf16_conv_reference_(cpu_bf16_conv_reference) {}

DeviceContext::~DeviceContext() = default;

void DeviceContext::WaitEvent(npu_event_t *event) {
  if (event == nullptr)
    throw std::runtime_error("Dayan runtime returned no transfer event");
  npu_event_info_t info;
  const int status = npu_event_wait(event, kTimeoutMs, &info);
  const int destroy_status = npu_event_destroy(event);
  RequireSuccess(status, "npu_event_wait");
  if (info.status < std::numeric_limits<int>::min() ||
      info.status > std::numeric_limits<int>::max())
    throw std::runtime_error("Dayan transfer event status is out of range");
  if (info.status < 0)
    throw std::runtime_error("Dayan transfer failed: device status=" +
                             std::to_string(info.status));
  RequireSuccess(destroy_status, "npu_event_destroy");
}

void DeviceContext::CopyToDevice(npu_buffer_t *destination,
                                 const void *source, size_t bytes) {
  npu_event_t *event = nullptr;
  RequireSuccess(npu_memcpy_h2d_async(runtime_.stream(), destination, 0,
                                      source, bytes, &event),
                 "npu_memcpy_h2d_async");
  WaitEvent(event);
}

void DeviceContext::CopyFromDevice(void *destination, npu_buffer_t *source,
                                   size_t bytes) {
  npu_event_t *event = nullptr;
  RequireSuccess(npu_memcpy_d2h_async(runtime_.stream(), destination, source,
                                      0, bytes, &event),
                 "npu_memcpy_d2h_async");
  WaitEvent(event);
}

void DeviceContext::RunAdd(const float *lhs, const float *rhs, float *output,
                           const BinaryElementwiseShape &shape) {
  RunBinary("Add", lhs, rhs, output, shape, 0);
}
} // namespace dayan
