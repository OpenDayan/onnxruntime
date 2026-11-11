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

#include <cstdlib>

#include "plugin/plugin_utils.h"

namespace dayan {

struct HostStagingAllocator final : OrtAllocator {
  explicit HostStagingAllocator(const OrtMemoryInfo *memory_info);

  static void *ORT_API_CALL Allocate(OrtAllocator *, size_t size);
  static void ORT_API_CALL Deallocate(OrtAllocator *, void *pointer);
  static const OrtMemoryInfo *ORT_API_CALL
  InfoImpl(const OrtAllocator *allocator);

  const OrtMemoryInfo *memory_info_;
};

struct HostStagingDataTransfer final : OrtDataTransferImpl {
  HostStagingDataTransfer(const OrtApi &api, const OrtEpApi &ep_api,
                          const OrtMemoryDevice *device);

  static bool ORT_API_CALL CanCopyImpl(
      const OrtDataTransferImpl *transfer, const OrtMemoryDevice *source,
      const OrtMemoryDevice *destination) noexcept;
  static OrtStatus *ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl *transfer,
                                                 const OrtValue **sources,
                                                 OrtValue **destinations,
                                                 OrtSyncStream **streams,
                                                 size_t count) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl *) noexcept;

  const OrtApi &api_;
  const OrtEpApi &ep_api_;
  const OrtMemoryDevice *device_;
};

} // namespace dayan
