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

#include <memory>

#include "plugin/kernel_registry.h"
#include "runtime/device_context.h"

namespace dayan {

class MatMulKernel final : public OrtKernelImpl {
public:
  MatMulKernel(const OrtKernelInfo *info, DeviceContext *context);

  static OrtStatus *Create(const OrtKernelInfo *info, void *state,
                           OrtKernelImpl *&output) noexcept;
  static OrtStatus *ORT_API_CALL
  ComputeImpl(OrtKernelImpl *kernel, OrtKernelContext *context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl *kernel) noexcept;

private:
  Ort::ConstKernelInfo info_;
  DeviceContext *context_;
};

} // namespace dayan
