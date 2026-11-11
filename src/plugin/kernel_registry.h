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

#include "plugin/plugin_utils.h"

namespace dayan {

struct KernelCreateInfo {
  Ort::KernelDef definition{nullptr};
  OrtKernelCreateFunc create{};
  void *state{};
};

using BuildKernelCreateInfoFn = OrtStatus *(*)(const char *, void *,
                                               KernelCreateInfo *);

OrtStatus *CreateKernelRegistry(const char *ep_name, void *kernel_state,
                                OrtKernelRegistry **registry);
size_t GetKernelCount();

const OrtDataType *TensorType(ONNXTensorElementDataType type);

template <typename T>
OrtStatus *BuildKernelCreateInfo(const char *, void *, KernelCreateInfo *);

static constexpr const char *kOnnxDomain = "";

#define DAYAN_KERNEL_CLASS_NAME(domain, start_version, end_version, name)      \
  dayan_##name##_##domain##_v##start_version##_##end_version

#define DAYAN_DEFINE_VERSIONED_KERNEL(name, domain, start_version,             \
                                      end_version, builder, kernel_class)      \
  class DAYAN_KERNEL_CLASS_NAME(domain, start_version, end_version, name);     \
  template <>                                                                  \
  OrtStatus *BuildKernelCreateInfo<DAYAN_KERNEL_CLASS_NAME(                    \
      domain, start_version, end_version, name)>(                              \
      const char *ep_name, void *kernel_state, KernelCreateInfo *result) {     \
    DAYAN_EXCEPTION_GUARD_BEGIN                                                \
    auto definition = builder.SetOperatorType(#name)                           \
                          .SetDomain(domain)                                   \
                          .SetSinceVersion(start_version, end_version)         \
                          .SetExecutionProvider(ep_name)                       \
                          .Build();                                            \
    auto create = [](void *state, const OrtKernelInfo *info,                   \
                     OrtKernelImpl **output) noexcept -> OrtStatus * {         \
      if (output == nullptr) {                                                 \
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,                \
                                          "kernel output pointer is null");    \
      }                                                                        \
      *output = nullptr;                                                       \
      return kernel_class::Create(info, state, *output);                       \
    };                                                                         \
    result->definition = std::move(definition);                                \
    result->create = create;                                                   \
    result->state = kernel_state;                                              \
    return nullptr;                                                            \
    DAYAN_EXCEPTION_GUARD_END                                                  \
  }

#define DAYAN_DEFINE_KERNEL(name, domain, version, builder, kernel_class)      \
  DAYAN_DEFINE_VERSIONED_KERNEL(name, domain, version, version, builder,       \
                                kernel_class)

} // namespace dayan
