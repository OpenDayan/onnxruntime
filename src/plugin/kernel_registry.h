#pragma once

#include "plugin/plugin_utils.h"

namespace rvnpu {

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

#define RVNPU_KERNEL_CLASS_NAME(domain, start_version, end_version, name)      \
  rvnpu_##name##_##domain##_v##start_version##_##end_version

#define RVNPU_DEFINE_VERSIONED_KERNEL(name, domain, start_version,             \
                                      end_version, builder, kernel_class)      \
  class RVNPU_KERNEL_CLASS_NAME(domain, start_version, end_version, name);     \
  template <>                                                                  \
  OrtStatus *BuildKernelCreateInfo<RVNPU_KERNEL_CLASS_NAME(                    \
      domain, start_version, end_version, name)>(                              \
      const char *ep_name, void *kernel_state, KernelCreateInfo *result) {     \
    RVNPU_EXCEPTION_GUARD_BEGIN                                                \
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
    RVNPU_EXCEPTION_GUARD_END                                                  \
  }

#define RVNPU_DEFINE_KERNEL(name, domain, version, builder, kernel_class)      \
  RVNPU_DEFINE_VERSIONED_KERNEL(name, domain, version, version, builder,       \
                                kernel_class)

} // namespace rvnpu
