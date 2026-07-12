#include <cstring>
#include <memory>
#include <string>

#include "plugin/ep_factory.h"

#if defined(__GNUC__)
#define RVNPU_EXPORT __attribute__((visibility("default")))
#else
#define RVNPU_EXPORT
#endif

extern "C" {

RVNPU_EXPORT OrtStatus *
CreateEpFactories(const char *registration_name, const OrtApiBase *api_base,
                  const OrtLogger *default_logger, OrtEpFactory **factories,
                  size_t max_factories, size_t *num_factories) {
  if (api_base == nullptr || default_logger == nullptr ||
      factories == nullptr || num_factories == nullptr) {
    return nullptr;
  }
  const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
  Ort::InitApi(api);
  if (max_factories < 1) {
    return api->CreateStatus(ORT_INVALID_ARGUMENT,
                             "RVNPU Plugin EP requires one factory slot");
  }
  const std::string name =
      registration_name == nullptr ? "" : registration_name;
  const bool allow_virtual =
      name.size() >= 8 && name.compare(name.size() - 8, 8, ".virtual") == 0;
  try {
    factories[0] = new rvnpu::RvnpuEpFactory(*api, *api->GetEpApi(),
                                             *default_logger, allow_virtual);
    *num_factories = 1;
    return nullptr;
  } catch (const std::exception &ex) {
    return api->CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

RVNPU_EXPORT OrtStatus *ReleaseEpFactory(OrtEpFactory *factory) {
  delete static_cast<rvnpu::RvnpuEpFactory *>(factory);
  return nullptr;
}

} // extern "C"
