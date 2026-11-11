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
#include <optional>
#include <string>
#include <unordered_set>

#include "plugin/plugin_utils.h"
#include "runtime/device_context.h"

namespace dayan {

class DayanEpFactory;

class DayanEp final : public OrtEp {
public:
  struct Config {
    std::string kernel_elf;
    std::string device_mode;
    std::unordered_set<std::string> enabled_ops;
    std::unordered_set<std::string> disabled_ops;
    uint32_t device_id{};
  };

  DayanEp(DayanEpFactory &factory, Config config, const OrtLogger &logger);
  ~DayanEp();

private:
  static const char *ORT_API_CALL GetNameImpl(const OrtEp *) noexcept;
  static OrtStatus *ORT_API_CALL GetCapabilityImpl(
      OrtEp *, const OrtGraph *, OrtEpGraphSupportInfo *) noexcept;
  static OrtStatus *ORT_API_CALL
  GetKernelRegistryImpl(OrtEp *, const OrtKernelRegistry **) noexcept;
  static OrtStatus *ORT_API_CALL CompileImpl(
      OrtEp *, const OrtGraph **, const OrtNode **, size_t,
      OrtNodeComputeInfo **, OrtNode **) noexcept;
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp *, OrtNodeComputeInfo **, size_t) noexcept;

  static bool SupportsMatMul(const Ort::ConstNode &node);
  static bool SupportsBinaryElementwise(const Ort::ConstNode &node,
                                        const char *operator_type);
  static bool SupportsConcat(const Ort::ConstNode &node);
  static bool SupportsSplit(const Ort::ConstNode &node);
  static bool SupportsSlice(const Ort::ConstNode &node);
  static bool SupportsResize(const Ort::ConstNode &node);
  static bool SupportsTranspose(const Ort::ConstNode &node);
  static bool SupportsSoftmax(const Ort::ConstNode &node);
  static bool SupportsReduceMax(const Ort::ConstNode &node);
  static bool SupportsArgMax(const Ort::ConstNode &node);
  static bool SupportsSigmoid(const Ort::ConstNode &node);
  static bool SupportsConv(const Ort::ConstNode &node);
  static bool SupportsMaxPool(const Ort::ConstNode &node);
  static bool SupportsTopK(const Ort::ConstNode &node);
  static bool SupportsNms(const Ort::ConstNode &node);
  static bool SupportsGatherNd(const Ort::ConstNode &node);
  static std::optional<Ort::ConstNode>
  MatchSiluPattern(const Ort::ConstNode &mul);
  bool IsOperatorEnabled(const std::string &operator_type) const;

  DayanEpFactory &factory_;
  const OrtApi &api_;
  const OrtEpApi &ep_api_;
  const std::string name_;
  Config config_;
  const OrtLogger &logger_;
  DeviceContextPtr device_context_;
  OrtKernelRegistry *kernel_registry_{};
};

} // namespace dayan
