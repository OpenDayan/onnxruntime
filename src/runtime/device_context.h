#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <chrono>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "NPURuntime.h"
#include "runtime/tensor_utils.h"

namespace rvnpu {

class DeviceContext {
public:
  explicit DeviceContext(std::string kernel_elf, bool cpu_bf16_conv_reference = false);
  ~DeviceContext();

  DeviceContext(const DeviceContext &) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;

  void RunMatMul(const float *a, const float *b, float *y,
                 const MatMulShape &shape);
  void RunAdd(const float *lhs, const float *rhs, float *output,
              const BinaryElementwiseShape &shape);
  void RunMul(const float *lhs, const float *rhs, float *output,
              const BinaryElementwiseShape &shape);
  void RunSub(const float *lhs, const float *rhs, float *output,
              const BinaryElementwiseShape &shape);
  void RunDiv(const float *lhs, const float *rhs, float *output,
              const BinaryElementwiseShape &shape);
  void RunConcat(const std::vector<const float *> &inputs, float *output,
                 const ConcatShape &shape);
  void RunSplit(const float *input, const std::vector<float *> &outputs,
                const SplitShape &shape);
  void RunSlice(const float *input, float *output, const SliceShape &shape);
  void RunSlice(const int64_t *input, int64_t *output,
                const SliceShape &shape);
  void RunResize(const float *input, float *output, const ResizeShape &shape);
  void RunTranspose(const float *input, float *output,
                    const TransposeShape &shape);
  void RunTranspose(const int64_t *input, int64_t *output,
                    const TransposeShape &shape);
  void RunSoftmax(const float *input, float *output, const SoftmaxShape &shape);
  void RunReduceMax(const float *input, float *output,
                    const ReduceMaxShape &shape);
  void RunArgMax(const float *input, int64_t *output,
                 const ArgMaxShape &shape);
  void RunSigmoid(const float *input, float *output,
                  const UnaryTensorShape &shape);
  void RunSilu(const float *input, float *output,
               const UnaryTensorShape &shape);
  void RunConv(const float *input, const float *weight, const float *bias,
               float *output, const ConvShape &shape);
  void RunMaxPool(const float *input, float *output, const MaxPoolShape &shape);
  void RunTopK(const float *input, float *values, int64_t *indices,
               const TopKShape &shape);
  std::vector<int64_t> RunNms(const float *boxes, const float *scores,
                              float iou_threshold, float score_threshold,
                              const NmsShape &shape);
  void RunGatherNd(const float *data, const int64_t *indices, float *output,
                   const GatherNdShape &shape);
  void RunGatherNd(const int64_t *data, const int64_t *indices,
                   int64_t *output, const GatherNdShape &shape);
  const std::string &kernel_elf() const noexcept { return kernel_elf_; }
  bool cpu_bf16_conv_reference() const noexcept { return cpu_bf16_conv_reference_; }

public:
  struct ProfileRecord {
    std::string operator_name;
    std::string kernel_name;
    std::string details;
    uint64_t cycles{};
    double wall_ms{};
  };

  const std::vector<ProfileRecord> &profile_records() const noexcept {
    return profile_records_;
  }
  void ClearProfile() noexcept { profile_records_.clear(); }

private:
  NPUKernelDesc *RequireKernel(const char *name);
  void RunBinaryElementwise(const char *operator_name, const char *kernel_name,
                            const float *lhs, const float *rhs, float *output,
                            const BinaryElementwiseShape &shape);

  template <typename... Args>
  void LaunchKernelLocked(const char *operator_name, const char *kernel_name,
                          const std::string &details, Args... args) {
    static_assert((std::is_convertible_v<Args, uint64_t> && ...),
                  "device kernel arguments must convert to uint64_t");
    NPUKernelDesc *kernel = RequireKernel(kernel_name);
    int kernel_result = -1;
    const auto wall_begin = std::chrono::steady_clock::now();
    const ssize_t runtime_result = npu_runtime_kernel_start(
        &runtime_, kernel, &kernel_result, sizeof...(Args),
        static_cast<uint64_t>(args)...);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_begin)
            .count();
    if (runtime_result != NPU_SUCCESS) {
      throw std::runtime_error(
          std::string(operator_name) + " launch failed: kernel=" + kernel_name +
          details + " runtime code=" + std::to_string(runtime_result));
    }
    if (kernel_result != 0) {
      throw std::runtime_error(std::string(operator_name) +
                               " device kernel=" + kernel_name + details +
                               " returned " + std::to_string(kernel_result));
    }
    // Profiling kernels overwrite __perf_counter[0] with execution cycles.
    uint64_t cycles = 0;
    if (runtime_.kernel_set_desc.perf != 0 &&
        npu_memcpy_from_dev(&runtime_.dev_state, &cycles,
                            runtime_.kernel_set_desc.perf, sizeof(cycles)) ==
            static_cast<ssize_t>(sizeof(cycles))) {
      profile_records_.push_back(
          {operator_name, kernel_name, details, cycles, wall_ms});
    } else {
      profile_records_.push_back({operator_name, kernel_name, details, 0, wall_ms});
    }
  }

  std::string kernel_elf_;
  bool cpu_bf16_conv_reference_ = false;
  NPURuntimeState runtime_{};
  bool initialized_{false};
  std::unordered_map<std::string, NPUKernelDesc *> kernels_;
  std::vector<ProfileRecord> profile_records_;
  std::mutex mutex_;
};

using DeviceContextPtr = std::shared_ptr<DeviceContext>;

} // namespace rvnpu
