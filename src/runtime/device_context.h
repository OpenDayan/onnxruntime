#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "NPURuntime.h"
#include "runtime/tensor_utils.h"

namespace rvnpu {

class DeviceContext {
public:
  explicit DeviceContext(std::string kernel_elf);
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
  void RunResize(const float *input, float *output, const ResizeShape &shape);
  void RunTranspose(const float *input, float *output,
                    const TransposeShape &shape);
  void RunSoftmax(const float *input, float *output, const SoftmaxShape &shape);
  void RunReduceMax(const float *input, float *output,
                    const ReduceMaxShape &shape);
  void RunSigmoid(const float *input, float *output,
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
  const std::string &kernel_elf() const noexcept { return kernel_elf_; }

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
    const ssize_t runtime_result = npu_runtime_kernel_start(
        &runtime_, kernel, &kernel_result, sizeof...(Args),
        static_cast<uint64_t>(args)...);
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
  }

  std::string kernel_elf_;
  NPURuntimeState runtime_{};
  bool initialized_{false};
  std::unordered_map<std::string, NPUKernelDesc *> kernels_;
  std::mutex mutex_;
};

using DeviceContextPtr = std::shared_ptr<DeviceContext>;

} // namespace rvnpu
