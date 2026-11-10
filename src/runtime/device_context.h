#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "dayan_kernel.h"
#include "npu_runtime.h"
#include "runtime/tensor_utils.h"

int rvnpu_add_fp32(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint64_t, const npu_kernel_launch_attrs_t *);
int rvnpu_mul_fp32(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint64_t, const npu_kernel_launch_attrs_t *);
int rvnpu_sub_fp32(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint64_t, const npu_kernel_launch_attrs_t *);
int rvnpu_div_fp32(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint64_t, const npu_kernel_launch_attrs_t *);
int rvnpu_binary_broadcast_fp32(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_sigmoid_f32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_silu_f32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_concat_fp32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_split_fp32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_slice_fp32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_slice_int64(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_resize_nearest_nchw_fp32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_transpose_fp32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_transpose_10_int64(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_softmax_axis1_fp32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_reduce_max_lastdim_fp32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_argmax_lastdim_fp32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_maxpool2d_nchw_fp32(npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_topk_lastdim_fp32(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_gather_nd_fp32(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_gather_nd_int64(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
int rvnpu_nms_fp32(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const npu_kernel_launch_attrs_t *);
// demo matmul_bf16_1c 原版内核：形状与步长打包成 MatmulBf16Config 走 DDR，
// 布局必须与 demo/onnx/kernel_matmul_1c.cpp 里的定义逐字段一致。
struct MatmulBf16Config {
  uint64_t m, n, k;
  uint64_t a_stride, bt_stride, c_stride;
};
int matmul_bf16_1c(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, const npu_kernel_launch_attrs_t *);
int matmul_bf16_1c_small(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, const npu_kernel_launch_attrs_t *);
struct Conv2dTileConfig {
  uint32_t ci, co, wi, hi, wo, ho, wk, hk;
  uint32_t pad_l, pad_u, stride_w, stride_h, dilation_w, dilation_h;
  uint32_t input_stride_w, input_stride_h;
  uint32_t weight_stride_w, weight_stride_h, weight_stride_co;
  uint32_t output_stride_w, output_stride_h;
  uint32_t tile_h, tile_w, tile_ci, tile_co;
  uint32_t pipeline;
};
static_assert(sizeof(Conv2dTileConfig) == 26U * sizeof(uint32_t),
              "Conv host/device ABI mismatch");
bool conv2d_plan_tiles(Conv2dTileConfig *);
#if defined(RVNPU_CONV_PROFILE)
struct alignas(64) Conv2dProfileRecord {
  uint64_t start, end, call_id, task, hart, completed, phase[16];
};
static_assert(sizeof(Conv2dProfileRecord) == 192, "Conv timing ABI mismatch");
int conv2d_tile_4c(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, npu_buffer_t *,
                  npu_buffer_t *, uint64_t, const npu_kernel_launch_attrs_t *);
#else
int conv2d_tile_4c(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, const npu_kernel_launch_attrs_t *);
#endif

namespace rvnpu {

class DayanRuntime final {
public:
  DayanRuntime() = default;
  ~DayanRuntime();
  DayanRuntime(const DayanRuntime &) = delete;
  DayanRuntime &operator=(const DayanRuntime &) = delete;

  npu_context_t *context() { EnsureInitialized(); return context_; }
  npu_stream_t *stream() { EnsureInitialized(); return stream_; }
  bool supports_four_core() { EnsureInitialized(); return four_core_; }
  uint64_t uram_capacity() { EnsureInitialized(); return uram_capacity_; }

private:
  // Graph assignment / all-CPU fallback must not open a device connection.
  // Failed initialization may be retried; call_once publishes only success.
  void EnsureInitialized();
  void Initialize();
  std::once_flag initialized_;
  npu_runtime_t *runtime_ = nullptr;
  npu_context_t *context_ = nullptr;
  npu_stream_t *stream_ = nullptr;
  bool four_core_ = false;
  uint64_t uram_capacity_ = 0;
};

class DeviceContext final {
public:
  // Four-core Conv; the operator plans double-buffered tiles within 1.5 MiB.
  // Shared by EP assignment and execution; this checks capacity, not planning.
  static void ValidateConvResources(const ConvShape &);
  explicit DeviceContext(std::string kernel_elf = {},
                         bool cpu_bf16_conv_reference = false);
  ~DeviceContext();
  DeviceContext(const DeviceContext &) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;

  void RunAdd(const float *lhs, const float *rhs, float *output,
              const BinaryElementwiseShape &shape);
  void RunMul(const float *, const float *, float *,
              const BinaryElementwiseShape &);
  void RunSub(const float *, const float *, float *,
              const BinaryElementwiseShape &);
  void RunDiv(const float *, const float *, float *,
              const BinaryElementwiseShape &);
  void RunMatMul(const float *, const float *, float *, const MatMulShape &);
  void RunConcat(const std::vector<const float *> &, float *, const ConcatShape &);
  void RunSplit(const float *, const std::vector<float *> &, const SplitShape &);
  void RunSlice(const float *, float *, const SliceShape &);
  void RunSlice(const int64_t *, int64_t *, const SliceShape &);
  void RunResize(const float *, float *, const ResizeShape &);
  void RunTranspose(const float *, float *, const TransposeShape &);
  void RunTranspose(const int64_t *, int64_t *, const TransposeShape &);
  void RunSoftmax(const float *, float *, const SoftmaxShape &);
  void RunReduceMax(const float *, float *, const ReduceMaxShape &);
  void RunArgMax(const float *, int64_t *, const ArgMaxShape &);
  void RunSigmoid(const float *, float *, const UnaryTensorShape &);
  void RunSilu(const float *, float *, const UnaryTensorShape &);
  void RunConv(const float *, const float *, const float *, float *,
               const ConvShape &);
  void RunMaxPool(const float *, float *, const MaxPoolShape &);
  void RunTopK(const float *, float *, int64_t *, const TopKShape &);
  std::vector<int64_t> RunNms(const float *, const float *, float, float, const NmsShape &);
  void RunGatherNd(const float *, const int64_t *, float *, const GatherNdShape &);
  void RunGatherNd(const int64_t *, const int64_t *, int64_t *,
                   const GatherNdShape &);

  const std::string &kernel_elf() const noexcept { return kernel_elf_; }
  bool cpu_bf16_conv_reference() const noexcept {
    return cpu_bf16_conv_reference_;
  }

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
  void CopyToDevice(npu_buffer_t *destination, const void *source,
                    size_t bytes);
  void CopyFromDevice(void *destination, npu_buffer_t *source, size_t bytes);
  void RunBinary(const char *, const float *, const float *, float *,
                 const BinaryElementwiseShape &, int);
  void RunUnary(const float *, float *, const UnaryTensorShape &, const char *,
                bool);
  void WaitEvent(npu_event_t *event);

  std::string kernel_elf_;
  bool cpu_bf16_conv_reference_ = false;
  DayanRuntime runtime_;
  std::vector<ProfileRecord> profile_records_;
  std::mutex mutex_;
};

using DeviceContextPtr = std::shared_ptr<DeviceContext>;

} // namespace rvnpu
