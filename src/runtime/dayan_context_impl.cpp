#include "runtime/device_context.h"

#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/device_buffer.h"
#include "runtime/cpu_reference.h"
#include "runtime/tensor_utils.h"

namespace rvnpu {
namespace {

constexpr uint32_t kTimeout = 18000000;

void PackFloat(float value, uint32_t &bits) { std::memcpy(&bits, &value, 4); }

void RequireSuccess(int status, const std::string &stage) {
  if (status != 0)
    throw std::runtime_error(stage + " failed: runtime code=" +
                             std::to_string(status));
}

// MAT 只支持 BF16，且要求 K 是 32 的倍数、N 是 16 的倍数：host 侧把操作数转成
// 补零后的 BF16 布局。FloatToBf16 复用 cpu_reference.h 里的实现。
size_t AlignUp(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// src 是 [rows][cols] 行主序 -> [rows][kpad] 的 BF16，右侧补零。
std::vector<uint16_t> PadToBf16(const float *src, size_t rows, size_t cols,
                                size_t kpad) {
  std::vector<uint16_t> padded(rows * kpad, 0);
  for (size_t row = 0; row < rows; ++row)
    for (size_t column = 0; column < cols; ++column)
      padded[row * kpad + column] = FloatToBf16(src[row * cols + column]);
  return padded;
}

// src 是 [rows][cols] 行主序（rows=k、cols=n）-> [npad][kpad] 的 BF16：
// 一次做完 B[k][n] -> BT[n][k] 的转置和补零。
std::vector<uint16_t> TransposePadToBf16(const float *src, size_t rows,
                                         size_t cols, size_t kpad,
                                         size_t npad) {
  std::vector<uint16_t> padded(npad * kpad, 0);
  for (size_t row = 0; row < rows; ++row)
    for (size_t column = 0; column < cols; ++column)
      padded[column * kpad + row] = FloatToBf16(src[row * cols + column]);
  return padded;
}
} // namespace

// 每次 launch 前填一遍 attrs。这里的 task_dim/task_mode 是启动网格（单核），
// 与内核描述符里声明的 tasks-per-Core 上限是两件事。
#define DAYAN_ATTRS(name)                                                      \
  npu_kernel_launch_attrs_t attrs{};                                           \
  npu_kernel_launch_attrs_init(&attrs, runtime_.context(), runtime_.stream(),  \
                               name);                                          \
  attrs.config.task_dim = NPUTaskDim3{1, 1, 1};                                \
  attrs.config.task_mode = NPUTaskMode::kSingleCore;                           \
  attrs.timeout_ms = kTimeout

#define DAYAN_ATTRS_RENAME(attrs, name)                       \
  npu_kernel_launch_attrs_init(&attrs, runtime_.context(),    \
                               runtime_.stream(), name);      \
  attrs.config.task_dim = NPUTaskDim3{1, 1, 1};               \
  attrs.config.task_mode = NPUTaskMode::kSingleCore;          \
  attrs.timeout_ms = kTimeout

void DeviceContext::RunBinary(const char *op, const float *lhs,
                              const float *rhs, float *out,
                              const BinaryElementwiseShape &shape, int code) {
  if (shape.output_elements == 0)
    return;
  if (!lhs || !rhs || !out)
    throw std::invalid_argument(std::string(op) + " null tensor");
  std::lock_guard lock(mutex_);
  const bool same = shape.lhs_elements == shape.output_elements &&
                    shape.rhs_elements == shape.output_elements;
  const size_t bytes = shape.output_elements * 4;
  DAYAN_ATTRS("ort-binary");
  int status = 0;
  if (same) {
    // 逐元素内核按 1024 个元素一批处理，两侧输入和输出都补零到这个粒度。
    const size_t padded = (shape.output_elements + 1023) / 1024 * 1024;
    std::vector<float> pl(padded, 0.0f), pr(padded, 0.0f);
    std::copy_n(lhs, shape.output_elements, pl.begin());
    std::copy_n(rhs, shape.output_elements, pr.begin());
    DayanDeviceBuffer a(runtime_, padded * 4), b(runtime_, padded * 4),
        c(runtime_, padded * 4);
    CopyToDevice(a.Get(), pl.data(), padded * 4);
    CopyToDevice(b.Get(), pr.data(), padded * 4);
    status =
        code == 0 ? rvnpu_add_fp32(a.Get(), b.Get(), c.Get(), padded, &attrs)
        : code == 1
            ? rvnpu_mul_fp32(a.Get(), b.Get(), c.Get(), padded, &attrs)
        : code == 2
            ? rvnpu_sub_fp32(a.Get(), b.Get(), c.Get(), padded, &attrs)
            : rvnpu_div_fp32(a.Get(), b.Get(), c.Get(), padded, &attrs);
    if (status == 0)
      CopyFromDevice(out, c.Get(), bytes);
  } else {
    // 广播：a 侧按输出形状（连续 NCHW），b 侧保持原始形状，由内核按下标取值。
    const float *a_source = shape.swap_inputs ? rhs : lhs;
    const float *b_source = shape.swap_inputs ? lhs : rhs;
    const size_t b_elements =
        shape.swap_inputs ? shape.lhs_elements : shape.rhs_elements;
    DayanDeviceBuffer a(runtime_, bytes), b(runtime_, b_elements * 4),
        c(runtime_, bytes);
    CopyToDevice(a.Get(), a_source, bytes);
    CopyToDevice(b.Get(), b_source, b_elements * 4);
    const auto &ad = shape.output_dims;
    const auto &bd = shape.broadcast_dims;
    status = rvnpu_binary_broadcast_fp32(a.Get(), b.Get(), c.Get(), ad[0], ad[1],
                                         ad[2], ad[3], bd[0], bd[1], bd[2],
                                         bd[3], code, &attrs);
    if (status == 0)
      CopyFromDevice(out, c.Get(), bytes);
  }
  RequireSuccess(status, std::string(op) + " launch");
}

void DeviceContext::RunMul(const float *l, const float *r, float *o,
                           const BinaryElementwiseShape &s) {
  RunBinary("Mul", l, r, o, s, 1);
}

void DeviceContext::RunSub(const float *l, const float *r, float *o,
                           const BinaryElementwiseShape &s) {
  RunBinary("Sub", l, r, o, s, 2);
}

void DeviceContext::RunDiv(const float *l, const float *r, float *o,
                           const BinaryElementwiseShape &s) {
  RunBinary("Div", l, r, o, s, 3);
}

void DeviceContext::RunUnary(const float *x, float *y,
                             const UnaryTensorShape &s, const char *op,
                             bool silu) {
  if (s.elements == 0)
    return;
  if (!x || !y)
    throw std::invalid_argument(op);
  const size_t padded = (s.elements + 1023) / 1024 * 1024;
  std::vector<float> px;
  if (padded != s.elements) {
    px.assign(padded, 0);
    std::copy_n(x, s.elements, px.begin());
    x = px.data();
  }
  std::lock_guard lock(mutex_);
  size_t b = padded * 4, ob = s.elements * 4;
  DayanDeviceBuffer i(runtime_, b), o(runtime_, b);
  CopyToDevice(i.Get(), x, b);
  DAYAN_ATTRS("ort-unary");
  int st = silu ? rvnpu_silu_f32(i.Get(), o.Get(), 1U, 1U, 1U, uint32_t(padded),
                                 &attrs)
                : rvnpu_sigmoid_f32(i.Get(), o.Get(), 1U, 1U, 1U,
                                    uint32_t(padded), &attrs);
  RequireSuccess(st, op);
  CopyFromDevice(y, o.Get(), ob);
}

void DeviceContext::RunSigmoid(const float *x, float *y,
                               const UnaryTensorShape &s) {
  RunUnary(x, y, s, "Sigmoid", false);
}

void DeviceContext::RunSilu(const float *x, float *y,
                            const UnaryTensorShape &s) {
  RunUnary(x, y, s, "SiLU", true);
}

void DeviceContext::RunConcat(const std::vector<const float *> &inputs,
                              float *output, const ConcatShape &s) {
  if (inputs.size() != s.input_axes.size())
    throw std::invalid_argument("Concat input count");
  if (s.output_elements == 0)
    return;
  if (!output)
    throw std::invalid_argument("Concat");
  std::lock_guard lock(mutex_);
  size_t outb = s.output_elements * 4;
  DayanDeviceBuffer out(runtime_, outb);
  CopyToDevice(out.Get(), output, outb);
  DAYAN_ATTRS("ort-concat");
  int32_t axis_offset = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const size_t b = size_t(s.outer) * s.input_axes[i] * s.inner * 4;
    if (!b) {
      axis_offset += int32_t(s.input_axes[i]);
      continue; // 空张量不参与拼接
    }
    if (!inputs[i])
      throw std::invalid_argument("Concat input");
    DayanDeviceBuffer in(runtime_, b);
    CopyToDevice(in.Get(), inputs[i], b);
    RequireSuccess(rvnpu_concat_fp32(in.Get(), out.Get(), s.outer,
                                     s.input_axes[i], s.output_axis, s.inner,
                                     uint32_t(axis_offset), &attrs),
                   "Concat");
    axis_offset += int32_t(s.input_axes[i]);
  }
  CopyFromDevice(output, out.Get(), outb);
}

void DeviceContext::RunSplit(const float *input,
                             const std::vector<float *> &outputs,
                             const SplitShape &s) {
  if (input == nullptr || outputs.size() != s.output_axes.size())
    throw std::invalid_argument("Split");
  std::lock_guard lock(mutex_);
  size_t b = s.input_elements * 4;
  DayanDeviceBuffer in(runtime_, b);
  CopyToDevice(in.Get(), input, b);
  int32_t axis_offset = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const size_t ob = size_t(s.outer) * s.output_axes[i] * s.inner * 4;
    if (!ob) {
      axis_offset += int32_t(s.output_axes[i]);
      continue; // 空张量没有内容可搬
    }
    if (!outputs[i])
      throw std::invalid_argument("Split output");
    DayanDeviceBuffer out(runtime_, ob);
    DAYAN_ATTRS("ort-split");
    RequireSuccess(rvnpu_split_fp32(in.Get(), out.Get(), s.outer, s.input_axis,
                                    s.output_axes[i], s.inner,
                                    uint32_t(axis_offset), &attrs),
                   "Split");
    CopyFromDevice(outputs[i], out.Get(), ob);
    axis_offset += int32_t(s.output_axes[i]);
  }
}

void DeviceContext::RunSlice(const float *input, float *output,
                             const SliceShape &s) {
  if (s.output_elements == 0)
    return;
  if (!input || !output)
    throw std::invalid_argument("Slice");
  std::lock_guard lock(mutex_);
  size_t ib = s.input_elements * 4, ob = s.output_elements * 4;
  DayanDeviceBuffer in(runtime_, ib), out(runtime_, ob);
  CopyToDevice(in.Get(), input, ib);
  DAYAN_ATTRS("ort-slice");
  RequireSuccess(rvnpu_slice_fp32(in.Get(), out.Get(), s.outer, s.input_axis,
                                  s.start, s.length, s.step, s.inner, &attrs),
                 "Slice");
  CopyFromDevice(output, out.Get(), ob);
}

void DeviceContext::RunSlice(const int64_t *input, int64_t *output,
                             const SliceShape &s) {
  if (s.output_elements == 0)
    return;
  if (!input || !output)
    throw std::invalid_argument("Slice int64");
  std::lock_guard lock(mutex_);
  size_t ib = s.input_elements * 8, ob = s.output_elements * 8;
  DayanDeviceBuffer in(runtime_, ib), out(runtime_, ob);
  CopyToDevice(in.Get(), input, ib);
  DAYAN_ATTRS("ort-slice-i64");
  RequireSuccess(rvnpu_slice_int64(in.Get(), out.Get(), s.outer, s.input_axis,
                                   s.start, s.length, s.step, s.inner, &attrs),
                 "Slice int64");
  CopyFromDevice(output, out.Get(), ob);
}

void DeviceContext::RunResize(const float *input, float *output,
                              const ResizeShape &s) {
  if (!input || !output)
    throw std::invalid_argument("Resize");
  std::lock_guard lock(mutex_);
  size_t ib = s.input_elements * 4, ob = s.output_elements * 4;
  DayanDeviceBuffer in(runtime_, ib), out(runtime_, ob);
  CopyToDevice(in.Get(), input, ib);
  DAYAN_ATTRS("ort-resize");
  RequireSuccess(rvnpu_resize_nearest_nchw_fp32(
                     in.Get(), out.Get(), s.batch, s.channels, s.input_h,
                     s.input_w, s.output_h, s.output_w, &attrs),
                 "Resize");
  CopyFromDevice(output, out.Get(), ob);
}

void DeviceContext::RunTranspose(const float *input, float *output,
                                 const TransposeShape &s) {
  if (!input || !output)
    throw std::invalid_argument("Transpose");
  std::lock_guard lock(mutex_);
  size_t b = s.elements * 4;
  DayanDeviceBuffer in(runtime_, b), out(runtime_, b);
  CopyToDevice(in.Get(), input, b);
  DAYAN_ATTRS("ort-transpose");
  int mode = s.rank == 2 ? 0 : s.rank == 3 ? 1 : 2;
  RequireSuccess(rvnpu_transpose_fp32(in.Get(), out.Get(), s.dims[0], s.dims[1],
                                      s.dims[2], s.dims[3], mode, &attrs),
                 "Transpose");
  CopyFromDevice(output, out.Get(), b);
}

void DeviceContext::RunTranspose(const int64_t *input, int64_t *output,
                                 const TransposeShape &s) {
  if (s.elements == 0)
    return;
  if (!input || !output)
    throw std::invalid_argument("Transpose int64");
  if (s.rank != 2)
    throw std::invalid_argument("Transpose int64 requires rank 2");
  std::lock_guard lock(mutex_);
  size_t b = s.elements * 8;
  DayanDeviceBuffer in(runtime_, b), out(runtime_, b);
  CopyToDevice(in.Get(), input, b);
  DAYAN_ATTRS("ort-trans-i64");
  RequireSuccess(rvnpu_transpose_10_int64(in.Get(), out.Get(), s.dims[0],
                                          s.dims[1], &attrs),
                 "Transpose int64");
  CopyFromDevice(output, out.Get(), b);
}

void DeviceContext::RunSoftmax(const float *x, float *y,
                               const SoftmaxShape &s) {
  if (!x || !y)
    throw std::invalid_argument("Softmax");
  std::lock_guard lock(mutex_);
  size_t b = s.elements * 4;
  DayanDeviceBuffer i(runtime_, b), o(runtime_, b);
  CopyToDevice(i.Get(), x, b);
  DAYAN_ATTRS("ort-softmax");
  RequireSuccess(
      rvnpu_softmax_axis1_fp32(i.Get(), o.Get(), s.outer, s.axis, s.inner, &attrs),
      "Softmax");
  CopyFromDevice(y, o.Get(), b);
}

void DeviceContext::RunReduceMax(const float *x, float *y,
                                 const ReduceMaxShape &s) {
  if (!x || !y)
    throw std::invalid_argument("ReduceMax");
  std::lock_guard lock(mutex_);
  size_t ib = s.input_elements * 4, ob = s.output_elements * 4;
  DayanDeviceBuffer in(runtime_, ib), out(runtime_, ob);
  CopyToDevice(in.Get(), x, ib);
  DAYAN_ATTRS("ort-reduce");
  RequireSuccess(rvnpu_reduce_max_lastdim_fp32(in.Get(), out.Get(), s.rows,
                                               s.columns, &attrs),
                 "ReduceMax");
  CopyFromDevice(y, out.Get(), ob);
}

void DeviceContext::RunArgMax(const float *x, int64_t *y,
                              const ArgMaxShape &s) {
  if (!x || !y)
    throw std::invalid_argument("ArgMax");
  std::lock_guard lock(mutex_);
  size_t ib = s.input_elements * 4, ob = s.output_elements * 8;
  DayanDeviceBuffer in(runtime_, ib), out(runtime_, ob);
  CopyToDevice(in.Get(), x, ib);
  DAYAN_ATTRS("ort-argmax");
  RequireSuccess(
      rvnpu_argmax_lastdim_fp32(in.Get(), out.Get(), s.rows, s.columns, 0, &attrs),
      "ArgMax");
  CopyFromDevice(y, out.Get(), ob);
}

void DeviceContext::RunMaxPool(const float *x, float *y,
                               const MaxPoolShape &s) {
  if (!x || !y)
    throw std::invalid_argument("MaxPool");
  std::lock_guard lock(mutex_);
  size_t ib = s.input_elements * 4, ob = s.output_elements * 4;
  DayanDeviceBuffer in(runtime_, ib), out(runtime_, ob);
  CopyToDevice(in.Get(), x, ib);
  DAYAN_ATTRS("ort-maxpool");
  RequireSuccess(rvnpu_maxpool2d_nchw_fp32(
                     in.Get(), out.Get(), s.batch, s.channels, s.input_h,
                     s.input_w, s.output_h, s.output_w, s.kernel_h, s.kernel_w,
                     s.stride_h, s.stride_w, s.pad_h, s.pad_w, &attrs),
                 "MaxPool");
  CopyFromDevice(y, out.Get(), ob);
}

void DeviceContext::RunTopK(const float *x, float *v, int64_t *i,
                            const TopKShape &s) {
  if (!x || !v || !i)
    throw std::invalid_argument("TopK");
  std::lock_guard lock(mutex_);
  size_t ib = s.input_elements * 4, vb = s.output_elements * 4,
         ii = s.output_elements * 8;
  DayanDeviceBuffer in(runtime_, ib), ov(runtime_, vb), oi(runtime_, ii);
  CopyToDevice(in.Get(), x, ib);
  DAYAN_ATTRS("ort-topk");
  RequireSuccess(rvnpu_topk_lastdim_fp32(in.Get(), ov.Get(), oi.Get(), s.rows,
                                         s.columns, s.k, &attrs),
                 "TopK");
  CopyFromDevice(v, ov.Get(), vb);
  CopyFromDevice(i, oi.Get(), ii);
}

void DeviceContext::RunGatherNd(const float *data, const int64_t *indices,
                                float *out, const GatherNdShape &s) {
  if (s.rows == 0)
    return;
  if (!data || !indices || !out)
    throw std::invalid_argument("GatherND");
  std::lock_guard lock(mutex_);
  size_t db = s.data_elements * 4, ib = s.indices_elements * 8,
         ob = s.output_elements * 4;
  DayanDeviceBuffer d(runtime_, db), in(runtime_, ib), o(runtime_, ob);
  CopyToDevice(d.Get(), data, db);
  CopyToDevice(in.Get(), indices, ib);
  DAYAN_ATTRS("ort-gather");
  RequireSuccess(rvnpu_gather_nd_fp32(d.Get(), in.Get(), o.Get(), s.rows,
                                      s.index_depth, s.data_dims[0],
                                      s.data_dims[1], s.data_dims[2], &attrs),
                 "GatherND");
  CopyFromDevice(out, o.Get(), ob);
}

void DeviceContext::RunGatherNd(const int64_t *data, const int64_t *indices,
                                int64_t *out, const GatherNdShape &s) {
  if (s.rows == 0)
    return;
  if (!data || !indices || !out)
    throw std::invalid_argument("GatherND int64");
  std::lock_guard lock(mutex_);
  size_t db = s.data_elements * 8, ib = s.indices_elements * 8,
         ob = s.output_elements * 8;
  DayanDeviceBuffer d(runtime_, db), in(runtime_, ib), o(runtime_, ob);
  CopyToDevice(d.Get(), data, db);
  CopyToDevice(in.Get(), indices, ib);
  DAYAN_ATTRS("ort-gather-i64");
  RequireSuccess(rvnpu_gather_nd_int64(d.Get(), in.Get(), o.Get(), s.rows,
                                       s.index_depth, s.data_dims[0],
                                       s.data_dims[1], s.data_dims[2], &attrs),
                 "GatherND int64");
  CopyFromDevice(out, o.Get(), ob);
}

std::vector<int64_t> DeviceContext::RunNms(const float *boxes,
                                           const float *scores, float iou,
                                           float score, const NmsShape &s) {
  if (s.num_boxes == 0)
    return {};
  if (!boxes || !scores)
    throw std::invalid_argument("NMS");
  std::lock_guard lock(mutex_);
  size_t bb = s.boxes_elements * 4, sb = s.scores_elements * 4,
         ob = s.max_selected * 3 * 8;
  DayanDeviceBuffer bi(runtime_, bb), si(runtime_, sb), out(runtime_, ob),
      count(runtime_, 8);
  CopyToDevice(bi.Get(), boxes, bb);
  CopyToDevice(si.Get(), scores, sb);
  uint64_t zero = 0;
  CopyToDevice(count.Get(), &zero, 8);
  uint32_t ibits, sbits;
  PackFloat(iou, ibits);
  PackFloat(score, sbits);
  DAYAN_ATTRS("ort-nms");
  RequireSuccess(rvnpu_nms_fp32(bi.Get(), si.Get(), out.Get(), count.Get(),
                                s.num_boxes, s.num_classes,
                                s.max_output_per_class, ibits, sbits,
                                s.center_point_box, &attrs),
                 "NMS");
  CopyFromDevice(&zero, count.Get(), 8);
  if (zero > s.max_selected)
    throw std::runtime_error("NMS invalid count");
  std::vector<int64_t> result(size_t(zero) * 3);
  if (!result.empty())
    CopyFromDevice(result.data(), out.Get(), result.size() * 8);
  return result;
}

void DeviceContext::RunMatMul(const float *a, const float *b, float *y,
                              const MatMulShape &s) {
  if (!a || !b || !y)
    throw std::invalid_argument("MatMul");
  if (s.m == 0 || s.n == 0 || s.k == 0)
    return;
  std::lock_guard lock(mutex_);
  // A/B 转成 BF16 并按 MAT 的对齐要求补零：kpad 是 32 的倍数、npad 是 16 的倍数。
  // 补零处乘出来是 0，不影响结果。
  const size_t kpad = AlignUp(static_cast<size_t>(s.k), 32);
  const size_t npad = AlignUp(static_cast<size_t>(s.n), 16);
  const std::vector<uint16_t> a_bf16 =
      PadToBf16(a, static_cast<size_t>(s.m), static_cast<size_t>(s.k), kpad);
  const std::vector<uint16_t> bt_bf16 =
      TransposePadToBf16(b, static_cast<size_t>(s.k), static_cast<size_t>(s.n),
                         kpad, npad);
  const size_t ab = a_bf16.size() * 2, btb = bt_bf16.size() * 2,
               ob = static_cast<size_t>(s.m) * npad * 4;
  DayanDeviceBuffer ai(runtime_, ab), bi(runtime_, btb), out(runtime_, ob);
  CopyToDevice(ai.Get(), a_bf16.data(), ab);
  CopyToDevice(bi.Get(), bt_bf16.data(), btb);
  // GEMM 换用 demo 的 matmul_bf16_1c 原版内核：形状/步长打包成 config 走 DDR。
  const MatmulBf16Config config{
      static_cast<uint64_t>(s.m), static_cast<uint64_t>(npad),
      static_cast<uint64_t>(kpad), kpad * 2, kpad * 2, npad * 4};
  DayanDeviceBuffer cfg(runtime_, sizeof(config));
  CopyToDevice(cfg.Get(), &config, sizeof(config));
  DAYAN_ATTRS("matmul-bf16-1c");
  RequireSuccess(
      matmul_bf16_1c(cfg.Get(), ai.Get(), bi.Get(), out.Get(), &attrs),
      "MatMul");
  // C 按 [m][npad] 落盘，只取回前 n 列。
  std::vector<float> padded(static_cast<size_t>(s.m) * npad);
  CopyFromDevice(padded.data(), out.Get(), ob);
  for (int32_t row = 0; row < s.m; ++row)
    std::copy_n(&padded[static_cast<size_t>(row) * npad], s.n,
                &y[static_cast<size_t>(row) * s.n]);
}

void DeviceContext::ValidateConvResources(const ConvShape &s) {
  const uint64_t ci = s.input_channels_per_group;
  const uint64_t co = s.output_channels_per_group;
  const uint64_t limit = std::numeric_limits<uint32_t>::max();
  if (!ci || !co || ci * 2 > limit || ci * s.input_w * 2 > limit ||
      ci * co * 2 > limit || ci * co * s.kernel_w * 2 > limit ||
      co * 4 > limit || co * s.output_w * 4 > limit)
    throw std::invalid_argument("Conv device byte stride exceeds uint32");
  // Conservative per-invocation release envelope, not total/free DRAM.
  // The qualified RTL server uses a 4 GiB normal-memory region. Keep ample
  // room for firmware, launch descriptors and other operators. RunConv keeps
  // only one group and one batch's input/output buffers resident at a time.
  constexpr uint64_t resident_limit = 256ULL * 1024 * 1024;
  const uint64_t input_bytes = CheckedMultiply(
      CheckedMultiply(ci, s.input_h, "Conv resident input"),
      uint64_t(s.input_w) * 2, "Conv resident input bytes");
  const uint64_t weight_bytes = CheckedMultiply(
      CheckedMultiply(ci, co, "Conv resident weight"),
      uint64_t(s.kernel_h) * s.kernel_w * 2, "Conv resident weight bytes");
  const uint64_t output_bytes = CheckedMultiply(
      CheckedMultiply(co, s.output_h, "Conv resident output"),
      uint64_t(s.output_w) * 4, "Conv resident output bytes");
  // Server rounds each normal Buffer to 4 KiB with its MMU enabled.
  auto page_bytes = [](uint64_t bytes) {
    return (bytes + 4095) & ~uint64_t(4095);
  };
  if (input_bytes > resident_limit || weight_bytes > resident_limit ||
      output_bytes > resident_limit ||
      page_bytes(input_bytes) + page_bytes(weight_bytes) +
          page_bytes(output_bytes) + 4096 > resident_limit)
    throw std::invalid_argument("Conv resident buffers exceed 256 MiB envelope");
  // Conservative release envelope, not a claim about undocumented register
  // maxima. Broaden only with device numerical tests.
  if (s.kernel_h > 7 || s.kernel_w > 7 || s.stride_h > 3 || s.stride_w > 3 ||
      s.dilation_h > 3 || s.dilation_w > 3)
    throw std::invalid_argument(
        "Conv outside tested device candidate envelope");
  const uint64_t ih = uint64_t(kConvTileH - 1) * s.stride_h +
                      uint64_t(s.kernel_h - 1) * s.dilation_h + 1;
  const uint64_t iw = uint64_t(kConvTileW - 1) * s.stride_w +
                      uint64_t(s.kernel_w - 1) * s.dilation_w + 1;
  auto align = [](uint64_t bytes) { return (bytes + 63) & ~uint64_t(63); };
  const uint64_t uram_bytes = 2 *
      (align(ih * iw * kConvTileCi * 2) +
       align(uint64_t(s.kernel_h) * s.kernel_w * kConvTileCi * kConvTileCo * 2) +
       align(uint64_t(kConvTileH) * kConvTileW * kConvTileCo * 4));
  if (uram_bytes > 65536)
    throw std::invalid_argument("Conv fixed tile exceeds 64 KiB URAM");
}

void DeviceContext::RunConv(const float *input, const float *weight,
                            const float *bias, float *output,
                            const ConvShape &shape) {
  if (!input || !weight || !output || (shape.has_bias && !bias))
    throw std::invalid_argument("Conv");
  ValidateConvResources(shape);
  std::lock_guard lock(mutex_);
  ConvShape s = shape;
  s.input_channels = shape.input_channels_per_group;
  s.output_channels = shape.output_channels_per_group;
  const size_t input_group_size =
      size_t(s.input_channels) * s.input_h * s.input_w;
  const size_t output_group_size =
      size_t(s.output_channels) * s.output_h * s.output_w;
  const size_t weight_group_size =
      size_t(s.output_channels) * s.input_channels * s.kernel_h * s.kernel_w;
  const float *all_input = input;
  const float *all_weight = weight;
  const float *all_bias = bias;
  float *all_output = output;
  for (int32_t group = 0; group < shape.group; ++group) {
    weight = all_weight + size_t(group) * weight_group_size;
    bias = all_bias ? all_bias + size_t(group) * s.output_channels : nullptr;
    const size_t input_plane = static_cast<size_t>(s.input_h) * s.input_w;
    const size_t input_stride =
        static_cast<size_t>(s.input_channels) * sizeof(uint16_t);
    const size_t input_row_stride =
        static_cast<size_t>(s.input_w) * input_stride;
    const size_t weight_channel_stride =
        static_cast<size_t>(s.input_channels) * sizeof(uint16_t);
    const size_t weight_col_stride =
        static_cast<size_t>(s.output_channels) * weight_channel_stride;
    const size_t weight_row_stride =
        static_cast<size_t>(s.kernel_w) * weight_col_stride;
    const size_t output_stride =
        static_cast<size_t>(s.output_channels) * sizeof(float);
    const size_t output_row_stride =
        static_cast<size_t>(s.output_w) * output_stride;

    std::vector<uint16_t> packed_input(input_plane * s.input_channels);

    std::vector<uint16_t> packed_weight(static_cast<size_t>(s.kernel_h) *
                                        s.kernel_w * s.output_channels *
                                        s.input_channels);
    for (int32_t output_channel = 0; output_channel < s.output_channels;
         ++output_channel)
      for (int32_t input_channel = 0; input_channel < s.input_channels;
           ++input_channel)
        for (int32_t kernel_height = 0; kernel_height < s.kernel_h;
             ++kernel_height)
          for (int32_t kernel_width = 0; kernel_width < s.kernel_w;
               ++kernel_width) {
            const size_t source =
                ((static_cast<size_t>(output_channel) * s.input_channels +
                  input_channel) *
                     s.kernel_h +
                 kernel_height) *
                    s.kernel_w +
                kernel_width;
            const size_t destination =
                ((static_cast<size_t>(kernel_height) * s.kernel_w +
                  kernel_width) *
                     s.output_channels +
                 output_channel) *
                    s.input_channels +
                input_channel;
            packed_weight[destination] = FloatToBf16(weight[source]);
          }

    std::vector<float> packed_output(static_cast<size_t>(s.output_h) *
                                     s.output_w * s.output_channels);
    DayanDeviceBuffer in(runtime_, packed_input.size() * sizeof(uint16_t));
    DayanDeviceBuffer weights(runtime_,
                              packed_weight.size() * sizeof(uint16_t));
    DayanDeviceBuffer out(runtime_, packed_output.size() * sizeof(float));
    CopyToDevice(weights.Get(), packed_weight.data(),
                 packed_weight.size() * sizeof(uint16_t));

    const Conv2dTileConfig config{static_cast<uint32_t>(s.input_channels),
                                  static_cast<uint32_t>(s.output_channels),
                                  static_cast<uint32_t>(s.input_w),
                                  static_cast<uint32_t>(s.input_h),
                                  static_cast<uint32_t>(s.output_w),
                                  static_cast<uint32_t>(s.output_h),
                                  static_cast<uint32_t>(s.kernel_w),
                                  static_cast<uint32_t>(s.kernel_h),
                                  static_cast<uint32_t>(s.pad_w),
                                  static_cast<uint32_t>(s.pad_h),
                                  static_cast<uint32_t>(s.stride_w),
                                  static_cast<uint32_t>(s.stride_h),
                                  static_cast<uint32_t>(s.dilation_w),
                                  static_cast<uint32_t>(s.dilation_h),
                                  static_cast<uint32_t>(input_stride),
                                  static_cast<uint32_t>(input_row_stride),
                                  static_cast<uint32_t>(weight_col_stride),
                                  static_cast<uint32_t>(weight_row_stride),
                                  static_cast<uint32_t>(weight_channel_stride),
                                  static_cast<uint32_t>(output_stride),
                                  static_cast<uint32_t>(output_row_stride),
                                  kConvTileH,
                                  kConvTileW,
                                  kConvTileCi,
                                  kConvTileCo,
        1U};
    DayanDeviceBuffer config_buffer(runtime_, sizeof(config));
    CopyToDevice(config_buffer.Get(), &config, sizeof(config));

    for (int32_t batch = 0; batch < shape.batch; ++batch) {
      input =
          all_input + (size_t(batch) * shape.group + group) * input_group_size;
      output = all_output +
               (size_t(batch) * shape.group + group) * output_group_size;
      for (int32_t channel = 0; channel < s.input_channels; ++channel)
        for (size_t pixel = 0; pixel < input_plane; ++pixel)
          packed_input[pixel * s.input_channels + channel] =
              FloatToBf16(input[size_t(channel) * input_plane + pixel]);
      CopyToDevice(in.Get(), packed_input.data(),
                   packed_input.size() * sizeof(uint16_t));
      npu_kernel_launch_attrs_t attrs{};
      npu_kernel_launch_attrs_init(&attrs, runtime_.context(),
                                   runtime_.stream(), "conv2d_tile_4c");
      attrs.config.task_dim = NPUTaskDim3{4U, 1, 1};
      attrs.config.task_mode =
          NPUTaskMode::kFourCore;
      attrs.timeout_ms = kTimeout;
      RequireSuccess(conv2d_tile_4c(config_buffer.Get(), in.Get(),
                                   weights.Get(), out.Get(), &attrs),
                     "Conv tiled 4c");
      CopyFromDevice(packed_output.data(), out.Get(),
                     packed_output.size() * sizeof(float));

      for (int32_t output_channel = 0; output_channel < s.output_channels;
           ++output_channel)
        for (int32_t height = 0; height < s.output_h; ++height)
          for (int32_t width = 0; width < s.output_w; ++width) {
            const size_t source =
                (static_cast<size_t>(height) * s.output_w + width) *
                    s.output_channels +
                output_channel;
            const size_t destination =
                (static_cast<size_t>(output_channel) * s.output_h + height) *
                    s.output_w +
                width;
            output[destination] =
                packed_output[source] + (bias ? bias[output_channel] : 0.0f);
          }
    }
  }

}

} // namespace rvnpu
