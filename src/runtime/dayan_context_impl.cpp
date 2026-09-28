#include "runtime/device_context.h"

#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/device_buffer.h"
#include "runtime/cpu_reference.h"

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

void DeviceContext::RunConv(const float *input, const float *weight,
                            const float *bias, float *output,
                            const ConvShape &s) {
  if (!input || !weight || !output || (s.has_bias && !bias))
    throw std::invalid_argument("Conv");
  std::lock_guard lock(mutex_);
  size_t ib = s.input_elements * 4, cb = s.has_bias ? s.bias_elements * 4 : 0,
         ob = s.output_elements * 4, colb = s.column_elements * 4;
  DayanDeviceBuffer in(runtime_, ib), cols(runtime_, colb);
  CopyToDevice(in.Get(), input, ib);
  DAYAN_ATTRS("ort-im2col");
  RequireSuccess(rvnpu_im2col_nchw_fp32(
                     in.Get(), cols.Get(), s.input_channels, s.input_h,
                     s.input_w, s.output_h, s.output_w, s.kernel_h, s.kernel_w,
                     s.stride_h, s.stride_w, s.pad_h, s.pad_w, &attrs),
                 "Conv im2col");
  // im2col 的结果要转 BF16 并补零，取回 host 处理。
  const size_t kpad = AlignUp(static_cast<size_t>(s.matmul_k), 32);
  const size_t npad = AlignUp(static_cast<size_t>(s.matmul_n), 16);
  std::vector<float> columns(s.column_elements);
  CopyFromDevice(columns.data(), cols.Get(), colb);
  if (const char *raw = getenv("RVNPU_DUMP_RAW")) {
    const size_t limit = (size_t)atoll(raw);
    size_t shown = 0;
    for (size_t i = 0; i < limit && shown < 40; ++i) {
      if (columns[i] != input[i]) {
        fprintf(stderr, "[raw] %zu: in=%.3f col=%.3f\n", i, input[i],
                columns[i]);
        ++shown;
      }
    }
    if (shown == 0)
      fprintf(stderr, "[raw] identical for %zu\n", limit);
  }
  if (const char *dump = getenv("RVNPU_DUMP_COLUMNS")) {
    const size_t limit = (size_t)atoll(dump);
    size_t shown = 0;
    for (size_t p = 0; p * s.matmul_k < columns.size() && shown < limit;
         ++p) {
      const size_t py = p / s.output_w, px = p % s.output_w;
      for (size_t k = 0; k < (size_t)s.matmul_k && shown < limit; ++k) {
        const size_t c = k / (s.kernel_h * s.kernel_w);
        const size_t r = k % (s.kernel_h * s.kernel_w);
        const long iy = (long)py * s.stride_h + (long)(r / s.kernel_w) -
                        s.pad_h;
        const long ix = (long)px * s.stride_w + (long)(r % s.kernel_w) -
                        s.pad_w;
        float expect = 0.0f;
        if (iy >= 0 && iy < s.input_h && ix >= 0 && ix < s.input_w)
          expect = input[c * s.input_h * s.input_w + iy * s.input_w + ix];
        const size_t index = p * s.matmul_k + k;
        if (index >= columns.size())
          continue;
        if (columns[index] != expect) {
          fprintf(stderr, "[col] p=%zu(k=%zu,c=%zu) got=%.3f exp=%.3f\n",
                  p, k, c, columns[index], expect);
          ++shown;
        }
      }
    }
  }
  const size_t mm = static_cast<size_t>(s.matmul_m);
  const std::vector<uint16_t> a_bf16 =
      PadToBf16(columns.data(), mm, static_cast<size_t>(s.matmul_k), kpad);
  // weight 是 [output_channels][matmul_k]，按 BT[n][k] 看已经是行主序的 [n][k]，
  // 正好是 MAT 要的 BT 布局，直接补零即可。
  const std::vector<uint16_t> bt_bf16 =
      PadToBf16(weight, static_cast<size_t>(s.matmul_n),
                static_cast<size_t>(s.matmul_k), kpad);
  const size_t ab = a_bf16.size() * 2, btb = bt_bf16.size() * 2,
               ped = mm * npad * 4;
  DayanDeviceBuffer ai(runtime_, ab), bi(runtime_, btb), padded(runtime_, ped);
  CopyToDevice(ai.Get(), a_bf16.data(), ab);
  CopyToDevice(bi.Get(), bt_bf16.data(), btb);
  const MatmulBf16Config config{
      static_cast<uint64_t>(s.matmul_m), static_cast<uint64_t>(npad),
      static_cast<uint64_t>(kpad), kpad * 2, kpad * 2, npad * 4};
  DayanDeviceBuffer cfg(runtime_, sizeof(config));
  CopyToDevice(cfg.Get(), &config, sizeof(config));
  DAYAN_ATTRS_RENAME(attrs, "matmul-bf16-1c");
  RequireSuccess(
      matmul_bf16_1c(cfg.Get(), ai.Get(), bi.Get(), padded.Get(), &attrs),
      "Conv matmul");
  // MatMul 结果按 [row=output_h*output_w][column=output_channels] 排列，
  // ONNX Conv 的输出是 NCHW [output_channels][H*W]：转置落盘，顺带加 bias。
  std::vector<float> tmp(mm * npad);
  CopyFromDevice(tmp.data(), padded.Get(), ped);
  const size_t plane = static_cast<size_t>(s.output_h) *
                       static_cast<size_t>(s.output_w);
  for (size_t row = 0; row < plane; ++row)
    for (size_t column = 0; column < static_cast<size_t>(s.output_channels);
         ++column)
      output[column * plane + row] =
          tmp[row * npad + column] + (s.has_bias ? bias[column] : 0.0f);
}
} // namespace rvnpu
