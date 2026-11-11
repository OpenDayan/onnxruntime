#include "runtime/device_context.h"

#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(DAYAN_CONV_PROFILE)
#include <atomic>
#endif

#include "runtime/device_buffer.h"
#include "runtime/cpu_reference.h"
#include "runtime/tensor_utils.h"

namespace dayan {
namespace {

constexpr uint32_t kTimeout = 18000000;

void PackFloat(float value, uint32_t &bits) { std::memcpy(&bits, &value, 4); }

void RequireSuccess(int status, const std::string &stage) {
  if (status != 0)
    throw std::runtime_error(stage + " failed: runtime code=" +
                             std::to_string(status));
}

void ConfigureLargeTransfer(npu_kernel_launch_attrs_t &attrs,
                            bool four_core_available, size_t elements) {
  constexpr size_t kFourCoreThreshold = 8192;
  if (four_core_available && elements >= kFourCoreThreshold) {
    attrs.config.task_dim = NPUTaskDim3{4U, 1U, 1U};
    attrs.config.task_mode = NPUTaskMode::kFourCore;
  }
}

size_t AlignUp(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

std::vector<uint16_t> PadToBf16(const float *src, size_t rows, size_t cols,
                                size_t kpad) {
  std::vector<uint16_t> padded(rows * kpad, 0);
  for (size_t row = 0; row < rows; ++row)
    for (size_t column = 0; column < cols; ++column)
      padded[row * kpad + column] = FloatToBf16(src[row * cols + column]);
  return padded;
}

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
  const bool named_profile = std::getenv("DAYAN_PROFILE_OPERATOR_NAMES") != nullptr;
  const char *binary_task = code == 0 ? "ort-Add" : code == 1 ? "ort-Mul"
      : code == 2 ? "ort-Sub" : "ort-Div";
  DAYAN_ATTRS(named_profile ? binary_task : "ort-binary");
  if (same && runtime_.supports_four_core() && shape.output_elements >= 8192) {
    attrs.config.task_dim = NPUTaskDim3{4U, 1U, 1U};
    attrs.config.task_mode = NPUTaskMode::kFourCore;
  }
  int status = 0;
  if (same) {
    const size_t padded = (shape.output_elements + 1023) / 1024 * 1024;
    std::vector<float> pl(padded, 0.0f), pr(padded, 0.0f);
    std::copy_n(lhs, shape.output_elements, pl.begin());
    std::copy_n(rhs, shape.output_elements, pr.begin());
    DayanDeviceBuffer a(runtime_, padded * 4), b(runtime_, padded * 4),
        c(runtime_, padded * 4);
    CopyToDevice(a.Get(), pl.data(), padded * 4);
    CopyToDevice(b.Get(), pr.data(), padded * 4);
    status =
        code == 0 ? dayan_add_fp32(a.Get(), b.Get(), c.Get(), padded, &attrs)
        : code == 1
            ? dayan_mul_fp32(a.Get(), b.Get(), c.Get(), padded, &attrs)
        : code == 2
            ? dayan_sub_fp32(a.Get(), b.Get(), c.Get(), padded, &attrs)
            : dayan_div_fp32(a.Get(), b.Get(), c.Get(), padded, &attrs);
    if (status == 0)
      CopyFromDevice(out, c.Get(), bytes);
  } else {
    const bool commutative_swap = shape.swap_inputs && code <= 1;
    const size_t a_elements =
        commutative_swap ? shape.rhs_elements : shape.lhs_elements;
    const size_t b_elements =
        commutative_swap ? shape.lhs_elements : shape.rhs_elements;
    DayanDeviceBuffer a(runtime_, a_elements * 4),
        b(runtime_, b_elements * 4), c(runtime_, bytes);
    CopyToDevice(a.Get(), commutative_swap ? rhs : lhs, a_elements * 4);
    CopyToDevice(b.Get(), commutative_swap ? lhs : rhs, b_elements * 4);
    const auto &ld = commutative_swap ? shape.output_dims : shape.lhs_dims;
    const auto &rd = commutative_swap ? shape.lhs_dims : shape.rhs_dims;
    uint64_t contiguous = 1;
    for (size_t dimension = 0; dimension < ld.size(); ++dimension) {
      if (ld[dimension] != rd[dimension])
        break;
      contiguous *= ld[dimension];
    }
    if (code <= 1 && ld == shape.output_dims && contiguous >= 16 &&
        contiguous % 16 == 0) {
      ConfigureLargeTransfer(attrs, runtime_.supports_four_core(),
                             shape.output_elements);
    }
    status = dayan_binary_broadcast_fp32(
        a.Get(), b.Get(), c.Get(), ld[0], ld[1], ld[2], ld[3], rd[0], rd[1],
        rd[2], rd[3], code, &attrs);
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
  const bool named_profile = std::getenv("DAYAN_PROFILE_OPERATOR_NAMES") != nullptr;
  DAYAN_ATTRS(named_profile ? (silu ? "ort-Silu" : "ort-Sigmoid") : "ort-unary");
  if (padded >= 8192U && runtime_.supports_four_core()) {
    attrs.config.task_dim = NPUTaskDim3{4U, 1, 1};
    attrs.config.task_mode = NPUTaskMode::kFourCore;
  }
  int st = silu ? dayan_silu_f32(i.Get(), o.Get(), 1U, 1U, 1U, uint32_t(padded),
                                 &attrs)
                : dayan_sigmoid_f32(i.Get(), o.Get(), 1U, 1U, 1U,
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
    ConfigureLargeTransfer(
        attrs, runtime_.supports_four_core(),
        size_t(s.outer) * s.input_axes[i] * s.inner);
    RequireSuccess(dayan_concat_fp32(in.Get(), out.Get(), s.outer,
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
    ConfigureLargeTransfer(
        attrs, runtime_.supports_four_core(),
        size_t(s.outer) * s.output_axes[i] * s.inner);
    RequireSuccess(dayan_split_fp32(in.Get(), out.Get(), s.outer, s.input_axis,
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(),
                         s.output_elements);
  RequireSuccess(dayan_slice_fp32(in.Get(), out.Get(), s.outer, s.input_axis,
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(),
                         s.output_elements);
  RequireSuccess(dayan_slice_int64(in.Get(), out.Get(), s.outer, s.input_axis,
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(),
                         s.output_elements);
  RequireSuccess(dayan_resize_nearest_nchw_fp32(
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(), s.elements);
  int mode = s.rank == 2 ? 0 : s.rank == 3 ? 1 : 2;
  RequireSuccess(dayan_transpose_fp32(in.Get(), out.Get(), s.dims[0], s.dims[1],
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(), s.elements);
  RequireSuccess(dayan_transpose_10_int64(in.Get(), out.Get(), s.dims[0],
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(), s.elements);
  RequireSuccess(
      dayan_softmax_axis1_fp32(i.Get(), o.Get(), s.outer, s.axis, s.inner, &attrs),
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(),
                         static_cast<size_t>(s.rows) * s.columns);
  RequireSuccess(dayan_reduce_max_lastdim_fp32(in.Get(), out.Get(), s.rows,
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(),
                         static_cast<size_t>(s.rows) * s.columns);
  RequireSuccess(
      dayan_argmax_lastdim_fp32(in.Get(), out.Get(), s.rows, s.columns, 0, &attrs),
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
  // 四核 MaxPool 的 5x5 YOLO 形状尚未通过 RTL 验证，保持单核。
  RequireSuccess(dayan_maxpool2d_nchw_fp32(
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(), s.rows);
  RequireSuccess(dayan_topk_lastdim_fp32(in.Get(), ov.Get(), oi.Get(), s.rows,
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(), s.rows);
  RequireSuccess(dayan_gather_nd_fp32(d.Get(), in.Get(), o.Get(), s.rows,
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
  ConfigureLargeTransfer(attrs, runtime_.supports_four_core(), s.rows);
  RequireSuccess(dayan_gather_nd_int64(d.Get(), in.Get(), o.Get(), s.rows,
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
  RequireSuccess(dayan_nms_fp32(bi.Get(), si.Get(), out.Get(), count.Get(),
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
  const MatmulBf16Config config{
      static_cast<uint64_t>(s.m), static_cast<uint64_t>(npad),
      static_cast<uint64_t>(kpad), kpad * 2, kpad * 2, npad * 4};
  DayanDeviceBuffer cfg(runtime_, sizeof(config));
  CopyToDevice(cfg.Get(), &config, sizeof(config));
  DAYAN_ATTRS("matmul-bf16-1c");
  constexpr uint64_t kSmallMatmulUram = 768U * 1024U;
  constexpr uint64_t kLargeMatmulUram = 1536U * 1024U;
  const uint64_t uram = runtime_.uram_capacity();
  if (uram < kSmallMatmulUram)
    throw std::runtime_error("MatMul requires at least 768 KiB URAM per Core");
  const bool small_uram = uram < kLargeMatmulUram;
  const bool four_core = runtime_.supports_four_core();
  if (four_core) {
    attrs.config.task_dim = NPUTaskDim3{4U, 1U, 1U};
    attrs.config.task_mode = NPUTaskMode::kFourCore;
  }
  const auto launch = small_uram ? matmul_bf16_1c_small : matmul_bf16_1c;
  RequireSuccess(launch(cfg.Get(), ai.Get(), bi.Get(), out.Get(), &attrs),
                 "MatMul");
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
  auto page_bytes = [](uint64_t bytes) {
    return (bytes + 4095) & ~uint64_t(4095);
  };
  if (input_bytes > resident_limit || weight_bytes > resident_limit ||
      output_bytes > resident_limit ||
      page_bytes(input_bytes) + page_bytes(weight_bytes) +
          page_bytes(output_bytes) + 4096 > resident_limit)
    throw std::invalid_argument("Conv resident buffers exceed 256 MiB envelope");
  if (s.kernel_h > 7 || s.kernel_w > 7 || s.stride_h > 3 || s.stride_w > 3 ||
      s.dilation_h > 3 || s.dilation_w > 3)
    throw std::invalid_argument(
        "Conv outside tested device candidate envelope");
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

    Conv2dTileConfig config{static_cast<uint32_t>(s.input_channels),
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
                                  0, 0, 0, 0,
        1U};
    if (!conv2d_plan_tiles(&config))
      throw std::invalid_argument("Conv cannot plan tiles within 1.5 MiB URAM");
    const uint64_t tile_ih = (config.tile_h - 1ULL) * config.stride_h +
                            (config.hk - 1ULL) * config.dilation_h + 1;
    const uint64_t tile_iw = (config.tile_w - 1ULL) * config.stride_w +
                            (config.wk - 1ULL) * config.dilation_w + 1;
    auto aligned = [](uint64_t bytes) { return (bytes + 63) & ~uint64_t(63); };
    const uint64_t tile_bytes = 2 *
        (aligned(tile_ih * tile_iw * config.tile_ci * 2) +
         aligned(uint64_t(config.hk) * config.wk * config.tile_ci * config.tile_co * 2) +
         aligned(uint64_t(config.tile_h) * config.tile_w * config.tile_co * 4));
    const uint32_t reserved_uram = 1536U * 1024U;
    if (tile_bytes > reserved_uram)
      throw std::invalid_argument("Conv tile exceeds reserved URAM");
#if defined(DAYAN_CONV_PROFILE)
    std::printf("CONV_PROFILE_CONFIG tile=%ux%u ci=%u co=%u "
                "uram_used=%llu uram_reserved=%u\n",
                config.tile_h, config.tile_w, config.tile_ci, config.tile_co,
                (unsigned long long)tile_bytes, reserved_uram);
#endif
    DayanDeviceBuffer config_buffer(runtime_, sizeof(config));
    CopyToDevice(config_buffer.Get(), &config, sizeof(config));

#if defined(DAYAN_CONV_PROFILE)
    Conv2dProfileRecord records[4]{};
    DayanDeviceBuffer timing_buffer(runtime_, sizeof(records));
    static std::atomic<uint64_t> next_call_id{0};
#endif
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
      const auto launch_conv = conv2d_tile_4c;
      attrs.config.task_dim = NPUTaskDim3{4U, 1, 1};
      attrs.config.task_mode =
          NPUTaskMode::kFourCore;
      attrs.timeout_ms = kTimeout;
#if defined(DAYAN_CONV_PROFILE)
      // 超时后必须确认任务已终止，才能再次提交。
      attrs.timeout_ms = 45U * 60U * 1000U;
      const uint64_t call_id = ++next_call_id;
      std::memset(records, 0, sizeof(records));
      CopyToDevice(timing_buffer.Get(), records, sizeof(records));
      RequireSuccess(launch_conv(config_buffer.Get(), in.Get(),
                                   weights.Get(), out.Get(), timing_buffer.Get(),
                                   call_id, &attrs),
                     "Conv tiled 4c profiling");
      CopyFromDevice(records, timing_buffer.Get(), sizeof(records));
      for (uint64_t task = 0; task < 4; ++task) {
        const auto &r = records[task];
        if (r.completed != 1 || r.call_id != call_id || r.task != task ||
            r.end <= r.start)
          throw std::runtime_error("Conv kernel timing record incomplete");
        std::printf("CONV_KERNEL_TIME call=%llu batch=%d group=%d task=%llu "
                    "hart=%llu start=%llu end=%llu cycles=%llu completed=1\n",
                    (unsigned long long)call_id, batch, group,
                    (unsigned long long)task, (unsigned long long)r.hart,
                    (unsigned long long)r.start, (unsigned long long)r.end,
                    (unsigned long long)(r.end - r.start));
        static const char *phase_names[16] = {
            "setup", "input_cache", "weight_cache", "output_cache", "clear",
            "input_issue", "weight_load", "mat_config", "mat_wait", "output_store",
            "output_flush", "lsu_wait", "body", "tiles", "mat_calls", "weight_hits"};
        std::printf("CONV_PHASE_TIME call=%llu task=%llu",
                    (unsigned long long)call_id, (unsigned long long)task);
        for (unsigned phase = 0; phase < 16; ++phase)
          std::printf(" %s=%llu", phase_names[phase],
                      (unsigned long long)r.phase[phase]);
        std::printf("\n");
      }
#else
      RequireSuccess(launch_conv(config_buffer.Get(), in.Get(),
                                   weights.Get(), out.Get(), &attrs),
                     "Conv tiled 4c");
#endif
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

} // namespace dayan
