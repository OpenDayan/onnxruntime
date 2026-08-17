#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rvnpu {

struct MatMulShape {
  int32_t m;
  int32_t n;
  int32_t k;
  size_t a_elements;
  size_t b_elements;
  size_t y_elements;
};

struct UnaryTensorShape {
  std::vector<int64_t> output_shape;
  std::array<uint32_t, 4> dims{};
  size_t elements{};
};

struct BinaryElementwiseShape {
  std::vector<int64_t> output_shape;
  std::array<uint32_t, 4> output_dims{};
  std::array<uint32_t, 4> broadcast_dims{};
  size_t lhs_elements{};
  size_t rhs_elements{};
  size_t output_elements{};
  bool swap_inputs{};
};

struct ConcatShape {
  std::vector<int64_t> output_shape;
  std::vector<int32_t> input_axes;
  int32_t outer{};
  int32_t output_axis{};
  int32_t inner{};
  size_t output_elements{};
};

struct SplitShape {
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int32_t> output_axes;
  int32_t outer{};
  int32_t input_axis{};
  int32_t inner{};
  size_t input_elements{};
};

struct SliceShape {
  std::vector<int64_t> output_shape;
  int32_t outer{};
  int32_t input_axis{};
  int32_t start{};
  int32_t length{};
  int32_t inner{};
  size_t input_elements{};
  size_t output_elements{};
};

struct ResizeShape {
  std::vector<int64_t> output_shape;
  int32_t batch{};
  int32_t channels{};
  int32_t input_h{};
  int32_t input_w{};
  int32_t output_h{};
  int32_t output_w{};
  size_t input_elements{};
  size_t output_elements{};
};

struct TransposeShape {
  std::vector<int64_t> output_shape;
  std::array<int32_t, 4> dims{};
  int32_t rank{};
  size_t elements{};
};

struct SoftmaxShape {
  std::vector<int64_t> output_shape;
  int32_t outer{};
  int32_t axis{};
  int32_t inner{};
  size_t elements{};
};

struct ReduceMaxShape {
  std::vector<int64_t> output_shape;
  int32_t rows{};
  int32_t columns{};
  size_t input_elements{};
  size_t output_elements{};
};

struct ConvAttributes {
  std::vector<int64_t> dilations{1, 1};
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads{0, 0, 0, 0};
  std::vector<int64_t> strides{1, 1};
  int64_t group{1};
  std::string auto_pad{"NOTSET"};
};

struct ConvShape {
  std::vector<int64_t> output_shape;
  int32_t input_channels{};
  int32_t input_h{};
  int32_t input_w{};
  int32_t output_channels{};
  int32_t output_h{};
  int32_t output_w{};
  int32_t kernel_h{};
  int32_t kernel_w{};
  int32_t stride_h{};
  int32_t stride_w{};
  int32_t pad_h{};
  int32_t pad_w{};
  int32_t matmul_m{};
  int32_t matmul_n{};
  int32_t matmul_k{};
  size_t input_elements{};
  size_t weight_elements{};
  size_t bias_elements{};
  size_t column_elements{};
  size_t output_elements{};
  bool has_bias{};
};

struct MaxPoolAttributes {
  std::vector<int64_t> dilations{1, 1};
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads{0, 0, 0, 0};
  std::vector<int64_t> strides{1, 1};
  int64_t ceil_mode{0};
  int64_t storage_order{0};
  std::string auto_pad{"NOTSET"};
};

struct MaxPoolShape {
  std::vector<int64_t> output_shape;
  int32_t batch{};
  int32_t channels{};
  int32_t input_h{};
  int32_t input_w{};
  int32_t output_h{};
  int32_t output_w{};
  int32_t kernel_h{};
  int32_t kernel_w{};
  int32_t stride_h{};
  int32_t stride_w{};
  int32_t pad_h{};
  int32_t pad_w{};
  size_t input_elements{};
  size_t output_elements{};
};

struct TopKAttributes {
  int64_t axis{-1};
  int64_t largest{1};
  int64_t sorted{1};
};

struct TopKShape {
  std::vector<int64_t> output_shape;
  int32_t rows{};
  int32_t columns{};
  int32_t k{};
  size_t input_elements{};
  size_t output_elements{};
};

struct NmsAttributes {
  int64_t center_point_box{0};
};

struct NmsShape {
  int32_t num_boxes{};
  int32_t num_classes{};
  int32_t max_output_per_class{};
  int32_t center_point_box{};
  size_t boxes_elements{};
  size_t scores_elements{};
  size_t max_selected{};
};

struct GatherNdShape {
  std::vector<int64_t> output_shape;
  int32_t rows{};
  int32_t index_depth{};
  std::array<int32_t, 3> data_dims{};
  size_t data_elements{};
  size_t indices_elements{};
  size_t output_elements{};
};

inline size_t CheckedMultiply(size_t a, size_t b, const char *label) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    throw std::overflow_error(std::string(label) + " size overflow");
  }
  return a * b;
}

inline int32_t CheckedDimension(int64_t value, const char *label) {
  if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument(std::string(label) +
                                " must be in [1, INT32_MAX]");
  }
  return static_cast<int32_t>(value);
}

inline int32_t CheckedSizeDimension(size_t value, const char *label) {
  if (value == 0 ||
      value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument(std::string(label) +
                                " must be in [1, INT32_MAX]");
  }
  return static_cast<int32_t>(value);
}

inline size_t ShapeProduct(const std::vector<int64_t> &shape, size_t begin,
                           size_t end, const char *label) {
  size_t result = 1;
  for (size_t index = begin; index < end; ++index) {
    result = CheckedMultiply(result,
                             static_cast<size_t>(CheckedDimension(shape[index],
                                                                  label)),
                             label);
  }
  return result;
}

inline size_t NormalizeAxis(int64_t axis, size_t rank, const char *label) {
  if (rank == 0)
    throw std::invalid_argument(std::string(label) + " requires rank >= 1");
  const int64_t normalized = axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
  if (normalized < 0 || normalized >= static_cast<int64_t>(rank))
    throw std::invalid_argument(std::string(label) + " axis is invalid");
  return static_cast<size_t>(normalized);
}

inline ConcatShape
ValidateConcatShapes(const std::vector<std::vector<int64_t>> &inputs,
                     int64_t axis) {
  if (inputs.size() < 2 || inputs[0].empty() || inputs[0].size() > 4)
    throw std::invalid_argument("RVNPU Concat requires 2 or more rank-1..4 inputs");
  const size_t rank = inputs[0].size();
  const size_t normalized_axis = NormalizeAxis(axis, rank, "Concat");
  ConcatShape result;
  result.output_shape = inputs[0];
  result.output_shape[normalized_axis] = 0;
  for (const auto &input : inputs) {
    if (input.size() != rank)
      throw std::invalid_argument("RVNPU Concat input ranks do not match");
    for (size_t dimension = 0; dimension < rank; ++dimension) {
      CheckedDimension(input[dimension], "Concat");
      if (dimension != normalized_axis &&
          input[dimension] != inputs[0][dimension]) {
        throw std::invalid_argument("RVNPU Concat non-axis dimensions differ");
      }
    }
    result.input_axes.push_back(
        CheckedDimension(input[normalized_axis], "Concat axis"));
    result.output_shape[normalized_axis] += input[normalized_axis];
  }
  result.outer = CheckedSizeDimension(
      ShapeProduct(inputs[0], 0, normalized_axis, "Concat outer"),
      "Concat outer");
  result.inner = CheckedSizeDimension(
      ShapeProduct(inputs[0], normalized_axis + 1, rank, "Concat inner"),
      "Concat inner");
  result.output_axis =
      CheckedDimension(result.output_shape[normalized_axis], "Concat axis");
  result.output_elements = ShapeProduct(result.output_shape, 0, rank,
                                        "Concat output");
  return result;
}

inline SplitShape ValidateSplitShape(const std::vector<int64_t> &input,
                                     const std::vector<int64_t> &splits,
                                     int64_t axis, size_t output_count) {
  if (input.empty() || input.size() > 4 || output_count < 2 ||
      splits.size() != output_count) {
    throw std::invalid_argument("RVNPU Split requires rank-1..4 and explicit splits");
  }
  const size_t normalized_axis = NormalizeAxis(axis, input.size(), "Split");
  const int32_t input_axis = CheckedDimension(input[normalized_axis], "Split axis");
  int64_t total = 0;
  SplitShape result;
  for (const int64_t split : splits) {
    const int32_t part = CheckedDimension(split, "Split part");
    total += part;
    result.output_axes.push_back(part);
    auto output = input;
    output[normalized_axis] = part;
    result.output_shapes.push_back(std::move(output));
  }
  if (total != input_axis)
    throw std::invalid_argument("RVNPU Split sizes do not cover the axis");
  result.outer = CheckedSizeDimension(
      ShapeProduct(input, 0, normalized_axis, "Split outer"), "Split outer");
  result.inner = CheckedSizeDimension(
      ShapeProduct(input, normalized_axis + 1, input.size(), "Split inner"),
      "Split inner");
  result.input_axis = input_axis;
  result.input_elements = ShapeProduct(input, 0, input.size(), "Split input");
  return result;
}

inline SliceShape ValidateSliceShape(const std::vector<int64_t> &input,
                                     int64_t start, int64_t end, int64_t axis,
                                     int64_t step) {
  if (input.empty() || input.size() > 4 || step != 1)
    throw std::invalid_argument("RVNPU Slice requires rank-1..4 and step=1");
  const size_t normalized_axis = NormalizeAxis(axis, input.size(), "Slice");
  const int64_t dimension = CheckedDimension(input[normalized_axis], "Slice axis");
  int64_t normalized_start = start < 0 ? std::max<int64_t>(0, start + dimension)
                                       : std::min(start, dimension);
  int64_t normalized_end = end < 0 ? std::max<int64_t>(0, end + dimension)
                                   : std::min(end, dimension);
  if (normalized_end <= normalized_start)
    throw std::invalid_argument("RVNPU Slice requires a non-empty range");
  SliceShape result;
  result.output_shape = input;
  result.start = static_cast<int32_t>(normalized_start);
  result.length = static_cast<int32_t>(normalized_end - normalized_start);
  result.output_shape[normalized_axis] = result.length;
  result.outer = CheckedSizeDimension(
      ShapeProduct(input, 0, normalized_axis, "Slice outer"), "Slice outer");
  result.inner = CheckedSizeDimension(
      ShapeProduct(input, normalized_axis + 1, input.size(), "Slice inner"),
      "Slice inner");
  result.input_axis = static_cast<int32_t>(dimension);
  result.input_elements = ShapeProduct(input, 0, input.size(), "Slice input");
  result.output_elements =
      ShapeProduct(result.output_shape, 0, input.size(), "Slice output");
  return result;
}

inline ResizeShape
ValidateResizeShape(const std::vector<int64_t> &input,
                    const std::vector<int64_t> &output) {
  if (input.size() != 4 || output.size() != 4 || input[0] != output[0] ||
      input[1] != output[1])
    throw std::invalid_argument("RVNPU Resize requires NCHW with unchanged N/C");
  ResizeShape result;
  result.output_shape = output;
  result.batch = CheckedDimension(input[0], "Resize N");
  result.channels = CheckedDimension(input[1], "Resize C");
  result.input_h = CheckedDimension(input[2], "Resize H");
  result.input_w = CheckedDimension(input[3], "Resize W");
  result.output_h = CheckedDimension(output[2], "Resize output H");
  result.output_w = CheckedDimension(output[3], "Resize output W");
  if (result.output_h < result.input_h || result.output_w < result.input_w ||
      result.output_h % result.input_h != 0 ||
      result.output_w % result.input_w != 0)
    throw std::invalid_argument("RVNPU Resize requires integer upscaling");
  result.input_elements = ShapeProduct(input, 0, input.size(), "Resize input");
  result.output_elements =
      ShapeProduct(output, 0, output.size(), "Resize output");
  return result;
}

inline TransposeShape ValidateTransposeShape(
    const std::vector<int64_t> &input, const std::vector<int64_t> &perm) {
  const bool rank3 =
      input.size() == 3 && perm == std::vector<int64_t>({0, 2, 1}) &&
      input[1] > 0 && input[1] <= 128 && input[2] > 0 &&
      input[2] % 16 == 0;
  const bool rank4 =
      input.size() == 4 && perm == std::vector<int64_t>({0, 2, 1, 3});
  if (!rank3 && !rank4)
    throw std::invalid_argument(
        "RVNPU Transpose requires rank-3 [N,C,K], C<=128, K divisible by 16 and perm [0,2,1], or rank-4 perm [0,2,1,3]");
  TransposeShape result;
  result.rank = static_cast<int32_t>(input.size());
  for (size_t index = 0; index < input.size(); ++index)
    result.dims[index] = CheckedDimension(input[index], "Transpose");
  if (rank3) {
    result.dims[3] = 1;
    result.output_shape = {input[0], input[2], input[1]};
  } else {
    result.output_shape = {input[0], input[2], input[1], input[3]};
  }
  result.elements = ShapeProduct(input, 0, input.size(), "Transpose");
  return result;
}

inline ReduceMaxShape ValidateReduceMaxShape(
    const std::vector<int64_t> &input, const std::vector<int64_t> &axes,
    int64_t keepdims) {
  if (input.empty() || input.size() > 4 || axes.size() != 1 ||
      NormalizeAxis(axes[0], input.size(), "ReduceMax") + 1 != input.size() ||
      (keepdims != 0 && keepdims != 1)) {
    throw std::invalid_argument(
        "RVNPU ReduceMax requires static rank 1-4, one last axis and keepdims 0 or 1");
  }
  if (std::any_of(input.begin(), input.end(),
                  [](int64_t dimension) { return dimension <= 0; })) {
    throw std::invalid_argument("RVNPU ReduceMax requires static dimensions");
  }
  ReduceMaxShape result;
  result.columns = CheckedDimension(input.back(), "ReduceMax columns");
  result.rows = CheckedSizeDimension(
      ShapeProduct(input, 0, input.size() - 1, "ReduceMax rows"),
      "ReduceMax rows");
  result.output_shape = input;
  if (keepdims == 0)
    result.output_shape.pop_back();
  else
    result.output_shape.back() = 1;
  result.input_elements = ShapeProduct(input, 0, input.size(), "ReduceMax");
  result.output_elements = static_cast<size_t>(result.rows);
  return result;
}

inline SoftmaxShape ValidateSoftmaxAxis1Shape(
    const std::vector<int64_t> &input, int64_t axis) {
  if (input.size() != 4 || NormalizeAxis(axis, input.size(), "Softmax") != 1 ||
      input[1] != 16)
    throw std::invalid_argument(
        "RVNPU Softmax requires rank-4, axis=1 and dimension 16");
  SoftmaxShape result;
  result.output_shape = input;
  result.outer = CheckedDimension(input[0], "Softmax outer");
  result.axis = CheckedDimension(input[1], "Softmax axis");
  result.inner = CheckedSizeDimension(
      ShapeProduct(input, 2, input.size(), "Softmax inner"),
      "Softmax inner");
  result.elements = ShapeProduct(input, 0, input.size(), "Softmax");
  return result;
}

inline MatMulShape ValidateMatMulShapes(const std::vector<int64_t> &a,
                                        const std::vector<int64_t> &b) {
  if (a.size() != 2 || b.size() != 2) {
    throw std::invalid_argument("RVNPU MatMul requires rank-2 inputs");
  }
  const int32_t m = CheckedDimension(a[0], "M");
  const int32_t k = CheckedDimension(a[1], "K");
  const int32_t b_k = CheckedDimension(b[0], "B.K");
  const int32_t n = CheckedDimension(b[1], "N");
  if (k != b_k) {
    throw std::invalid_argument("RVNPU MatMul K dimensions do not match");
  }
  const size_t a_elements =
      CheckedMultiply(static_cast<size_t>(m), static_cast<size_t>(k), "A");
  const size_t b_elements =
      CheckedMultiply(static_cast<size_t>(k), static_cast<size_t>(n), "B");
  const size_t y_elements =
      CheckedMultiply(static_cast<size_t>(m), static_cast<size_t>(n), "Y");
  CheckedMultiply(a_elements, sizeof(float), "A bytes");
  CheckedMultiply(b_elements, sizeof(float), "B bytes");
  CheckedMultiply(y_elements, sizeof(float), "Y bytes");
  return {m, n, k, a_elements, b_elements, y_elements};
}

inline UnaryTensorShape ValidateUnaryTensorShape(
    const std::vector<int64_t> &shape, const char *label) {
  if (shape.size() > 4) {
    throw std::invalid_argument(std::string(label) +
                                " requires tensor rank in [0, 4]");
  }
  UnaryTensorShape result;
  result.output_shape = shape;
  result.dims.fill(1);
  result.elements = 1;
  for (size_t index = 0; index < shape.size(); ++index) {
    const int32_t dimension = CheckedDimension(shape[index], label);
    result.elements =
        CheckedMultiply(result.elements, static_cast<size_t>(dimension), label);
    result.dims[shape.size() - 1 - index] = static_cast<uint32_t>(dimension);
  }
  CheckedMultiply(result.elements, sizeof(float), label);
  return result;
}

inline BinaryElementwiseShape ValidateBinaryElementwiseShapes(
    const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs,
    const char *label) {
  const auto lhs_info = ValidateUnaryTensorShape(lhs, label);
  const auto rhs_info = ValidateUnaryTensorShape(rhs, label);
  const size_t rank = std::max(lhs.size(), rhs.size());
  std::vector<int64_t> output(rank, 1);
  for (size_t offset = 0; offset < rank; ++offset) {
    const int64_t lhs_dim =
        offset < lhs.size() ? lhs[lhs.size() - 1 - offset] : 1;
    const int64_t rhs_dim =
        offset < rhs.size() ? rhs[rhs.size() - 1 - offset] : 1;
    if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
      throw std::invalid_argument(std::string(label) +
                                  " inputs are not broadcast-compatible");
    }
    output[rank - 1 - offset] = std::max(lhs_dim, rhs_dim);
  }

  const bool lhs_is_output = lhs == output;
  const bool rhs_is_output = rhs == output;
  if (!lhs_is_output && !rhs_is_output) {
    throw std::invalid_argument(
        std::string(label) +
        " requires one input shape to equal the broadcast output shape");
  }

  const auto output_info = ValidateUnaryTensorShape(output, label);
  BinaryElementwiseShape result;
  result.output_shape = std::move(output);
  result.output_dims = output_info.dims;
  result.lhs_elements = lhs_info.elements;
  result.rhs_elements = rhs_info.elements;
  result.output_elements = output_info.elements;
  result.swap_inputs = !lhs_is_output && rhs_is_output;
  result.broadcast_dims =
      result.swap_inputs ? lhs_info.dims : rhs_info.dims;
  return result;
}

inline void ValidateSpatialVector(const std::vector<int64_t> &values,
                                  size_t expected, const char *label) {
  if (values.size() != expected ||
      std::any_of(values.begin(), values.end(),
                  [](int64_t value) {
                    return value < 0 ||
                           value > std::numeric_limits<int32_t>::max();
                  })) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
}

inline int32_t SpatialOutputDimension(int32_t input, int32_t kernel,
                                      int32_t stride, int32_t pad_before,
                                      int32_t pad_after,
                                      const char *label) {
  const int64_t numerator = static_cast<int64_t>(input) + pad_before +
                            pad_after - kernel;
  if (numerator < 0) {
    throw std::invalid_argument(std::string(label) +
                                " has no valid output positions");
  }
  return CheckedDimension(numerator / stride + 1, label);
}

inline ConvShape ValidateConvShapes(
    const std::vector<int64_t> &input,
    const std::vector<int64_t> &weight,
    const std::vector<int64_t> *bias,
    const ConvAttributes &attributes) {
  if (input.size() != 4 || weight.size() != 4) {
    throw std::invalid_argument("RVNPU Conv requires rank-4 NCHW tensors");
  }
  const int32_t batch = CheckedDimension(input[0], "Conv batch");
  const int32_t input_channels =
      CheckedDimension(input[1], "Conv input channels");
  const int32_t input_h = CheckedDimension(input[2], "Conv input height");
  const int32_t input_w = CheckedDimension(input[3], "Conv input width");
  const int32_t output_channels =
      CheckedDimension(weight[0], "Conv output channels");
  const int32_t weight_channels =
      CheckedDimension(weight[1], "Conv weight channels");
  const int32_t kernel_h = CheckedDimension(weight[2], "Conv kernel height");
  const int32_t kernel_w = CheckedDimension(weight[3], "Conv kernel width");

  if (batch != 1 || attributes.group != 1 ||
      attributes.auto_pad != "NOTSET" || input_channels != weight_channels) {
    throw std::invalid_argument(
        "RVNPU Conv requires batch=1, group=1 and auto_pad=NOTSET");
  }
  ValidateSpatialVector(attributes.dilations, 2, "Conv dilations");
  ValidateSpatialVector(attributes.strides, 2, "Conv strides");
  ValidateSpatialVector(attributes.pads, 4, "Conv pads");
  if (attributes.dilations != std::vector<int64_t>{1, 1} ||
      (kernel_h != 1 && kernel_h != 3) || kernel_w != kernel_h ||
      (attributes.strides[0] != 1 && attributes.strides[0] != 2) ||
      (attributes.strides[1] != 1 && attributes.strides[1] != 2) ||
      attributes.pads[0] != attributes.pads[2] ||
      attributes.pads[1] != attributes.pads[3]) {
    throw std::invalid_argument("RVNPU Conv attributes are unsupported");
  }
  if (!attributes.kernel_shape.empty() &&
      attributes.kernel_shape !=
          std::vector<int64_t>{kernel_h, kernel_w}) {
    throw std::invalid_argument("RVNPU Conv kernel_shape does not match W");
  }

  const int32_t stride_h =
      CheckedDimension(attributes.strides[0], "Conv stride height");
  const int32_t stride_w =
      CheckedDimension(attributes.strides[1], "Conv stride width");
  const int32_t pad_h = static_cast<int32_t>(attributes.pads[0]);
  const int32_t pad_w = static_cast<int32_t>(attributes.pads[1]);
  const int32_t output_h = SpatialOutputDimension(
      input_h, kernel_h, stride_h, pad_h, pad_h, "Conv output height");
  const int32_t output_w = SpatialOutputDimension(
      input_w, kernel_w, stride_w, pad_w, pad_w, "Conv output width");

  if (bias != nullptr &&
      (bias->size() != 1 || (*bias)[0] != output_channels)) {
    throw std::invalid_argument("RVNPU Conv bias must have shape [M]");
  }

  ConvShape result;
  result.output_shape = {1, output_channels, output_h, output_w};
  result.input_channels = input_channels;
  result.input_h = input_h;
  result.input_w = input_w;
  result.output_channels = output_channels;
  result.output_h = output_h;
  result.output_w = output_w;
  result.kernel_h = kernel_h;
  result.kernel_w = kernel_w;
  result.stride_h = stride_h;
  result.stride_w = stride_w;
  result.pad_h = pad_h;
  result.pad_w = pad_w;
  result.matmul_m = CheckedSizeDimension(
      CheckedMultiply(static_cast<size_t>(output_h), output_w,
                      "Conv MatMul M"),
      "Conv MatMul M");
  result.matmul_n = output_channels;
  result.matmul_k = CheckedSizeDimension(
      CheckedMultiply(
          CheckedMultiply(static_cast<size_t>(input_channels), kernel_h,
                          "Conv MatMul K"),
          kernel_w, "Conv MatMul K"),
      "Conv MatMul K");
  result.input_elements = CheckedMultiply(
      CheckedMultiply(static_cast<size_t>(input_channels), input_h,
                      "Conv input"),
      input_w, "Conv input");
  result.weight_elements = CheckedMultiply(
      static_cast<size_t>(output_channels), result.matmul_k, "Conv weight");
  result.bias_elements = bias == nullptr ? 0 : output_channels;
  result.column_elements = CheckedMultiply(
      static_cast<size_t>(result.matmul_m), result.matmul_k, "Conv im2col");
  result.output_elements = CheckedMultiply(
      static_cast<size_t>(result.matmul_m), result.matmul_n, "Conv output");
  result.has_bias = bias != nullptr;
  CheckedMultiply(result.input_elements, sizeof(float), "Conv input bytes");
  CheckedMultiply(result.weight_elements, sizeof(float), "Conv weight bytes");
  CheckedMultiply(result.column_elements, sizeof(float), "Conv im2col bytes");
  CheckedMultiply(result.output_elements, sizeof(float), "Conv output bytes");
  return result;
}

inline MaxPoolShape ValidateMaxPoolShape(
    const std::vector<int64_t> &input,
    const MaxPoolAttributes &attributes, size_t output_count = 1) {
  if (input.size() != 4) {
    throw std::invalid_argument("RVNPU MaxPool requires rank-4 NCHW input");
  }
  if (output_count != 1 || attributes.auto_pad != "NOTSET" ||
      attributes.ceil_mode != 0 || attributes.storage_order != 0) {
    throw std::invalid_argument("RVNPU MaxPool configuration is unsupported");
  }
  ValidateSpatialVector(attributes.kernel_shape, 2, "MaxPool kernel_shape");
  ValidateSpatialVector(attributes.dilations, 2, "MaxPool dilations");
  ValidateSpatialVector(attributes.strides, 2, "MaxPool strides");
  ValidateSpatialVector(attributes.pads, 4, "MaxPool pads");
  const int32_t kernel_h =
      CheckedDimension(attributes.kernel_shape[0], "MaxPool kernel height");
  const int32_t kernel_w =
      CheckedDimension(attributes.kernel_shape[1], "MaxPool kernel width");
  if ((kernel_h != 2 && kernel_h != 3 && kernel_h != 5) ||
      kernel_w != kernel_h ||
      attributes.dilations != std::vector<int64_t>{1, 1} ||
      (attributes.strides[0] != 1 && attributes.strides[0] != 2) ||
      (attributes.strides[1] != 1 && attributes.strides[1] != 2) ||
      attributes.pads[0] != attributes.pads[2] ||
      attributes.pads[1] != attributes.pads[3]) {
    throw std::invalid_argument("RVNPU MaxPool attributes are unsupported");
  }

  MaxPoolShape result;
  result.batch = CheckedDimension(input[0], "MaxPool batch");
  result.channels = CheckedDimension(input[1], "MaxPool channels");
  result.input_h = CheckedDimension(input[2], "MaxPool input height");
  result.input_w = CheckedDimension(input[3], "MaxPool input width");
  result.kernel_h = kernel_h;
  result.kernel_w = kernel_w;
  result.stride_h =
      CheckedDimension(attributes.strides[0], "MaxPool stride height");
  result.stride_w =
      CheckedDimension(attributes.strides[1], "MaxPool stride width");
  result.pad_h = static_cast<int32_t>(attributes.pads[0]);
  result.pad_w = static_cast<int32_t>(attributes.pads[1]);
  result.output_h = SpatialOutputDimension(
      result.input_h, kernel_h, result.stride_h, result.pad_h,
      result.pad_h, "MaxPool output height");
  result.output_w = SpatialOutputDimension(
      result.input_w, kernel_w, result.stride_w, result.pad_w,
      result.pad_w, "MaxPool output width");
  result.output_shape = {result.batch, result.channels, result.output_h,
                         result.output_w};
  result.input_elements = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(static_cast<size_t>(result.batch), result.channels,
                          "MaxPool input"),
          result.input_h, "MaxPool input"),
      result.input_w, "MaxPool input");
  result.output_elements = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(static_cast<size_t>(result.batch), result.channels,
                          "MaxPool output"),
          result.output_h, "MaxPool output"),
      result.output_w, "MaxPool output");
  CheckedMultiply(result.input_elements, sizeof(float),
                  "MaxPool input bytes");
  CheckedMultiply(result.output_elements, sizeof(float),
                  "MaxPool output bytes");
  return result;
}

inline TopKShape ValidateTopKShape(const std::vector<int64_t> &input,
                                   int64_t k,
                                   const TopKAttributes &attributes) {
  if (input.empty() || input.size() > 4 ||
      NormalizeAxis(attributes.axis, input.size(), "TopK") + 1 !=
          input.size() ||
      attributes.largest != 1 || attributes.sorted != 1) {
    throw std::invalid_argument(
        "RVNPU TopK requires static rank 1-4, last axis, largest=1 and sorted=1");
  }
  TopKShape result;
  result.rows = CheckedSizeDimension(
      ShapeProduct(input, 0, input.size() - 1, "TopK rows"), "TopK rows");
  result.columns = CheckedDimension(input.back(), "TopK columns");
  if (k <= 0 || k > result.columns || k > 300) {
    throw std::invalid_argument("RVNPU TopK requires K in [1, min(N, 300)]");
  }
  result.k = static_cast<int32_t>(k);
  result.output_shape = input;
  result.output_shape.back() = result.k;
  result.input_elements = CheckedMultiply(
      static_cast<size_t>(result.rows), result.columns, "TopK input");
  result.output_elements = CheckedMultiply(
      static_cast<size_t>(result.rows), result.k, "TopK output");
  CheckedMultiply(result.input_elements, sizeof(float), "TopK input bytes");
  CheckedMultiply(result.output_elements, sizeof(float),
                  "TopK values bytes");
  CheckedMultiply(result.output_elements, sizeof(int64_t),
                  "TopK indices bytes");
  return result;
}

inline NmsShape ValidateNmsShapes(const std::vector<int64_t> &boxes,
                                  const std::vector<int64_t> &scores,
                                  int64_t max_output_per_class,
                                  const NmsAttributes &attributes) {
  if (boxes.size() != 3 || scores.size() != 3 || boxes[0] != 1 ||
      scores[0] != 1 || boxes[2] != 4 ||
      (attributes.center_point_box != 0 &&
       attributes.center_point_box != 1)) {
    throw std::invalid_argument(
        "RVNPU NMS requires boxes [1,N,4], scores [1,C,N] and center_point_box 0 or 1");
  }
  NmsShape result;
  result.center_point_box = static_cast<int32_t>(attributes.center_point_box);
  result.num_boxes = CheckedDimension(boxes[1], "NMS num_boxes");
  result.num_classes = CheckedDimension(scores[1], "NMS num_classes");
  const int32_t score_boxes =
      CheckedDimension(scores[2], "NMS score num_boxes");
  if (score_boxes != result.num_boxes || max_output_per_class <= 0) {
    throw std::invalid_argument(
        "RVNPU NMS requires matching boxes and a positive max output");
  }
  result.max_output_per_class = static_cast<int32_t>(
      std::min<int64_t>(max_output_per_class, result.num_boxes));
  result.boxes_elements =
      CheckedMultiply(static_cast<size_t>(result.num_boxes), 4, "NMS boxes");
  result.scores_elements = CheckedMultiply(
      static_cast<size_t>(result.num_classes), result.num_boxes, "NMS scores");
  result.max_selected = CheckedMultiply(
      static_cast<size_t>(result.num_classes), result.max_output_per_class,
      "NMS selected");
  CheckedMultiply(result.boxes_elements, sizeof(float), "NMS boxes bytes");
  CheckedMultiply(result.scores_elements, sizeof(float), "NMS scores bytes");
  CheckedMultiply(CheckedMultiply(result.max_selected, 3, "NMS output"),
                  sizeof(int64_t), "NMS output bytes");
  return result;
}

inline GatherNdShape
ValidateGatherNdShape(const std::vector<int64_t> &data,
                      const std::vector<int64_t> &indices,
                      int64_t batch_dims) {
  if (batch_dims != 0 || data.empty() || data.size() > 3 ||
      indices.size() != 2 || indices[1] <= 0 ||
      indices[1] > static_cast<int64_t>(data.size()) || indices[0] < 0) {
    throw std::invalid_argument(
        "RVNPU GatherND requires FP32 rank 1-3 data, INT64 [K,D] indices and batch_dims=0");
  }

  GatherNdShape result;
  if (indices[0] > std::numeric_limits<int32_t>::max())
    throw std::invalid_argument("GatherND row count is too large");
  result.rows = static_cast<int32_t>(indices[0]);
  result.index_depth = static_cast<int32_t>(indices[1]);
  result.data_dims = {1, 1, 1};
  for (size_t dimension = 0; dimension < data.size(); ++dimension)
    result.data_dims[dimension] =
        CheckedDimension(data[dimension], "GatherND data");

  result.output_shape = {indices[0]};
  size_t output_width = 1;
  for (size_t dimension = static_cast<size_t>(result.index_depth);
       dimension < data.size(); ++dimension) {
    result.output_shape.push_back(data[dimension]);
    output_width = CheckedMultiply(output_width,
                                   static_cast<size_t>(data[dimension]),
                                   "GatherND output width");
  }
  result.data_elements = ShapeProduct(data, 0, data.size(), "GatherND data");
  result.indices_elements = CheckedMultiply(
      static_cast<size_t>(result.rows),
      static_cast<size_t>(result.index_depth), "GatherND indices");
  result.output_elements = CheckedMultiply(static_cast<size_t>(result.rows),
                                           output_width, "GatherND output");
  CheckedMultiply(result.data_elements, sizeof(float), "GatherND data bytes");
  CheckedMultiply(result.indices_elements, sizeof(int64_t),
                  "GatherND indices bytes");
  CheckedMultiply(result.output_elements, sizeof(float),
                  "GatherND output bytes");
  return result;
}

} // namespace rvnpu
