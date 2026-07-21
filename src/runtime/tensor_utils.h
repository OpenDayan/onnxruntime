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
  size_t boxes_elements{};
  size_t scores_elements{};
  size_t max_selected{};
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
  if (shape.empty() || shape.size() > 4) {
    throw std::invalid_argument(std::string(label) +
                                " requires tensor rank in [1, 4]");
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
  if (input.size() != 2 ||
      (attributes.axis != -1 && attributes.axis != 1) ||
      attributes.largest != 1 || attributes.sorted != 1) {
    throw std::invalid_argument(
        "RVNPU TopK requires rank-2, axis=-1, largest=1 and sorted=1");
  }
  TopKShape result;
  result.rows = CheckedDimension(input[0], "TopK rows");
  result.columns = CheckedDimension(input[1], "TopK columns");
  if (k <= 0 || k > result.columns || k > 300) {
    throw std::invalid_argument("RVNPU TopK requires K in [1, min(N, 300)]");
  }
  result.k = static_cast<int32_t>(k);
  result.output_shape = {result.rows, result.k};
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
      attributes.center_point_box != 0) {
    throw std::invalid_argument(
        "RVNPU NMS requires corner boxes [1,N,4] and scores [1,C,N]");
  }
  NmsShape result;
  result.num_boxes = CheckedDimension(boxes[1], "NMS num_boxes");
  result.num_classes = CheckedDimension(scores[1], "NMS num_classes");
  const int32_t score_boxes =
      CheckedDimension(scores[2], "NMS score num_boxes");
  if (score_boxes != result.num_boxes || max_output_per_class <= 0 ||
      max_output_per_class > 300) {
    throw std::invalid_argument(
        "RVNPU NMS requires matching boxes and max output in [1,300]");
  }
  result.max_output_per_class =
      static_cast<int32_t>(max_output_per_class);
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

} // namespace rvnpu
