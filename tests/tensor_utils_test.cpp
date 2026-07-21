#include "runtime/tensor_utils.h"

#include <array>
#include <cassert>
#include <climits>
#include <stdexcept>
#include <utility>
#include <vector>

int main() {
  const auto shape = rvnpu::ValidateMatMulShapes({1, 32}, {32, 32});
  assert(shape.m == 1 && shape.n == 32 && shape.k == 32);
  assert(shape.a_elements == 32 && shape.b_elements == 1024 &&
         shape.y_elements == 32);

  bool rejected = false;
  try {
    static_cast<void>(rvnpu::ValidateMatMulShapes({32}, {32, 32}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    static_cast<void>(rvnpu::ValidateMatMulShapes({1, 31}, {32, 32}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  for (const auto &shapes : {
           std::pair<std::vector<int64_t>, std::vector<int64_t>>{{1, 2, 3},
                                                                 {3, 4}},
           std::pair<std::vector<int64_t>, std::vector<int64_t>>{{0, 32},
                                                                 {32, 32}},
           std::pair<std::vector<int64_t>, std::vector<int64_t>>{{-1, 32},
                                                                 {32, 32}},
           std::pair<std::vector<int64_t>, std::vector<int64_t>>{
               {static_cast<int64_t>(INT32_MAX) + 1, 1}, {1, 1}},
           std::pair<std::vector<int64_t>, std::vector<int64_t>>{
               {INT32_MAX, INT32_MAX}, {INT32_MAX, INT32_MAX}},
       }) {
    rejected = false;
    try {
      static_cast<void>(
          rvnpu::ValidateMatMulShapes(shapes.first, shapes.second));
    } catch (const std::exception &) {
      rejected = true;
    }
    assert(rejected);
  }

  const auto unary = rvnpu::ValidateUnaryTensorShape({2, 3, 5}, "Sigmoid");
  assert(unary.elements == 30);
  assert((unary.dims == std::array<uint32_t, 4>{5, 3, 2, 1}));

  const auto same =
      rvnpu::ValidateBinaryElementwiseShapes({2, 3}, {2, 3}, "Add");
  assert(!same.swap_inputs && same.output_elements == 6);
  assert((same.output_shape == std::vector<int64_t>{2, 3}));
  assert((same.broadcast_dims == std::array<uint32_t, 4>{3, 2, 1, 1}));

  const auto broadcast = rvnpu::ValidateBinaryElementwiseShapes(
      {1, 3, 1, 1}, {1, 3, 4, 5}, "Mul");
  assert(broadcast.swap_inputs && broadcast.output_elements == 60);
  assert((broadcast.output_dims == std::array<uint32_t, 4>{5, 4, 3, 1}));
  assert((broadcast.broadcast_dims ==
          std::array<uint32_t, 4>{1, 1, 3, 1}));

  rejected = false;
  try {
    static_cast<void>(
        rvnpu::ValidateBinaryElementwiseShapes({2, 1}, {1, 3}, "Add"));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    static_cast<void>(rvnpu::ValidateUnaryTensorShape({}, "Sigmoid"));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rvnpu::ConvAttributes conv_attributes;
  conv_attributes.kernel_shape = {3, 3};
  conv_attributes.pads = {1, 1, 1, 1};
  conv_attributes.strides = {2, 2};
  const std::vector<int64_t> bias_shape{16};
  const auto conv = rvnpu::ValidateConvShapes(
      {1, 3, 64, 64}, {16, 3, 3, 3}, &bias_shape, conv_attributes);
  assert((conv.output_shape == std::vector<int64_t>{1, 16, 32, 32}));
  assert(conv.matmul_m == 1024 && conv.matmul_n == 16 &&
         conv.matmul_k == 27);
  assert(conv.column_elements == 1024 * 27 && conv.has_bias);

  rejected = false;
  try {
    auto unsupported = conv_attributes;
    unsupported.group = 3;
    static_cast<void>(rvnpu::ValidateConvShapes(
        {1, 3, 64, 64}, {16, 1, 3, 3}, nullptr, unsupported));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rvnpu::MaxPoolAttributes pool_attributes;
  pool_attributes.kernel_shape = {5, 5};
  pool_attributes.pads = {2, 2, 2, 2};
  const auto pool = rvnpu::ValidateMaxPoolShape(
      {1, 16, 32, 32}, pool_attributes);
  assert((pool.output_shape == std::vector<int64_t>{1, 16, 32, 32}));
  assert(pool.output_elements == 16 * 32 * 32);

  rejected = false;
  try {
    static_cast<void>(rvnpu::ValidateMaxPoolShape(
        {1, 16, 32, 32}, pool_attributes, 2));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
