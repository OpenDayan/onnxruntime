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

#include "runtime/tensor_utils.h"

#include <array>
#include <cassert>
#include <climits>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

int main() {
  const auto shape = dayan::ValidateMatMulShapes({1, 32}, {32, 32});
  assert(shape.m == 1 && shape.n == 32 && shape.k == 32);
  assert(shape.a_elements == 32 && shape.b_elements == 1024 &&
         shape.y_elements == 32);

  bool rejected = false;
  try {
    static_cast<void>(dayan::ValidateMatMulShapes({32}, {32, 32}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    static_cast<void>(dayan::ValidateMatMulShapes({1, 31}, {32, 32}));
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
       }) {
    rejected = false;
    try {
      static_cast<void>(
          dayan::ValidateMatMulShapes(shapes.first, shapes.second));
    } catch (const std::exception &) {
      rejected = true;
    }
    assert(rejected);
  }

  // These element/byte counts fit size_t on the supported 64-bit host.
  // Shape validation does not allocate these tensors.
  const auto large = dayan::ValidateMatMulShapes(
      {INT32_MAX, INT32_MAX}, {INT32_MAX, INT32_MAX});
  assert(large.a_elements == size_t(INT32_MAX) * INT32_MAX);

  const auto unary = dayan::ValidateUnaryTensorShape({2, 3, 5}, "Sigmoid");
  assert(unary.elements == 30);
  assert((unary.dims == std::array<uint32_t, 4>{5, 3, 2, 1}));

  const auto same =
      dayan::ValidateBinaryElementwiseShapes({2, 3}, {2, 3}, "Add");
  assert(!same.swap_inputs && same.output_elements == 6);
  assert((same.output_shape == std::vector<int64_t>{2, 3}));
  assert((same.broadcast_dims == std::array<uint32_t, 4>{3, 2, 1, 1}));

  const auto broadcast = dayan::ValidateBinaryElementwiseShapes(
      {1, 3, 1, 1}, {1, 3, 4, 5}, "Mul");
  assert(broadcast.swap_inputs && broadcast.output_elements == 60);
  assert((broadcast.output_dims == std::array<uint32_t, 4>{5, 4, 3, 1}));
  assert((broadcast.broadcast_dims ==
          std::array<uint32_t, 4>{1, 1, 3, 1}));

  const auto empty =
      dayan::ValidateBinaryElementwiseShapes({0}, {}, "Div");
  assert(empty.output_elements == 0);
  assert((empty.output_shape == std::vector<int64_t>{0}));

  const auto empty_concat =
      dayan::ValidateConcatShapes({{0, 2}, {0, 3}}, 1);
  assert(empty_concat.output_elements == 0 && empty_concat.outer == 0);
  assert((empty_concat.output_shape == std::vector<int64_t>{0, 5}));

  const auto empty_slice =
      dayan::ValidateSliceShape({0, 4}, 0, 2, 1, 1);
  assert(empty_slice.output_elements == 0 && empty_slice.outer == 0);
  assert((empty_slice.output_shape == std::vector<int64_t>{0, 2}));

  rejected = false;
  try {
    static_cast<void>(
        dayan::ValidateBinaryElementwiseShapes({2, 1}, {1, 3}, "Add"));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  // Scalars are supported by the current shape utility.
  const auto scalar = dayan::ValidateUnaryTensorShape({}, "Sigmoid");
  assert(scalar.elements == 1 && scalar.output_shape.empty());

  dayan::ConvAttributes conv_attributes;
  conv_attributes.kernel_shape = {3, 3};
  conv_attributes.pads = {1, 1, 1, 1};
  conv_attributes.strides = {2, 2};
  const std::vector<int64_t> bias_shape{16};
  const auto conv = dayan::ValidateConvShapes(
      {1, 3, 64, 64}, {16, 3, 3, 3}, &bias_shape, conv_attributes);
  assert((conv.output_shape == std::vector<int64_t>{1, 16, 32, 32}));
  assert(conv.output_elements == 1024 * 16);
  assert(conv.weight_elements == 16 * 27 && conv.has_bias);

  rejected = false;
  try {
    auto unsupported = conv_attributes;
    unsupported.group = 3;
    static_cast<void>(dayan::ValidateConvShapes(
        {1, 3, 64, 64}, {16, 1, 3, 3}, nullptr, unsupported));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  dayan::MaxPoolAttributes pool_attributes;
  pool_attributes.kernel_shape = {5, 5};
  pool_attributes.pads = {2, 2, 2, 2};
  const auto pool = dayan::ValidateMaxPoolShape(
      {1, 16, 32, 32}, pool_attributes);
  assert((pool.output_shape == std::vector<int64_t>{1, 16, 32, 32}));
  assert(pool.output_elements == 16 * 32 * 32);

  const auto transpose =
      dayan::ValidateTransposeShape({1, 4, 8400}, {0, 2, 1});
  assert(transpose.rank == 3 && transpose.dims[3] == 1);
  assert((transpose.output_shape == std::vector<int64_t>{1, 8400, 4}));

  const auto transpose_int64 =
      dayan::ValidateTransposeShape({5, 2}, {1, 0});
  assert(transpose_int64.rank == 2 && transpose_int64.elements == 10);
  assert((transpose_int64.output_shape == std::vector<int64_t>{2, 5}));

  const auto empty_transpose =
      dayan::ValidateTransposeShape({1, 0}, {1, 0});
  assert(empty_transpose.elements == 0);
  assert((empty_transpose.output_shape == std::vector<int64_t>{0, 1}));

  const auto reverse_slice = dayan::ValidateSliceShape(
      {5, 2}, -1, std::numeric_limits<int64_t>::min() + 1, 0, -1);
  assert(reverse_slice.start == 4 && reverse_slice.length == 5 &&
         reverse_slice.step == -1);
  assert((reverse_slice.output_shape == std::vector<int64_t>{5, 2}));

  const auto nms = dayan::ValidateNmsShapes(
      {1, 8400, 4}, {1, 80, 8400}, 300, {1});
  assert(nms.center_point_box == 1 && nms.num_boxes == 8400 &&
         nms.num_classes == 80);

  const auto empty_nms =
      dayan::ValidateNmsShapes({1, 0, 4}, {1, 1, 0}, 300, {});
  assert(empty_nms.num_boxes == 0 && empty_nms.boxes_elements == 0 &&
         empty_nms.scores_elements == 0);

  const auto topk = dayan::ValidateTopKShape({1, 8400, 80}, 1, {});
  assert(topk.rows == 8400 && topk.columns == 80 && topk.k == 1);
  assert((topk.output_shape == std::vector<int64_t>{1, 8400, 1}));

  dayan::ArgMaxAttributes argmax_attributes;
  argmax_attributes.axis = -1;
  argmax_attributes.keepdims = 0;
  const auto argmax =
      dayan::ValidateArgMaxShape({1, 8400, 80}, argmax_attributes);
  assert(argmax.rows == 8400 && argmax.columns == 80);
  assert((argmax.output_shape == std::vector<int64_t>{1, 8400}));

  const auto gather_nd =
      dayan::ValidateGatherNdShape({8400}, {64, 1}, 0, sizeof(int64_t));
  assert(gather_nd.rows == 64 && gather_nd.index_depth == 1 &&
         gather_nd.data_elements == 8400 && gather_nd.output_elements == 64);

  rejected = false;
  try {
    static_cast<void>(dayan::ValidateMaxPoolShape(
        {1, 16, 32, 32}, pool_attributes, 2));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
