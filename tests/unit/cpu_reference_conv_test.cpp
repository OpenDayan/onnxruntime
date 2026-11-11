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

#include "runtime/cpu_reference.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
  assert(dayan::Bf16ToFloat(dayan::FloatToBf16(1.00390625f)) == 1.0f);
  assert(dayan::Bf16ToFloat(dayan::FloatToBf16(1.01171875f)) == 1.015625f);
  assert(dayan::Bf16ToFloat(dayan::FloatToBf16(-1.00390625f)) == -1.0f);
  assert(dayan::Bf16ToFloat(dayan::FloatToBf16(-1.01171875f)) == -1.015625f);
  assert(dayan::FloatToBf16(0.f) == 0);
  assert(dayan::FloatToBf16(-0.f) == 0x8000);

  dayan::ConvAttributes attributes;
  attributes.kernel_shape = {3, 3};
  attributes.pads = {1, 1, 1, 1};
  attributes.strides = {1, 1};
  const std::vector<int64_t> bias_shape{1};
  const auto shape = dayan::ValidateConvShapes(
      {1, 1, 3, 3}, {1, 1, 3, 3}, &bias_shape, attributes);

  const std::vector<float> input(9, 1.00390625f);
  const std::vector<float> weight(9, 1.00390625f);
  const std::vector<float> bias{0.25f};
  std::vector<float> output;
  dayan::ConvBf16Reference(input, weight, &bias, output, shape);

  // Center output has nine BF16 products of 1.0 and a separate FP32 bias.
  assert(output.size() == 9);
  assert(std::abs(output[4] - 9.25f) < 1e-6f);
  return 0;
}
