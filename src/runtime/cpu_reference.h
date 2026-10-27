#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "runtime/tensor_utils.h"

namespace rvnpu {

// Test-only numerical reference. The ORT CPU EP still executes its normal
// FP32 Conv; this helper is used to compare the NPU's reduced-precision path.
// Normal finite values use round-to-nearest, ties-to-even, matching host
// packing. Special values/subnormals retain the legacy conversion below;
// this helper is not a claim that those cases have passed RTL qualification.
inline uint16_t FloatToBf16(float value) noexcept {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U)
    return static_cast<uint16_t>((bits >> 16) | 0x007fU);
  if ((bits & 0x7f800000U) == 0)
    return static_cast<uint16_t>(bits >> 16);
  uint32_t truncated = bits >> 16;
  const uint32_t round_bit = (bits >> 15) & 1U;
  const uint32_t sticky = bits & 0x7fffU;
  if (round_bit != 0 && (sticky != 0 || (truncated & 1U) != 0))
    ++truncated;
  if ((truncated & 0x7f80U) == 0x7f80U)
    truncated = (truncated & 0x8000U) | 0x7f80U;
  return static_cast<uint16_t>(truncated);
}

inline float Bf16ToFloat(uint16_t value) noexcept {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  static_assert(sizeof(bits) == sizeof(result));
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline void ConvReference(
    const std::vector<float> &input, const std::vector<float> &weight,
    const std::vector<float> *bias, std::vector<float> &output,
    const ConvShape &shape, bool bf16) {
  if (input.size() != shape.input_elements) {
    throw std::invalid_argument("BF16 Conv input size does not match shape");
  }
  if (weight.size() != shape.weight_elements) {
    throw std::invalid_argument("BF16 Conv weight size does not match shape");
  }
  if (bias != nullptr && bias->size() != shape.bias_elements) {
    throw std::invalid_argument("BF16 Conv bias size does not match shape");
  }

  std::vector<uint16_t> input_bf16(input.size());
  std::vector<uint16_t> weight_bf16(weight.size());
  for (size_t index = 0; index < input.size(); ++index) {
    input_bf16[index] = FloatToBf16(input[index]);
  }
  for (size_t index = 0; index < weight.size(); ++index) {
    weight_bf16[index] = FloatToBf16(weight[index]);
  }

  output.assign(shape.output_elements, 0.0f);
  for (int32_t batch = 0; batch < shape.batch; ++batch)
  for (int32_t output_channel = 0;
       output_channel < shape.output_channels; ++output_channel) {
    for (int32_t output_h = 0; output_h < shape.output_h; ++output_h) {
      for (int32_t output_w = 0; output_w < shape.output_w; ++output_w) {
        // Keep the same straightforward accumulation structure as the
        // existing FP32 Conv reference. Only the two operands are BF16.
        float sum = 0.0f;
        const int32_t group = output_channel / shape.output_channels_per_group;
        for (int32_t input_channel = 0;
             input_channel < shape.input_channels_per_group; ++input_channel) {
          for (int32_t kernel_h = 0; kernel_h < shape.kernel_h; ++kernel_h) {
            const int64_t input_h =
                int64_t(output_h) * shape.stride_h +
                int64_t(kernel_h) * shape.dilation_h - shape.pad_h;
            if (input_h < 0 || input_h >= shape.input_h) {
              continue;
            }
            for (int32_t kernel_w = 0; kernel_w < shape.kernel_w;
                 ++kernel_w) {
              const int64_t input_w =
                  int64_t(output_w) * shape.stride_w +
                  int64_t(kernel_w) * shape.dilation_w - shape.pad_w;
              if (input_w < 0 || input_w >= shape.input_w) {
                continue;
              }

              const size_t input_index =
                  ((static_cast<size_t>(batch) * shape.input_channels +
                    group * shape.input_channels_per_group + input_channel) * shape.input_h +
                   input_h) *
                      shape.input_w +
                  input_w;
              const size_t weight_index =
                  ((static_cast<size_t>(output_channel) *
                        shape.input_channels_per_group +
                    input_channel) *
                       shape.kernel_h +
                   kernel_h) *
                      shape.kernel_w +
                  kernel_w;

              sum += bf16 ? Bf16ToFloat(input_bf16[input_index]) *
                                Bf16ToFloat(weight_bf16[weight_index])
                          : input[input_index] * weight[weight_index];
            }
          }
        }
        output[((static_cast<size_t>(batch) * shape.output_channels + output_channel) * shape.output_h +
                output_h) *
                   shape.output_w +
               output_w] = sum + (bias ? (*bias)[output_channel] : 0.0f);
      }
    }
  }
}

inline void ConvBf16Reference(
    const std::vector<float> &input, const std::vector<float> &weight,
    const std::vector<float> *bias, std::vector<float> &output,
    const ConvShape &shape) {
  ConvReference(input, weight, bias, output, shape, true);
}

inline void MatMulBf16Reference(
    const std::vector<float> &a, const std::vector<float> &b,
    std::vector<float> &output, const MatMulShape &shape) {
  if (a.size() != static_cast<size_t>(shape.a_elements)) {
    throw std::invalid_argument("BF16 MatMul input size does not match shape");
  }
  if (b.size() != static_cast<size_t>(shape.b_elements)) {
    throw std::invalid_argument("BF16 MatMul weight size does not match shape");
  }

  std::vector<uint16_t> a_bf16(a.size());
  std::vector<uint16_t> b_bf16(b.size());
  for (size_t index = 0; index < a.size(); ++index) {
    a_bf16[index] = FloatToBf16(a[index]);
  }
  for (size_t index = 0; index < b.size(); ++index) {
    b_bf16[index] = FloatToBf16(b[index]);
  }

  // Keep the same straightforward accumulation structure as the FP32
  // reference; only the two operands are BF16.
  output.assign(static_cast<size_t>(shape.y_elements), 0.0f);
  for (int32_t m = 0; m < shape.m; ++m) {
    for (int32_t n = 0; n < shape.n; ++n) {
      float sum = 0.0f;
      for (int32_t k = 0; k < shape.k; ++k) {
        sum += Bf16ToFloat(a_bf16[static_cast<size_t>(m) * shape.k + k]) *
               Bf16ToFloat(b_bf16[static_cast<size_t>(k) * shape.n + n]);
      }
      output[static_cast<size_t>(m) * shape.n + n] = sum;
    }
  }
}

} // namespace rvnpu
