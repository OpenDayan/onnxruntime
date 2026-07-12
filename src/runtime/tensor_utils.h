#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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

} // namespace rvnpu
