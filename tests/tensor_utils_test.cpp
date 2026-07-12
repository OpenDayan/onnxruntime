#include "runtime/tensor_utils.h"

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
  return 0;
}
