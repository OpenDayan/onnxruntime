#include "runtime/device_context.h"
#include "runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string kernel_elf;
  int64_t rows{};
  int64_t columns{};
  int64_t k{};
};

Options Parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--kernel-elf" && index + 1 < argc)
      options.kernel_elf = argv[++index];
    else if (argument == "--rows" && index + 1 < argc)
      options.rows = std::stoll(argv[++index]);
    else if (argument == "--columns" && index + 1 < argc)
      options.columns = std::stoll(argv[++index]);
    else if (argument == "--k" && index + 1 < argc)
      options.k = std::stoll(argv[++index]);
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (options.kernel_elf.empty() || options.rows <= 0 ||
      options.columns <= 0 || options.k <= 0) {
    throw std::invalid_argument(
        "--kernel-elf, --rows, --columns and --k are required");
  }
  return options;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    const rvnpu::TopKShape shape = rvnpu::ValidateTopKShape(
        {options.rows, options.columns}, options.k, {});
    std::vector<float> input(shape.input_elements);
    for (size_t index = 0; index < input.size(); ++index) {
      input[index] = static_cast<float>((index * 7 + index / 5) % 11) *
                     0.125f;
    }
    std::vector<float> values(shape.output_elements);
    std::vector<int64_t> indices(shape.output_elements);
    rvnpu::DeviceContext context(options.kernel_elf);
    context.RunTopK(input.data(), values.data(), indices.data(), shape);

    for (int32_t row = 0; row < shape.rows; ++row) {
      std::vector<int64_t> order(shape.columns);
      std::iota(order.begin(), order.end(), 0);
      const size_t base = static_cast<size_t>(row) * shape.columns;
      std::stable_sort(order.begin(), order.end(),
                       [&](int64_t lhs, int64_t rhs) {
                         return input[base + lhs] > input[base + rhs];
                       });
      for (int32_t output_index = 0; output_index < shape.k;
           ++output_index) {
        const size_t result_index =
            static_cast<size_t>(row) * shape.k + output_index;
        const int64_t expected_index = order[output_index];
        const float expected_value = input[base + expected_index];
        if (indices[result_index] != expected_index ||
            std::abs(values[result_index] - expected_value) > 1e-6f) {
          throw std::runtime_error("TopK mismatch at output " +
                                   std::to_string(result_index));
        }
      }
    }
    std::cout << "PASS rows=" << shape.rows
              << " columns=" << shape.columns << " K=" << shape.k
              << std::endl;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
