#include "runtime/device_context.h"
#include "runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string kernel_elf;
  int m = 1;
  int n = 32;
  int k = 32;
  std::string a_file;
  std::string b_file;
};

Options Parse(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--kernel-elf" && i + 1 < argc)
      options.kernel_elf = argv[++i];
    else if (argument == "--m" && i + 1 < argc)
      options.m = std::stoi(argv[++i]);
    else if (argument == "--n" && i + 1 < argc)
      options.n = std::stoi(argv[++i]);
    else if (argument == "--k" && i + 1 < argc)
      options.k = std::stoi(argv[++i]);
    else if (argument == "--a-file" && i + 1 < argc)
      options.a_file = argv[++i];
    else if (argument == "--b-file" && i + 1 < argc)
      options.b_file = argv[++i];
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (options.kernel_elf.empty())
    throw std::invalid_argument("--kernel-elf is required");
  return options;
}

float AValue(int m, int k) {
  return static_cast<float>(((m * 17 + k * 13) % 29) - 14) * 0.0625f;
}

float BValue(int k, int n) {
  return static_cast<float>(((k * 7 + n * 19) % 31) - 15) * 0.03125f;
}

std::vector<float> ReadFloats(const std::string &path, size_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open input data: " + path);
  const auto bytes = stream.tellg();
  if (bytes != static_cast<std::streamoff>(expected * sizeof(float)))
    throw std::runtime_error("unexpected input data size: " + path);
  std::vector<float> values(expected);
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char *>(values.data()), bytes))
    throw std::runtime_error("cannot read input data: " + path);
  return values;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = Parse(argc, argv);
    const auto shape = rvnpu::ValidateMatMulShapes({options.m, options.k},
                                                   {options.k, options.n});
    std::vector<float> a(shape.a_elements);
    std::vector<float> b(shape.b_elements);
    std::vector<float> output(shape.y_elements, 0.0f);
    std::vector<float> reference(shape.y_elements, 0.0f);

    if (!options.a_file.empty()) {
      a = ReadFloats(options.a_file, shape.a_elements);
    } else {
      for (int m = 0; m < shape.m; ++m)
        for (int k = 0; k < shape.k; ++k)
          a[static_cast<size_t>(m) * shape.k + k] = AValue(m, k);
    }
    if (!options.b_file.empty()) {
      b = ReadFloats(options.b_file, shape.b_elements);
    } else {
      for (int k = 0; k < shape.k; ++k)
        for (int n = 0; n < shape.n; ++n)
          b[static_cast<size_t>(k) * shape.n + n] = BValue(k, n);
    }
    for (int m = 0; m < shape.m; ++m)
      for (int n = 0; n < shape.n; ++n)
        for (int k = 0; k < shape.k; ++k)
          reference[static_cast<size_t>(m) * shape.n + n] +=
              a[static_cast<size_t>(m) * shape.k + k] *
              b[static_cast<size_t>(k) * shape.n + n];

    rvnpu::DeviceContext context(options.kernel_elf);
    context.RunMatMul(a.data(), b.data(), output.data(), shape);

    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
      if (!std::isfinite(output[i]))
        throw std::runtime_error("non-finite output");
      max_abs_diff = std::max(max_abs_diff, std::abs(output[i] - reference[i]));
      const float tolerance = 1e-4f + 1e-4f * std::abs(reference[i]);
      if (std::abs(output[i] - reference[i]) > tolerance) {
        throw std::runtime_error("numeric mismatch at element " +
                                 std::to_string(i));
      }
    }
    std::cout << "PASS max_abs_diff=" << max_abs_diff << std::endl;
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "FAIL: " << ex.what() << std::endl;
    return 1;
  }
}
