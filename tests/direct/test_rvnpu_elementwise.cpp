#include "runtime/device_context.h"
#include "runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string ShapeText(const std::vector<int64_t> &shape) {
  std::string text = "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0)
      text += ",";
    text += std::to_string(shape[index]);
  }
  return text + "]";
}

struct Options {
  std::string kernel_elf;
  std::string op;
  std::vector<int64_t> shape;
  std::vector<int64_t> rhs_shape;
  bool swap_inputs{};
};

std::vector<int64_t> ParseShape(const std::string &text) {
  std::vector<int64_t> shape;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    shape.push_back(std::stoll(text.substr(begin, end - begin)));
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  if (shape.empty())
    throw std::invalid_argument("shape must not be empty");
  return shape;
}

Options Parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--kernel-elf" && index + 1 < argc)
      options.kernel_elf = argv[++index];
    else if (argument == "--op" && index + 1 < argc)
      options.op = argv[++index];
    else if (argument == "--shape" && index + 1 < argc)
      options.shape = ParseShape(argv[++index]);
    else if (argument == "--rhs-shape" && index + 1 < argc)
      options.rhs_shape = ParseShape(argv[++index]);
    else if (argument == "--swap-inputs")
      options.swap_inputs = true;
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (options.op.empty() ||
      options.shape.empty()) {
    throw std::invalid_argument("--op and --shape are required");
  }
  if (options.op != "add" && options.op != "mul" && options.op != "sub" &&
      options.op != "div" && options.op != "sigmoid" &&
      options.op != "silu") {
    throw std::invalid_argument(
        "--op must be add, mul, sub, div, sigmoid or silu");
  }
  if (options.op != "sigmoid" && options.op != "silu" &&
      options.rhs_shape.empty())
    options.rhs_shape = options.shape;
  return options;
}

size_t Elements(const std::vector<int64_t> &shape) {
  return rvnpu::ValidateUnaryTensorShape(shape, "test").elements;
}

float InputValue(size_t index, int salt) {
  return static_cast<float>((static_cast<int>(index * 17 + salt * 11) % 41) -
                            20) *
         0.125f;
}

size_t BroadcastIndex(size_t output_index,
                      const std::vector<int64_t> &output_shape,
                      const std::vector<int64_t> &input_shape) {
  size_t input_index = 0;
  size_t input_stride = 1;
  for (size_t offset = 0; offset < output_shape.size(); ++offset) {
    const size_t output_axis = output_shape.size() - 1 - offset;
    const size_t coordinate =
        output_index % static_cast<size_t>(output_shape[output_axis]);
    output_index /= static_cast<size_t>(output_shape[output_axis]);
    if (offset < input_shape.size()) {
      const size_t input_axis = input_shape.size() - 1 - offset;
      if (input_shape[input_axis] != 1)
        input_index += coordinate * input_stride;
      input_stride *= static_cast<size_t>(input_shape[input_axis]);
    }
  }
  return input_index;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    rvnpu::DeviceContext context(options.kernel_elf);

    if (options.op == "sigmoid" || options.op == "silu") {
      const auto shape =
          rvnpu::ValidateUnaryTensorShape(options.shape, options.op.c_str());
      std::vector<float> input(shape.elements);
      std::vector<float> output(shape.elements);
      for (size_t index = 0; index < input.size(); ++index)
        input[index] = InputValue(index, 1);
      if (options.op == "sigmoid")
        context.RunSigmoid(input.data(), output.data(), shape);
      else
        context.RunSilu(input.data(), output.data(), shape);
      float maximum = 0.0f;
      for (size_t index = 0; index < output.size(); ++index) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-input[index]));
        const float reference =
            options.op == "sigmoid" ? sigmoid : input[index] * sigmoid;
        maximum = std::max(maximum, std::abs(output[index] - reference));
        if (!std::isfinite(output[index]) ||
            std::abs(output[index] - reference) > 1e-4f) {
          throw std::runtime_error(options.op + " mismatch at element " +
                                   std::to_string(index) +
                                   ": input=" + std::to_string(input[index]) +
                                   " got=" + std::to_string(output[index]) +
                                   " reference=" + std::to_string(reference));
        }
      }
      std::cout << "PASS op=" << options.op
                << " shape=" << ShapeText(options.shape)
                << " max_abs_diff=" << maximum << std::endl;
      return 0;
    }

    std::vector<int64_t> lhs_shape = options.shape;
    std::vector<int64_t> rhs_shape = options.rhs_shape;
    if (options.swap_inputs)
      std::swap(lhs_shape, rhs_shape);
    const auto shape = rvnpu::ValidateBinaryElementwiseShapes(
        lhs_shape, rhs_shape, options.op.c_str());
    std::vector<float> lhs(Elements(lhs_shape));
    std::vector<float> rhs(Elements(rhs_shape));
    std::vector<float> output(shape.output_elements);
    for (size_t index = 0; index < lhs.size(); ++index)
      lhs[index] = InputValue(index, 2);
    for (size_t index = 0; index < rhs.size(); ++index)
      rhs[index] = InputValue(index, 3);

    if (options.op == "add")
      context.RunAdd(lhs.data(), rhs.data(), output.data(), shape);
    else if (options.op == "mul")
      context.RunMul(lhs.data(), rhs.data(), output.data(), shape);
    else if (options.op == "sub")
      context.RunSub(lhs.data(), rhs.data(), output.data(), shape);
    else
      context.RunDiv(lhs.data(), rhs.data(), output.data(), shape);

    float maximum = 0.0f;
    for (size_t index = 0; index < output.size(); ++index) {
      const float a = lhs[BroadcastIndex(index, shape.output_shape, lhs_shape)];
      const float b = rhs[BroadcastIndex(index, shape.output_shape, rhs_shape)];
      const float reference = options.op == "add"   ? a + b
                              : options.op == "mul" ? a * b
                              : options.op == "sub" ? a - b
                                                    : a / b;
      maximum = std::max(maximum, std::abs(output[index] - reference));
      const float tolerance = options.op == "div" ? 1e-4f : 1e-6f;
      if (!std::isfinite(output[index]) ||
          std::abs(output[index] - reference) > tolerance) {
        throw std::runtime_error(options.op + " mismatch at element " +
                                 std::to_string(index) +
                                 ": got=" + std::to_string(output[index]) +
                                 " reference=" + std::to_string(reference));
      }
    }
    std::cout << "PASS op=" << options.op << " lhs=" << ShapeText(lhs_shape)
              << " rhs=" << ShapeText(rhs_shape)
              << " output=" << ShapeText(shape.output_shape)
              << " max_abs_diff=" << maximum << std::endl;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
