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
  std::vector<int64_t> input_shape;
  std::vector<int64_t> kernel;
  std::vector<int64_t> stride{1, 1};
  std::vector<int64_t> pad{0, 0};
  bool fractional_input{};
};

std::vector<int64_t> ParseList(const std::string &text) {
  std::vector<int64_t> values;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    values.push_back(std::stoll(text.substr(begin, end - begin)));
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return values;
}

Options Parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--kernel-elf" && index + 1 < argc)
      options.kernel_elf = argv[++index];
    else if (argument == "--input-shape" && index + 1 < argc)
      options.input_shape = ParseList(argv[++index]);
    else if (argument == "--kernel" && index + 1 < argc)
      options.kernel = ParseList(argv[++index]);
    else if (argument == "--stride" && index + 1 < argc)
      options.stride = ParseList(argv[++index]);
    else if (argument == "--pad" && index + 1 < argc)
      options.pad = ParseList(argv[++index]);
    else if (argument == "--fractional-input")
      options.fractional_input = true;
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (options.input_shape.size() != 4 || options.kernel.size() != 2 ||
      options.stride.size() != 2 || options.pad.size() != 2)
    throw std::invalid_argument("--input-shape needs 4 values; kernel/stride/pad need 2");
  return options;
}

size_t Elements(const std::vector<int64_t> &shape) {
  size_t result = 1;
  for (const int64_t dimension : shape)
    result *= static_cast<size_t>(dimension);
  return result;
}

float Value(size_t index, int salt, bool fractional = false) {
  const int integer = (static_cast<int>(index * 17 + salt * 13) % 37) - 18;
  if (fractional)
    return static_cast<float>(integer) / 61.0f;
  return static_cast<float>(integer) * 0.0625f;
}

float Compare(const std::vector<float> &actual,
              const std::vector<float> &reference, float absolute,
              float relative) {

  if (actual.size() != reference.size())
    throw std::runtime_error("output size mismatch");
  float maximum = 0.0f;
  for (size_t index = 0; index < actual.size(); ++index) {
    const float difference = std::abs(actual[index] - reference[index]);
    maximum = std::max(maximum, difference);
    if (!std::isfinite(actual[index]) ||
        difference > absolute + relative * std::abs(reference[index])) {
      throw std::runtime_error(
          "mismatch at element " + std::to_string(index) + ": got=" +
          std::to_string(actual[index]) +
          " reference=" + std::to_string(reference[index]));
    }
  }
  return maximum;
}

void ReferenceMaxPool(const std::vector<float> &input,
                      std::vector<float> &output,
                      const rvnpu::MaxPoolShape &shape) {
  for (int32_t n = 0; n < shape.batch; ++n) {
    for (int32_t c = 0; c < shape.channels; ++c) {
      for (int32_t oh = 0; oh < shape.output_h; ++oh) {
        for (int32_t ow = 0; ow < shape.output_w; ++ow) {
          float maximum = -3.402823466e+38F;
          for (int32_t kh = 0; kh < shape.kernel_h; ++kh) {
            const int32_t ih = oh * shape.stride_h + kh - shape.pad_h;
            if (ih < 0 || ih >= shape.input_h)
              continue;
            for (int32_t kw = 0; kw < shape.kernel_w; ++kw) {
              const int32_t iw = ow * shape.stride_w + kw - shape.pad_w;
              if (iw < 0 || iw >= shape.input_w)
                continue;
              const size_t input_index =
                  ((static_cast<size_t>(n) * shape.channels + c) *
                       shape.input_h +
                   ih) *
                      shape.input_w +
                  iw;
              maximum = std::max(maximum, input[input_index]);
            }
          }
          output[((static_cast<size_t>(n) * shape.channels + c) *
                      shape.output_h +
                  oh) *
                     shape.output_w +
                 ow] = maximum;
        }
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    rvnpu::DeviceContext context(options.kernel_elf);
    std::vector<float> input(Elements(options.input_shape));
    for (size_t index = 0; index < input.size(); ++index)
      input[index] = Value(index, 1, options.fractional_input);

    rvnpu::MaxPoolAttributes attributes;
    attributes.kernel_shape = options.kernel;
    attributes.strides = options.stride;
    attributes.pads = {options.pad[0], options.pad[1], options.pad[0],
                       options.pad[1]};
    const auto shape =
        rvnpu::ValidateMaxPoolShape(options.input_shape, attributes);
    std::vector<float> output(shape.output_elements);
    std::vector<float> reference(shape.output_elements);
    ReferenceMaxPool(input, reference, shape);
    context.RunMaxPool(input.data(), output.data(), shape);
    std::cout << "PASS op=MaxPool input=" << ShapeText(options.input_shape)
              << " kernel=" << ShapeText(options.kernel)
              << " stride=" << ShapeText(options.stride)
              << " pad=" << ShapeText(options.pad)
              << " max_abs_diff="
              << Compare(output, reference, 1e-4f, 1e-4f)
              << std::endl;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
