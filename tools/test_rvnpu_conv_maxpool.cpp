#include "runtime/cpu_reference.h"
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
  std::vector<int64_t> input_shape;
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> kernel;
  std::vector<int64_t> stride{1, 1};
  std::vector<int64_t> pad{0, 0};
  std::string reference{"fp32"};
  bool bias{};
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
    else if (argument == "--op" && index + 1 < argc)
      options.op = argv[++index];
    else if (argument == "--input-shape" && index + 1 < argc)
      options.input_shape = ParseList(argv[++index]);
    else if (argument == "--weight-shape" && index + 1 < argc)
      options.weight_shape = ParseList(argv[++index]);
    else if (argument == "--kernel" && index + 1 < argc)
      options.kernel = ParseList(argv[++index]);
    else if (argument == "--stride" && index + 1 < argc)
      options.stride = ParseList(argv[++index]);
    else if (argument == "--pad" && index + 1 < argc)
      options.pad = ParseList(argv[++index]);
    else if (argument == "--reference" && index + 1 < argc)
      options.reference = argv[++index];
    else if (argument == "--bias")
      options.bias = true;
    else if (argument == "--fractional-input")
      options.fractional_input = true;
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (options.kernel_elf.empty() || options.input_shape.empty() ||
      (options.op != "conv" && options.op != "maxpool")) {
    throw std::invalid_argument(
        "--kernel-elf, --op and --input-shape are required");
  }
  if (options.stride.size() != 2 || options.pad.size() != 2)
    throw std::invalid_argument("--stride and --pad require two values");
  if (options.reference != "fp32" && options.reference != "bf16")
    throw std::invalid_argument("--reference must be fp32 or bf16");
  if (options.op == "maxpool" && options.reference != "fp32")
    throw std::invalid_argument("--reference bf16 is only valid for Conv");
  if (options.input_shape.size() != 4)
    throw std::invalid_argument("--input-shape requires four values");
  if (options.op == "conv" && options.weight_shape.size() != 4)
    throw std::invalid_argument("Conv requires four-value --weight-shape");
  if (options.op == "maxpool" && options.kernel.size() != 2)
    throw std::invalid_argument("MaxPool requires --kernel");
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

void ReferenceConv(const std::vector<float> &input,
                   const std::vector<float> &weight,
                   const std::vector<float> *bias,
                   std::vector<float> &output,
                   const rvnpu::ConvShape &shape) {
  for (int32_t oc = 0; oc < shape.output_channels; ++oc) {
    for (int32_t oh = 0; oh < shape.output_h; ++oh) {
      for (int32_t ow = 0; ow < shape.output_w; ++ow) {
        float sum = bias == nullptr ? 0.0f : (*bias)[oc];
        for (int32_t ic = 0; ic < shape.input_channels; ++ic) {
          for (int32_t kh = 0; kh < shape.kernel_h; ++kh) {
            const int32_t ih = oh * shape.stride_h + kh - shape.pad_h;
            if (ih < 0 || ih >= shape.input_h)
              continue;
            for (int32_t kw = 0; kw < shape.kernel_w; ++kw) {
              const int32_t iw = ow * shape.stride_w + kw - shape.pad_w;
              if (iw < 0 || iw >= shape.input_w)
                continue;
              const size_t input_index =
                  (static_cast<size_t>(ic) * shape.input_h + ih) *
                      shape.input_w +
                  iw;
              const size_t weight_index =
                  ((static_cast<size_t>(oc) * shape.input_channels + ic) *
                       shape.kernel_h +
                   kh) *
                      shape.kernel_w +
                  kw;
              sum += input[input_index] * weight[weight_index];
            }
          }
        }
        output[(static_cast<size_t>(oc) * shape.output_h + oh) *
                   shape.output_w +
               ow] = sum;
      }
    }
  }
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

    if (options.op == "conv") {
      rvnpu::ConvAttributes attributes;
      attributes.kernel_shape = {options.weight_shape[2],
                                 options.weight_shape[3]};
      attributes.strides = options.stride;
      attributes.pads = {options.pad[0], options.pad[1], options.pad[0],
                         options.pad[1]};
      std::vector<int64_t> bias_shape;
      if (options.bias)
        bias_shape = {options.weight_shape[0]};
      const auto shape = rvnpu::ValidateConvShapes(
          options.input_shape, options.weight_shape,
          options.bias ? &bias_shape : nullptr, attributes);
      std::vector<float> weight(shape.weight_elements);
      std::vector<float> bias(shape.bias_elements);
      std::vector<float> output(shape.output_elements);
      std::vector<float> reference(shape.output_elements);
      for (size_t index = 0; index < weight.size(); ++index)
        weight[index] = Value(index, 2, options.fractional_input);
      for (size_t index = 0; index < bias.size(); ++index)
        bias[index] = Value(index, 3, options.fractional_input);
      if (options.reference == "bf16") {
        rvnpu::ConvBf16Reference(
            input, weight, options.bias ? &bias : nullptr, reference, shape);
      } else {
        ReferenceConv(input, weight, options.bias ? &bias : nullptr, reference,
                      shape);
      }
      context.RunConv(input.data(), weight.data(),
                      options.bias ? bias.data() : nullptr, output.data(),
                      shape);
      std::cout << "PASS op=Conv input=" << ShapeText(options.input_shape)
                << " weight=" << ShapeText(options.weight_shape)
                << " stride=" << ShapeText(options.stride)
                << " pad=" << ShapeText(options.pad)
                << " reference=" << options.reference
                << " bias=" << (options.bias ? "yes" : "no")
                << " max_abs_diff="
                << Compare(output, reference, 2e-2f, 2e-2f)
                << std::endl;
      return 0;
    }

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
