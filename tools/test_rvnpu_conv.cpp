#include "runtime/cpu_reference.h"
#include "runtime/device_context.h"
#include "runtime/tensor_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <iostream>
#include <random>
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
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> stride{1, 1};
  std::vector<int64_t> pad{0, 0};
  std::vector<int64_t> dilation{1, 1};
  int64_t group{1};
  std::string auto_pad{"NOTSET"};
  std::string pattern{"legacy"};
  std::string data_dir;
  std::string reference_output;
  std::string device_output;
  std::string reference{"fp32"};
  bool bias{};
  bool fractional_input{};
  bool strict{};
  bool pad_provided{};
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
    else if (argument == "--weight-shape" && index + 1 < argc)
      options.weight_shape = ParseList(argv[++index]);
    else if (argument == "--stride" && index + 1 < argc)
      options.stride = ParseList(argv[++index]);
    else if (argument == "--pad" && index + 1 < argc) {
      options.pad = ParseList(argv[++index]);
      options.pad_provided = true;
    }
    else if (argument == "--dilation" && index + 1 < argc)
      options.dilation = ParseList(argv[++index]);
    else if (argument == "--group" && index + 1 < argc)
      options.group = std::stoll(argv[++index]);
    else if (argument == "--auto-pad" && index + 1 < argc)
      options.auto_pad = argv[++index];
    else if (argument == "--pattern" && index + 1 < argc)
      options.pattern = argv[++index];
    else if (argument == "--data-dir" && index + 1 < argc)
      options.data_dir = argv[++index];
    else if (argument == "--reference-output" && index + 1 < argc)
      options.reference_output = argv[++index];
    else if (argument == "--device-output" && index + 1 < argc)
      options.device_output = argv[++index];
    else if (argument == "--reference" && index + 1 < argc)
      options.reference = argv[++index];
    else if (argument == "--bias")
      options.bias = true;
    else if (argument == "--fractional-input")
      options.fractional_input = true;
    else if (argument == "--strict")
      options.strict = true;
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (options.input_shape.empty())
    throw std::invalid_argument("--input-shape is required");
  if (!options.device_output.empty()) {
    if (!options.reference_output.empty())
      throw std::invalid_argument("--device-output requires device Conv execution, not CPU reference export");
    if (std::filesystem::exists(options.device_output) ||
        std::filesystem::is_symlink(options.device_output))
      throw std::invalid_argument("--device-output refuses to overwrite an existing artifact");
  }
  if (options.stride.size() != 2 || options.dilation.size() != 2 ||
      (options.pad.size() != 2 && options.pad.size() != 4))
    throw std::invalid_argument("stride/dilation require 2 values; Conv pad requires 2 or 4");
  if (options.reference != "fp32" && options.reference != "bf16")
    throw std::invalid_argument("--reference must be fp32 or bf16");
  if (options.pattern != "legacy" && options.pattern != "random" &&
      options.pattern != "zero" && options.pattern != "impulse")
    throw std::invalid_argument("pattern must be legacy/random/zero/impulse");
  if (options.input_shape.size() != 4)
    throw std::invalid_argument("--input-shape requires four values");
  if (options.weight_shape.size() != 4)
    throw std::invalid_argument("Conv requires four-value --weight-shape");
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

void ReadData(const std::string &path, std::vector<float> &values) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const size_t bytes = values.size() * sizeof(float);
  if (!file || file.tellg() != static_cast<std::streamoff>(bytes))
    throw std::runtime_error("invalid tensor data size: " + path);
  file.seekg(0);
  if (!file.read(reinterpret_cast<char *>(values.data()), bytes))
    throw std::runtime_error("cannot read tensor data: " + path);
}

float Compare(const std::vector<float> &actual,
              const std::vector<float> &reference, float absolute,
              float relative) {

  if (actual.size() != reference.size())
    throw std::runtime_error("output size mismatch");
  float maximum = 0.0f;
  float max_relative = 0.0f;
  size_t failures = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const float difference = std::abs(actual[index] - reference[index]);
    maximum = std::max(maximum, difference);
    max_relative = std::max(max_relative, difference /
        std::max(std::abs(reference[index]), 1e-12f));
    if (!std::isfinite(actual[index]) || !std::isfinite(reference[index]) ||
        difference > absolute + relative * std::abs(reference[index])) {
      if (failures < 8)
        std::cerr << "MISMATCH index=" << index << " actual=" << actual[index]
                  << " reference=" << reference[index] << "\n";
      ++failures;
    }
  }
  std::cout << "COMPARE max_abs_diff=" << maximum
            << " max_rel_diff=" << max_relative
            << " failed_elements=" << failures << "\n";
  if (failures)
    throw std::runtime_error("Conv numerical comparison failed");
  return maximum;
}

void ReferenceConv(const std::vector<float> &input,
                   const std::vector<float> &weight,
                   const std::vector<float> *bias,
                   std::vector<float> &output,
                   const rvnpu::ConvShape &shape) {
  rvnpu::ConvReference(input, weight, bias, output, shape, false);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    std::vector<float> input(Elements(options.input_shape));
    for (size_t index = 0; index < input.size(); ++index)
      input[index] = Value(index, 1, options.fractional_input);
    std::mt19937 generator(20261020);
    std::uniform_real_distribution<float> random(-0.3f, 0.3f);
    if (options.pattern == "random")
      for (float &value : input) value = random(generator);
    else if (options.pattern == "zero" || options.pattern == "impulse") {
      std::fill(input.begin(), input.end(), 0.0f);
      if (options.pattern == "impulse" && !input.empty()) input[0] = 1;
    }

    rvnpu::ConvAttributes attributes;
    attributes.kernel_shape = {options.weight_shape[2],
                               options.weight_shape[3]};
    attributes.strides = options.stride;
    attributes.pads = options.pad.size() == 4 ? options.pad :
        std::vector<int64_t>{options.pad[0], options.pad[1], options.pad[0], options.pad[1]};
    attributes.pads_provided = options.pad_provided;
    attributes.dilations = options.dilation;
    attributes.group = options.group;
    attributes.auto_pad = options.auto_pad;
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
    if (options.pattern != "legacy") {
      for (float &value : weight) value = random(generator);
      for (float &value : bias) value = random(generator);
    }
    if (!options.data_dir.empty()) {
      ReadData(options.data_dir + "/input.bin", input);
      ReadData(options.data_dir + "/weight.bin", weight);
      if (options.bias) ReadData(options.data_dir + "/bias.bin", bias);
    }
    if (options.reference == "bf16") {
      rvnpu::ConvBf16Reference(
          input, weight, options.bias ? &bias : nullptr, reference, shape);
    } else {
      ReferenceConv(input, weight, options.bias ? &bias : nullptr, reference,
                    shape);
    }
    if (!options.reference_output.empty()) {
      std::ofstream file(options.reference_output, std::ios::binary);
      if (!file.write(reinterpret_cast<const char *>(reference.data()),
                      reference.size() * sizeof(float)))
        throw std::runtime_error("cannot write CPU reference");
      std::cout << "HOST_REFERENCE output=" << ShapeText(shape.output_shape)
                << " reference=" << options.reference << "\n";
      return 0;
    }
    rvnpu::DeviceContext context(options.kernel_elf);
    const auto start = std::chrono::steady_clock::now();
    context.RunConv(input.data(), weight.data(),
                    options.bias ? bias.data() : nullptr, output.data(),
                    shape);
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    // Historical dyadic data and zero inputs have exact expected results.
    // Do not silently turn their zero-error regressions into tolerant tests.
    const bool strict = options.strict || options.pattern == "zero" ||
        (options.pattern == "legacy" && !options.fractional_input &&
         options.data_dir.empty());
    const float tolerance = strict ? 0.f :
        (options.reference == "bf16" ? 1e-4f : 2e-2f);
    const float max_diff = Compare(output, reference, tolerance, tolerance);
    if (!options.device_output.empty()) {
      // Exclusive creation also prevents a race with another writer or a
      // symlink inserted after argument validation.
      FILE *file = std::fopen(options.device_output.c_str(), "wbx");
      if (!file)
        throw std::runtime_error("cannot create device output artifact exclusively");
      const bool written = std::fwrite(output.data(), sizeof(float), output.size(), file) == output.size();
      const bool closed = std::fclose(file) == 0;
      if (!written || !closed)
        throw std::runtime_error("cannot write checked device output");
      std::cout << "DEVICE_OUTPUT path=" << options.device_output
                << " elements=" << output.size() << "\n";
    }
    std::cout << "PASS op=Conv input=" << ShapeText(options.input_shape)
              << " weight=" << ShapeText(options.weight_shape)
              << " stride=" << ShapeText(options.stride)
              << " pad=" << ShapeText(options.pad)
              << " dilation=" << ShapeText(options.dilation)
              << " group=" << options.group << " auto_pad=" << options.auto_pad
              << " output=" << ShapeText(shape.output_shape)
              << " tile=8,8,32,16"
              << " pipeline=1 cores=4 wall_ms=" << wall_ms
              << " cycle=unavailable"
              << " reference=" << options.reference
              << " strict=" << (strict ? "yes" : "no")
              << " bias=" << (options.bias ? "yes" : "no")
              << " max_abs_diff="
              << max_diff
              << std::endl;
    return 0;

  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
