#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr const char *kRegistrationName = "RVNPUExecutionProvider.virtual";
constexpr const char *kEpName = "RVNPUExecutionProvider";

struct Options {
  std::string model;
  std::string plugin;
  std::string kernel_elf;
  std::string data_dir;
  std::string expected_operator;
  std::unordered_map<std::string, size_t> expected_rvnpu_counts;
  float absolute = 1e-4f;
  float relative = 1e-4f;
  bool report_max_error_percent = false;
};

std::unordered_map<std::string, size_t> ParseCounts(const std::string &text) {
  std::unordered_map<std::string, size_t> counts;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    const std::string item = text.substr(begin, end - begin);
    const size_t equals = item.find('=');
    if (equals == std::string::npos || equals == 0 ||
        equals + 1 == item.size()) {
      throw std::invalid_argument("invalid assignment count: " + item);
    }
    const std::string name = item.substr(0, equals);
    const size_t count = std::stoull(item.substr(equals + 1));
    if (!counts.emplace(name, count).second)
      throw std::invalid_argument("duplicate assignment count: " + name);
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return counts;
}

Options Parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--model" && index + 1 < argc)
      options.model = argv[++index];
    else if (argument == "--plugin" && index + 1 < argc)
      options.plugin = argv[++index];
    else if (argument == "--kernel-elf" && index + 1 < argc)
      options.kernel_elf = argv[++index];
    else if (argument == "--data-dir" && index + 1 < argc)
      options.data_dir = argv[++index];
    else if (argument == "--expect-rvnpu" && index + 1 < argc) {
      options.expected_operator = argv[++index];
    } else if (argument == "--expect-rvnpu-counts" && index + 1 < argc) {
      options.expected_rvnpu_counts = ParseCounts(argv[++index]);
    } else if (argument == "--atol" && index + 1 < argc)
      options.absolute = std::stof(argv[++index]);
    else if (argument == "--rtol" && index + 1 < argc)
      options.relative = std::stof(argv[++index]);
    else if (argument == "--report-max-error-percent")
      options.report_max_error_percent = true;
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (options.model.empty() || options.plugin.empty() ||
      options.kernel_elf.empty() || options.data_dir.empty() ||
      (options.expected_operator.empty() &&
       options.expected_rvnpu_counts.empty())) {
    throw std::invalid_argument(
        "--model, --plugin, --kernel-elf, --data-dir and assignment "
        "expectations are required");
  }
  if (options.absolute < 0.0f || options.relative < 0.0f)
    throw std::invalid_argument("invalid tolerance");
  return options;
}

std::vector<float> ReadFloats(const std::filesystem::path &path,
                              size_t expected = 0) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open data file: " + path.string());
  const auto bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
    throw std::runtime_error("invalid float data file: " + path.string());
  std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char *>(values.data()), bytes))
    throw std::runtime_error("cannot read data file: " + path.string());
  if (expected != 0 && values.size() != expected)
    throw std::runtime_error("unexpected element count: " + path.string());
  return values;
}

size_t Elements(const std::vector<int64_t> &shape) {
  if (shape.empty())
    throw std::runtime_error("scalar tensors are not supported by this runner");
  size_t result = 1;
  for (const int64_t dimension : shape) {
    if (dimension <= 0)
      throw std::runtime_error("runner requires static positive input shapes");
    result *= static_cast<size_t>(dimension);
  }
  return result;
}

struct Input {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

std::vector<Input> LoadInputs(const Ort::Session &session,
                              const std::filesystem::path &directory) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Input> inputs;
  for (size_t index = 0; index < session.GetInputCount(); ++index) {
    const auto name = session.GetInputNameAllocated(index, allocator);
    const auto type_info = session.GetInputTypeInfo(index);
    const auto info = type_info.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("operator runner only supports FP32 inputs");
    Input input;
    input.name = name.get();
    input.shape = info.GetShape();
    input.data =
        ReadFloats(directory / (input.name + ".bin"), Elements(input.shape));
    inputs.push_back(std::move(input));
  }
  return inputs;
}

std::vector<float> Run(Ort::Session &session, std::vector<Input> &inputs) {
  Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<const char *> input_names;
  std::vector<Ort::Value> input_values;
  input_names.reserve(inputs.size());
  input_values.reserve(inputs.size());
  for (auto &input : inputs) {
    input_names.push_back(input.name.c_str());
    input_values.push_back(Ort::Value::CreateTensor<float>(
        memory, input.data.data(), input.data.size(), input.shape.data(),
        input.shape.size()));
  }

  Ort::AllocatorWithDefaultOptions allocator;
  const auto output_name = session.GetOutputNameAllocated(0, allocator);
  const char *output_names[] = {output_name.get()};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                             input_values.data(), input_values.size(),
                             output_names, 1);
  if (outputs.size() != 1)
    throw std::runtime_error("operator model must have exactly one output");
  const auto info = outputs[0].GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("operator runner only supports FP32 output");
  const size_t count = info.GetElementCount();
  const float *data = outputs[0].GetTensorData<float>();
  return {data, data + count};
}

struct Comparison {
  float max_abs_diff;
  float max_error_percent;
};

Comparison Compare(const std::vector<float> &actual,
                   const std::vector<float> &expected, float absolute,
                   float relative, const std::string &label,
                   bool report_max_error_percent) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " output size mismatch");
  float maximum = 0.0f;
  float maximum_reference = 0.0f;
  size_t first_mismatch = actual.size();
  for (size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]))
      throw std::runtime_error(label + " has non-finite output");
    const float difference = std::abs(actual[index] - expected[index]);
    maximum = std::max(maximum, difference);
    maximum_reference =
        std::max(maximum_reference, std::abs(expected[index]));
    if (first_mismatch == actual.size() &&
        difference > absolute + relative * std::abs(expected[index])) {
      first_mismatch = index;
    }
  }
  const float maximum_error_percent =
      maximum_reference == 0.0f
          ? (maximum == 0.0f ? 0.0f
                             : std::numeric_limits<float>::infinity())
          : maximum / maximum_reference * 100.0f;
  if (first_mismatch != actual.size()) {
    std::string message =
        label + " mismatch at element " + std::to_string(first_mismatch) +
        ": got=" + std::to_string(actual[first_mismatch]) +
        " reference=" + std::to_string(expected[first_mismatch]) +
        " max_abs_diff=" + std::to_string(maximum);
    if (report_max_error_percent) {
      message +=
          " max_error_percent=" + std::to_string(maximum_error_percent) + "%";
    }
    throw std::runtime_error(message);
  }
  return {maximum, maximum_error_percent};
}

void CheckAssignment(const Ort::Session &session, const Options &options) {
  std::unordered_map<std::string, size_t> rvnpu_counts;
  for (const auto &subgraph : session.GetEpGraphAssignmentInfo()) {
    const std::string ep_name = subgraph.GetEpName();
    for (const auto &node : subgraph.GetNodes()) {
      const std::string op = node.GetOperatorType();
      if (ep_name == kEpName)
        ++rvnpu_counts[op];
    }
  }

  if (!options.expected_operator.empty()) {
    const size_t actual = rvnpu_counts[options.expected_operator];
    if (actual == 0)
      throw std::runtime_error(options.expected_operator +
                               " was not assigned to RVNPU EP");
  }

  for (const auto &[name, expected] : options.expected_rvnpu_counts) {
    const size_t actual = rvnpu_counts[name];
    if (actual != expected) {
      throw std::runtime_error("RVNPU assignment count for " + name +
                               " is " + std::to_string(actual) +
                               ", expected " + std::to_string(expected));
    }
  }
  for (const auto &[name, count] : rvnpu_counts) {
    const bool allowed_by_single =
        !options.expected_operator.empty() && name == options.expected_operator;
    if (!allowed_by_single && options.expected_rvnpu_counts.count(name) == 0) {
      throw std::runtime_error("RVNPU EP claimed unexpected operator " + name);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test-rvnpu-onnx-model");

    Ort::SessionOptions cpu_options;
    cpu_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session cpu_session(env, options.model.c_str(), cpu_options);
    auto inputs = LoadInputs(cpu_session, options.data_dir);
    const auto cpu = Run(cpu_session, inputs);

    env.RegisterExecutionProviderLibrary(kRegistrationName, options.plugin);
    {
      std::vector<Ort::ConstEpDevice> selected;
      for (const auto &device : env.GetEpDevices()) {
        if (std::string(device.EpName()) == kEpName)
          selected.push_back(device);
      }
      if (selected.size() != 1)
        throw std::runtime_error("expected exactly one RVNPU EP device");

      Ort::SessionOptions session_options;
      session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
      session_options.AddConfigEntry("session.record_ep_graph_assignment_info",
                                     "1");
      const std::unordered_map<std::string, std::string> ep_options{
          {"kernel_elf", options.kernel_elf}};
      session_options.AppendExecutionProvider_V2(env, selected, ep_options);
      Ort::Session session(env, options.model.c_str(), session_options);
      CheckAssignment(session, options);

      const auto output = Run(session, inputs);
      const Comparison comparison =
          Compare(output, cpu, options.absolute, options.relative,
                  "CPU/RVNPU", options.report_max_error_percent);
      std::cout << "CPU vs RVNPU max_abs_diff=" << comparison.max_abs_diff;
      if (options.report_max_error_percent) {
        std::cout << " max_error_percent=" << comparison.max_error_percent
                  << "%";
      }
      std::cout << std::endl;
    }

    env.UnregisterExecutionProviderLibrary(kRegistrationName);
    std::cout << "PASS" << std::endl;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
