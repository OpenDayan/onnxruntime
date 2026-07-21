#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char *kRegistrationName = "RVNPUExecutionProvider.virtual";
constexpr const char *kEpName = "RVNPUExecutionProvider";

struct Options {
  std::string model;
  std::string plugin;
  std::string kernel_elf;
  std::string data_dir;
};

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
    else {
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
    }
  }
  if (options.model.empty() || options.plugin.empty() ||
      options.kernel_elf.empty() || options.data_dir.empty()) {
    throw std::invalid_argument("missing TopK runner argument");
  }
  return options;
}

template <typename T>
std::vector<T> ReadBinary(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open " + path.string());
  const auto bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0)
    throw std::runtime_error("invalid binary file " + path.string());
  std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char *>(values.data()), bytes))
    throw std::runtime_error("cannot read " + path.string());
  return values;
}

struct Result {
  std::vector<float> values;
  std::vector<int64_t> indices;
};

Result Run(Ort::Session &session, std::vector<float> &input,
           const std::vector<int64_t> &input_shape) {
  Ort::AllocatorWithDefaultOptions allocator;
  const auto input_name = session.GetInputNameAllocated(0, allocator);
  const auto values_name = session.GetOutputNameAllocated(0, allocator);
  const auto indices_name = session.GetOutputNameAllocated(1, allocator);
  const char *input_names[] = {input_name.get()};
  const char *output_names[] = {values_name.get(), indices_name.get()};
  Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_value = Ort::Value::CreateTensor<float>(
      memory, input.data(), input.size(), input_shape.data(),
      input_shape.size());
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                             &input_value, 1, output_names, 2);
  if (outputs.size() != 2)
    throw std::runtime_error("TopK model must have two outputs");
  const auto values_info = outputs[0].GetTensorTypeAndShapeInfo();
  const auto indices_info = outputs[1].GetTensorTypeAndShapeInfo();
  if (values_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      indices_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      values_info.GetElementCount() != indices_info.GetElementCount()) {
    throw std::runtime_error("invalid TopK output types");
  }
  const size_t count = values_info.GetElementCount();
  const float *values = outputs[0].GetTensorData<float>();
  const int64_t *indices = outputs[1].GetTensorData<int64_t>();
  return {{values, values + count}, {indices, indices + count}};
}

float Compare(const Result &actual, const Result &reference,
              const char *label) {
  if (actual.values.size() != reference.values.size() ||
      actual.indices.size() != reference.indices.size()) {
    throw std::runtime_error(std::string(label) + " size mismatch");
  }
  float maximum = 0.0f;
  for (size_t index = 0; index < actual.values.size(); ++index) {
    const float difference =
        std::abs(actual.values[index] - reference.values[index]);
    maximum = std::max(maximum, difference);
    if (actual.indices[index] != reference.indices[index] ||
        difference > 1e-6f) {
      throw std::runtime_error(std::string(label) + " mismatch at " +
                               std::to_string(index));
    }
  }
  return maximum;
}

void CheckAssignment(const Ort::Session &session) {
  size_t rvnpu = 0;
  size_t cpu = 0;
  for (const auto &subgraph : session.GetEpGraphAssignmentInfo()) {
    for (const auto &node : subgraph.GetNodes()) {
      if (node.GetOperatorType() != "TopK")
        continue;
      if (std::string(subgraph.GetEpName()) == kEpName)
        ++rvnpu;
      else
        ++cpu;
    }
  }
  if (rvnpu != 1 || cpu != 0) {
    throw std::runtime_error("unexpected TopK assignment");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    const auto input_path =
        std::filesystem::path(options.data_dir) / "input.bin";
    std::vector<float> input = ReadBinary<float>(input_path);
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test-rvnpu-topk-onnx");
    Ort::SessionOptions cpu_options;
    cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session cpu_session(env, options.model.c_str(), cpu_options);
    const auto type_info = cpu_session.GetInputTypeInfo(0);
    const auto input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    const Result cpu = Run(cpu_session, input, input_shape);

    env.RegisterExecutionProviderLibrary(kRegistrationName, options.plugin);
    {
      std::vector<Ort::ConstEpDevice> selected;
      for (const auto &device : env.GetEpDevices()) {
        if (std::string(device.EpName()) == kEpName)
          selected.push_back(device);
      }
      Ort::SessionOptions session_options;
      session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
      session_options.AddConfigEntry("session.record_ep_graph_assignment_info",
                                     "1");
      const std::unordered_map<std::string, std::string> ep_options{
          {"kernel_elf", options.kernel_elf}};
      session_options.AppendExecutionProvider_V2(env, selected, ep_options);
      Ort::Session session(env, options.model.c_str(), session_options);
      CheckAssignment(session);
      const Result result = Run(session, input, input_shape);
      const float difference = Compare(result, cpu, "CPU/RVNPU");
      std::cout << "CPU vs RVNPU max_abs_diff=" << difference << std::endl;
    }
    env.UnregisterExecutionProviderLibrary(kRegistrationName);
    std::cout << "PASS" << std::endl;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
