#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char *kRegistrationName = "DayanExecutionProvider.virtual";
constexpr const char *kEpName = "DayanExecutionProvider";

struct Options {
  std::string model;
  std::string plugin;
  std::string kernel_elf;
  std::string data_dir;
  bool expect_topk{};
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
    else if (argument == "--expect-topk") {
      options.expect_topk = true;
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
    }
  }
  if (options.model.empty() || options.plugin.empty() ||
      options.data_dir.empty()) {
    throw std::invalid_argument("missing NMS runner argument");
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

std::vector<int64_t> Run(Ort::Session &session, std::vector<float> &boxes,
                         std::vector<float> &scores) {
  Ort::AllocatorWithDefaultOptions allocator;
  const auto boxes_name = session.GetInputNameAllocated(0, allocator);
  const auto scores_name = session.GetInputNameAllocated(1, allocator);
  const auto output_name = session.GetOutputNameAllocated(0, allocator);
  const auto boxes_shape = session.GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
  const auto scores_shape = session.GetInputTypeInfo(1)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
  auto elements = [](const std::vector<int64_t> &shape) {
    size_t count = 1;
    for (const int64_t dimension : shape) {
      if (dimension <= 0)
        throw std::runtime_error("NMS runner requires static input shapes");
      count *= static_cast<size_t>(dimension);
    }
    return count;
  };
  if (elements(boxes_shape) != boxes.size() ||
      elements(scores_shape) != scores.size()) {
    throw std::runtime_error("NMS input data size mismatch");
  }
  Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;
  inputs.push_back(Ort::Value::CreateTensor<float>(
      memory, boxes.data(), boxes.size(), boxes_shape.data(),
      boxes_shape.size()));
  inputs.push_back(Ort::Value::CreateTensor<float>(
      memory, scores.data(), scores.size(), scores_shape.data(),
      scores_shape.size()));
  const char *input_names[] = {boxes_name.get(), scores_name.get()};
  const char *output_names[] = {output_name.get()};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                             inputs.data(), inputs.size(), output_names, 1);
  const auto info = outputs[0].GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    throw std::runtime_error("NMS output must be INT64");
  const size_t count = info.GetElementCount();
  const int64_t *data = outputs[0].GetTensorData<int64_t>();
  return {data, data + count};
}

void CheckAssignment(const Ort::Session &session, bool expect_topk) {
  size_t dayan = 0;
  size_t cpu = 0;
  size_t dayan_topk = 0;
  for (const auto &subgraph : session.GetEpGraphAssignmentInfo()) {
    for (const auto &node : subgraph.GetNodes()) {
      if (node.GetOperatorType() == "TopK" &&
          std::string(subgraph.GetEpName()) == kEpName) {
        ++dayan_topk;
      }
      if (node.GetOperatorType() != "NonMaxSuppression")
        continue;
      if (std::string(subgraph.GetEpName()) == kEpName)
        ++dayan;
      else
        ++cpu;
    }
  }
  if (dayan != 1 || cpu != 0) {
    throw std::runtime_error("unexpected NMS assignment");
  }
  if ((expect_topk && dayan_topk != 1) || (!expect_topk && dayan_topk != 0))
    throw std::runtime_error("unexpected TopK assignment in NMS graph");
}

int64_t Compare(const std::vector<int64_t> &actual,
                const std::vector<int64_t> &expected) {
  if (actual.size() != expected.size())
    throw std::runtime_error("CPU/Dayan NMS output size mismatch");
  int64_t maximum = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const int64_t difference = std::abs(actual[index] - expected[index]);
    maximum = std::max(maximum, difference);
  }
  if (maximum != 0)
    throw std::runtime_error("CPU/Dayan NMS output mismatch");
  return maximum;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    const auto directory = std::filesystem::path(options.data_dir);
    std::vector<float> boxes = ReadBinary<float>(directory / "boxes.bin");
    std::vector<float> scores = ReadBinary<float>(directory / "scores.bin");
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test-dayan-nms-onnx");
    Ort::SessionOptions cpu_options;
    cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session cpu_session(env, options.model.c_str(), cpu_options);
    const auto cpu = Run(cpu_session, boxes, scores);

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
      CheckAssignment(session, options.expect_topk);
      const auto output = Run(session, boxes, scores);
      const int64_t difference = Compare(output, cpu);
      std::cout << "CPU vs Dayan max_abs_diff=" << difference << std::endl;
    }
    env.UnregisterExecutionProviderLibrary(kRegistrationName);
    std::cout << "PASS" << std::endl;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
