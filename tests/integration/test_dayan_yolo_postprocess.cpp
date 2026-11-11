#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char *kRegistrationName = "DayanExecutionProvider.virtual";
constexpr const char *kEpName = "DayanExecutionProvider";

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

size_t Elements(const std::vector<int64_t> &shape) {
  size_t result = 1;
  for (const int64_t dimension : shape) {
    if (dimension <= 0)
      throw std::runtime_error("postprocess input shape must be static");
    result *= static_cast<size_t>(dimension);
  }
  return result;
}

struct Output {
  std::vector<float> boxes;
  std::vector<float> scores;
  std::vector<int64_t> classes;
  std::vector<int64_t> boxes_shape;
  std::vector<int64_t> scores_shape;
  std::vector<int64_t> classes_shape;
};

Output Run(Ort::Session &session, std::vector<float> &raw) {
  if (session.GetInputCount() != 1 || session.GetOutputCount() != 3)
    throw std::runtime_error("unexpected YOLO postprocess model interface");
  Ort::AllocatorWithDefaultOptions allocator;
  const auto input_name = session.GetInputNameAllocated(0, allocator);
  const auto input_shape = session.GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
  if (Elements(input_shape) != raw.size())
    throw std::runtime_error("YOLO postprocess input size mismatch");

  std::vector<Ort::AllocatedStringPtr> output_names_owned;
  std::vector<const char *> output_names;
  for (size_t index = 0; index < 3; ++index) {
    output_names_owned.push_back(session.GetOutputNameAllocated(index, allocator));
    output_names.push_back(output_names_owned.back().get());
  }
  Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(
      memory, raw.data(), raw.size(), input_shape.data(), input_shape.size());
  const char *input_names[] = {input_name.get()};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                             output_names.data(), output_names.size());

  Output result;
  const auto read_float = [](const Ort::Value &value,
                             std::vector<int64_t> &shape) {
    const auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("postprocess tensor must be FP32");
    shape = info.GetShape();
    const float *data = value.GetTensorData<float>();
    return std::vector<float>{data, data + info.GetElementCount()};
  };
  result.boxes = read_float(outputs[0], result.boxes_shape);
  result.scores = read_float(outputs[1], result.scores_shape);
  const auto classes_info = outputs[2].GetTensorTypeAndShapeInfo();
  if (classes_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    throw std::runtime_error("postprocess classes must be INT64");
  result.classes_shape = classes_info.GetShape();
  const int64_t *classes = outputs[2].GetTensorData<int64_t>();
  result.classes.assign(classes, classes + classes_info.GetElementCount());
  return result;
}

float CompareFloat(const std::vector<float> &actual,
                   const std::vector<float> &expected) {
  if (actual.size() != expected.size())
    throw std::runtime_error("postprocess FP32 output size mismatch");
  float maximum = 0.0f;
  for (size_t index = 0; index < actual.size(); ++index)
    maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
  return maximum;
}

size_t CheckAssignment(const Ort::Session &session) {
  std::unordered_map<std::string, size_t> counts;
  for (const auto &subgraph : session.GetEpGraphAssignmentInfo()) {
    if (std::string(subgraph.GetEpName()) != kEpName)
      continue;
    for (const auto &node : subgraph.GetNodes())
      ++counts[node.GetOperatorType()];
  }
  for (const char *name : {"Split", "Transpose", "NonMaxSuppression",
                           "GatherND"}) {
    if (counts[name] == 0)
      throw std::runtime_error(std::string(name) +
                               " was not assigned to Dayan");
  }
  if (counts["GatherND"] != 2)
    throw std::runtime_error("unexpected Dayan GatherND assignment count");
  size_t total = 0;
  for (const auto &entry : counts)
    total += entry.second;
  return total;
}

void CheckFinalOutputs(const Output &output) {
  if (output.boxes_shape.size() != 2 || output.scores_shape.size() != 1 ||
      output.classes_shape.size() != 2)
    throw std::runtime_error("invalid postprocess output shape");
  const int64_t detections = output.boxes_shape[0];
  if (output.boxes_shape[1] != 4 || output.scores_shape[0] != detections ||
      output.classes_shape[0] != detections || output.classes_shape[1] != 1)
    throw std::runtime_error("final boxes, scores and classes do not match");
  if (!std::all_of(output.boxes.begin(), output.boxes.end(),
                   [](float value) { return std::isfinite(value); }) ||
      !std::all_of(output.scores.begin(), output.scores.end(),
                   [](float value) { return std::isfinite(value); }))
    throw std::runtime_error("postprocess produced a non-finite result");
}

void WriteOutputJson(const std::filesystem::path &path, const Output &cpu,
                     const Output &dayan) {
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path);
  if (!stream)
    throw std::runtime_error("cannot write " + path.string());
  stream << std::setprecision(9) << "{\n";
  const auto write = [&](const char *name, const Output &output, bool last) {
    stream << "  \"" << name << "\": [\n";
    for (size_t index = 0; index < output.classes.size(); ++index) {
      stream << "    {\"class_id\": " << output.classes[index]
             << ", \"score\": " << output.scores[index]
             << ", \"box\": [" << output.boxes[index * 4] << ", "
             << output.boxes[index * 4 + 1] << ", "
             << output.boxes[index * 4 + 2] << ", "
             << output.boxes[index * 4 + 3] << "]}";
      if (index + 1 != output.classes.size())
        stream << ',';
      stream << '\n';
    }
    stream << "  ]" << (last ? "\n" : ",\n");
  };
  write("cpu", cpu, false);
  write("dayan", dayan, true);
  stream << "}\n";
}

size_t ClassMismatches(const Output &actual, const Output &expected) {
  const size_t count = std::min(actual.classes.size(), expected.classes.size());
  size_t result = actual.classes.size() > expected.classes.size()
                      ? actual.classes.size() - expected.classes.size()
                      : expected.classes.size() - actual.classes.size();
  for (size_t index = 0; index < count; ++index)
    result += actual.classes[index] != expected.classes[index];
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 5)
      throw std::invalid_argument(
          "usage: test_dayan_yolo_postprocess MODEL PLUGIN ELF DATA_DIR "
          "[--smoke] [--results-json PATH]");
    const std::string model = argv[1];
    const std::string plugin = argv[2];
    const std::string kernel_elf = argv[3];
    const auto data_dir = std::filesystem::path(argv[4]);
    bool smoke = false;
    std::filesystem::path results_json;
    for (int index = 5; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--smoke") {
        smoke = true;
      } else if (argument == "--results-json" && index + 1 < argc) {
        results_json = argv[++index];
      } else {
        throw std::invalid_argument("unknown postprocess argument: " +
                                    argument);
      }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test-dayan-yolo-postprocess");
    Ort::SessionOptions cpu_options;
    cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session cpu_session(env, model.c_str(), cpu_options);
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = cpu_session.GetInputNameAllocated(0, allocator);
    std::vector<float> raw =
        ReadBinary<float>(data_dir / (std::string(input_name.get()) + ".bin"));
    const Output cpu = Run(cpu_session, raw);

    env.RegisterExecutionProviderLibrary(kRegistrationName, plugin);
    {
      std::vector<Ort::ConstEpDevice> devices;
      for (const auto &device : env.GetEpDevices())
        if (std::string(device.EpName()) == kEpName)
          devices.push_back(device);
      Ort::SessionOptions options;
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
      options.AddConfigEntry("session.record_ep_graph_assignment_info", "1");
      const std::unordered_map<std::string, std::string> ep_options{
          {"kernel_elf", kernel_elf}};
      options.AppendExecutionProvider_V2(env, devices, ep_options);
      Ort::Session session(env, model.c_str(), options);
      const size_t dayan_nodes = CheckAssignment(session);
      const Output dayan = Run(session, raw);
      CheckFinalOutputs(cpu);
      CheckFinalOutputs(dayan);
      if (!results_json.empty())
        WriteOutputJson(results_json, cpu, dayan);
      std::cout << "cpu_detections=" << cpu.classes.size()
                << " dayan_detections=" << dayan.classes.size() << std::endl;
      const bool same_sizes = dayan.boxes.size() == cpu.boxes.size() &&
                              dayan.scores.size() == cpu.scores.size();
      const float boxes_diff =
          same_sizes ? CompareFloat(dayan.boxes, cpu.boxes)
                     : std::numeric_limits<float>::quiet_NaN();
      const float scores_diff =
          same_sizes ? CompareFloat(dayan.scores, cpu.scores)
                     : std::numeric_limits<float>::quiet_NaN();
      const size_t class_mismatches = ClassMismatches(dayan, cpu);
      if (!smoke &&
          (!same_sizes || boxes_diff != 0.0f || scores_diff != 0.0f ||
           class_mismatches != 0)) {
        throw std::runtime_error("CPU/Dayan postprocess outputs mismatch");
      }
      std::cout << "dayan_nodes=" << dayan_nodes
                << " detections=" << dayan.classes.size()
                << " boxes_max_abs_diff=" << boxes_diff
                << " scores_max_abs_diff=" << scores_diff
                << " class_mismatches=" << class_mismatches << std::endl;
    }
    env.UnregisterExecutionProviderLibrary(kRegistrationName);
    std::cout << "PASS" << std::endl;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
