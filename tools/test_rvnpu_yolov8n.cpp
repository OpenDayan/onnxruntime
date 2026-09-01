#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
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
  std::filesystem::path data_dir;
  std::filesystem::path results_json;
  bool cpu_only = false;
  bool assignment_only = false;
  bool smoke = false;
};

Options Parse(int argc, char **argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--model" && index + 1 < argc)
      result.model = argv[++index];
    else if (argument == "--plugin" && index + 1 < argc)
      result.plugin = argv[++index];
    else if (argument == "--kernel-elf" && index + 1 < argc)
      result.kernel_elf = argv[++index];
    else if (argument == "--data-dir" && index + 1 < argc)
      result.data_dir = argv[++index];
    else if (argument == "--results-json" && index + 1 < argc)
      result.results_json = argv[++index];
    else if (argument == "--cpu-only")
      result.cpu_only = true;
    else if (argument == "--assignment-only")
      result.assignment_only = true;
    else if (argument == "--smoke")
      result.smoke = true;
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (result.model.empty() ||
      (!result.assignment_only && result.data_dir.empty()))
    throw std::invalid_argument(
        "--model and --data-dir are required unless --assignment-only is used");
  if (result.cpu_only && result.assignment_only)
    throw std::invalid_argument("CPU-only and assignment-only are exclusive");
  if (!result.cpu_only && (result.plugin.empty() || result.kernel_elf.empty()))
    throw std::invalid_argument(
        "--plugin and --kernel-elf are required for RVNPU execution");
  return result;
}

std::vector<float> ReadFloats(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open " + path.string());
  const auto bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
    throw std::runtime_error("invalid FP32 input " + path.string());
  std::vector<float> result(static_cast<size_t>(bytes) / sizeof(float));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char *>(result.data()), bytes))
    throw std::runtime_error("cannot read " + path.string());
  return result;
}

size_t Elements(const std::vector<int64_t> &shape) {
  size_t result = 1;
  for (const int64_t dimension : shape) {
    if (dimension <= 0)
      throw std::runtime_error("YOLOv8n input must have a static shape");
    result *= static_cast<size_t>(dimension);
  }
  return result;
}

struct Detection {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int64_t class_id;
};

std::vector<Detection> Run(Ort::Session &session, std::vector<float> &input) {
  if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
    throw std::runtime_error("official YOLOv8n must have one input and output");
  Ort::AllocatorWithDefaultOptions allocator;
  const auto input_name = session.GetInputNameAllocated(0, allocator);
  const auto input_type = session.GetInputTypeInfo(0);
  const auto input_info = input_type.GetTensorTypeAndShapeInfo();
  const auto input_shape = input_info.GetShape();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      input_shape != std::vector<int64_t>{1, 3, 640, 640} ||
      Elements(input_shape) != input.size()) {
    std::string shape_text;
    for (const int64_t dimension : input_shape)
      shape_text += (shape_text.empty() ? "" : ",") +
                    std::to_string(dimension);
    throw std::runtime_error(
        "official YOLOv8n input must be FP32 [1,3,640,640], got type=" +
        std::to_string(input_info.GetElementType()) + " shape=[" + shape_text +
        "] elements=" + std::to_string(input.size()));
  }

  Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_value = Ort::Value::CreateTensor<float>(
      memory, input.data(), input.size(), input_shape.data(), input_shape.size());
  const auto output_name = session.GetOutputNameAllocated(0, allocator);
  const char *input_names[] = {input_name.get()};
  const char *output_names[] = {output_name.get()};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_value,
                             1, output_names, 1);
  const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
  if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      output_info.GetShape() != std::vector<int64_t>{1, 300, 6}) {
    throw std::runtime_error("official YOLOv8n output must be FP32 [1,300,6]");
  }

  const float *data = outputs[0].GetTensorData<float>();
  std::vector<Detection> result;
  for (size_t row = 0; row < 300; ++row) {
    const float *values = data + row * 6;
    if (!std::all_of(values, values + 6,
                     [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error("official YOLOv8n produced a non-finite value");
    }
    if (values[4] <= 0.0f)
      continue;
    const float rounded_class = std::round(values[5]);
    if (std::abs(values[5] - rounded_class) > 1e-4f || rounded_class < 0.0f ||
        rounded_class > 79.0f) {
      throw std::runtime_error("official YOLOv8n produced an invalid class");
    }
    if (values[2] < values[0] || values[3] < values[1])
      throw std::runtime_error("official YOLOv8n produced an invalid xyxy box");
    result.push_back({values[0], values[1], values[2], values[3], values[4],
                      static_cast<int64_t>(rounded_class)});
  }
  return result;
}

float IoU(const Detection &left, const Detection &right) {
  const float width =
      std::max(0.0f, std::min(left.x2, right.x2) - std::max(left.x1, right.x1));
  const float height =
      std::max(0.0f, std::min(left.y2, right.y2) - std::max(left.y1, right.y1));
  const float intersection = width * height;
  const float left_area =
      std::max(0.0f, left.x2 - left.x1) * std::max(0.0f, left.y2 - left.y1);
  const float right_area = std::max(0.0f, right.x2 - right.x1) *
                           std::max(0.0f, right.y2 - right.y1);
  const float union_area = left_area + right_area - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

struct Comparison {
  size_t matched = 0;
  float mean_iou = 0.0f;
  float min_iou = 1.0f;
  float max_score_diff = 0.0f;
};

Comparison Compare(const std::vector<Detection> &cpu,
                   const std::vector<Detection> &rvnpu) {
  std::vector<bool> used(rvnpu.size(), false);
  Comparison result;
  float iou_sum = 0.0f;
  for (const auto &expected : cpu) {
    float best_iou = -1.0f;
    size_t best_index = rvnpu.size();
    for (size_t index = 0; index < rvnpu.size(); ++index) {
      if (used[index] || rvnpu[index].class_id != expected.class_id)
        continue;
      const float iou = IoU(expected, rvnpu[index]);
      if (iou > best_iou) {
        best_iou = iou;
        best_index = index;
      }
    }
    if (best_index == rvnpu.size() || best_iou < 0.5f)
      continue;
    used[best_index] = true;
    ++result.matched;
    iou_sum += best_iou;
    result.min_iou = std::min(result.min_iou, best_iou);
    result.max_score_diff =
        std::max(result.max_score_diff,
                 std::abs(expected.score - rvnpu[best_index].score));
  }
  if (result.matched != 0)
    result.mean_iou = iou_sum / static_cast<float>(result.matched);
  return result;
}

void WriteDetections(std::ostream &stream,
                     const std::vector<Detection> &detections) {
  stream << "[\n";
  for (size_t index = 0; index < detections.size(); ++index) {
    const auto &item = detections[index];
    stream << "    {\"class_id\": " << item.class_id
           << ", \"score\": " << item.score << ", \"box\": [" << item.x1
           << ", " << item.y1 << ", " << item.x2 << ", " << item.y2
           << "]}";
    if (index + 1 != detections.size())
      stream << ',';
    stream << '\n';
  }
  stream << "  ]";
}

void WriteJson(const std::filesystem::path &path,
               const std::vector<Detection> &cpu,
               const std::optional<std::vector<Detection>> &rvnpu) {
  if (path.empty())
    return;
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path);
  if (!stream)
    throw std::runtime_error("cannot write " + path.string());
  stream << std::setprecision(9)
         << "{\n  \"box_format\": \"xyxy\",\n  \"cpu\": ";
  WriteDetections(stream, cpu);
  if (rvnpu) {
    stream << ",\n  \"rvnpu\": ";
    WriteDetections(stream, *rvnpu);
  }
  stream << "\n}\n";
}

size_t CheckAssignment(const Ort::Session &session) {
  std::unordered_map<std::string, size_t> counts;
  size_t silu_fusions = 0;
  size_t rvnpu_execution_units = 0;
  for (const auto &subgraph : session.GetEpGraphAssignmentInfo()) {
    if (std::string(subgraph.GetEpName()) != kEpName)
      continue;
    ++rvnpu_execution_units;
    const auto nodes = subgraph.GetNodes();
    bool has_sigmoid = false;
    bool has_mul = false;
    for (const auto &node : nodes) {
      ++counts[node.GetOperatorType()];
      has_sigmoid = has_sigmoid || node.GetOperatorType() == "Sigmoid";
      has_mul = has_mul || node.GetOperatorType() == "Mul";
    }
    if (nodes.size() == 2 && has_sigmoid && has_mul)
      ++silu_fusions;
  }
  if (counts["Conv"] != 64) {
    throw std::runtime_error(
        "official YOLOv8n must assign 64 Conv nodes to RVNPU");
  }
  std::vector<std::pair<std::string, size_t>> sorted(counts.begin(), counts.end());
  std::sort(sorted.begin(), sorted.end());
  size_t total = 0;
  std::cout << "assignment";
  for (const auto &[name, count] : sorted) {
    total += count;
    std::cout << ' ' << name << '=' << count;
  }
  std::cout << " SiluFusion=" << silu_fusions
            << " ExecutionUnits=" << rvnpu_execution_units << std::endl;
  return total;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test-rvnpu-yolov8n");
    if (options.assignment_only) {
      env.RegisterExecutionProviderLibrary(kRegistrationName, options.plugin);
      std::vector<Ort::ConstEpDevice> devices;
      for (const auto &device : env.GetEpDevices())
        if (std::string(device.EpName()) == kEpName)
          devices.push_back(device);
      if (devices.size() != 1)
        throw std::runtime_error("expected exactly one RVNPU EP device");
      Ort::SessionOptions session_options;
      session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
      session_options.AddConfigEntry("session.record_ep_graph_assignment_info",
                                     "1");
      const std::unordered_map<std::string, std::string> ep_options{
          {"kernel_elf", options.kernel_elf}};
      session_options.AppendExecutionProvider_V2(env, devices, ep_options);
      Ort::Session session(env, options.model.c_str(), session_options);
      const size_t rvnpu_nodes = CheckAssignment(session);
      env.UnregisterExecutionProviderLibrary(kRegistrationName);
      std::cout << "rvnpu_nodes=" << rvnpu_nodes << std::endl;
      std::cout << "PASS" << std::endl;
      return 0;
    }
    Ort::SessionOptions cpu_options;
    cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session cpu_session(env, options.model.c_str(), cpu_options);
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = cpu_session.GetInputNameAllocated(0, allocator);
    std::vector<float> input =
        ReadFloats(options.data_dir / (std::string(input_name.get()) + ".bin"));
    const auto cpu = Run(cpu_session, input);
    if (options.cpu_only) {
      WriteJson(options.results_json, cpu, std::nullopt);
      std::cout << "cpu_detections=" << cpu.size() << std::endl;
      std::cout << "PASS" << std::endl;
      return 0;
    }

    env.RegisterExecutionProviderLibrary(kRegistrationName, options.plugin);
    std::vector<Detection> rvnpu;
    size_t rvnpu_nodes = 0;
    {
      std::vector<Ort::ConstEpDevice> devices;
      for (const auto &device : env.GetEpDevices())
        if (std::string(device.EpName()) == kEpName)
          devices.push_back(device);
      if (devices.size() != 1)
        throw std::runtime_error("expected exactly one RVNPU EP device");
      Ort::SessionOptions options_with_ep;
      options_with_ep.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
      options_with_ep.AddConfigEntry("session.record_ep_graph_assignment_info",
                                     "1");
      const std::unordered_map<std::string, std::string> ep_options{
          {"kernel_elf", options.kernel_elf}};
      options_with_ep.AppendExecutionProvider_V2(env, devices, ep_options);
      Ort::Session session(env, options.model.c_str(), options_with_ep);
      rvnpu_nodes = CheckAssignment(session);
      rvnpu = Run(session, input);
    }
    env.UnregisterExecutionProviderLibrary(kRegistrationName);

    const Comparison comparison = Compare(cpu, rvnpu);
    WriteJson(options.results_json, cpu, rvnpu);
    std::cout << "cpu_detections=" << cpu.size()
              << " rvnpu_detections=" << rvnpu.size()
              << " matched=" << comparison.matched
              << " mean_iou=" << comparison.mean_iou
              << " min_iou=" << comparison.min_iou
              << " max_score_abs_diff=" << comparison.max_score_diff
              << " rvnpu_nodes=" << rvnpu_nodes << std::endl;
    if (!options.smoke &&
        (cpu.size() != rvnpu.size() || comparison.matched != cpu.size() ||
         comparison.min_iou < 0.99f || comparison.max_score_diff > 1e-3f)) {
      throw std::runtime_error("CPU/RVNPU official YOLOv8n outputs mismatch");
    }
    std::cout << "PASS" << std::endl;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
