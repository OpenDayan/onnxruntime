#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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
  int repeat = 2;
  bool compare_cpu = false;
};

Options Parse(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--model" && i + 1 < argc)
      options.model = argv[++i];
    else if (argument == "--plugin" && i + 1 < argc)
      options.plugin = argv[++i];
    else if (argument == "--kernel-elf" && i + 1 < argc)
      options.kernel_elf = argv[++i];
    else if (argument == "--data-dir" && i + 1 < argc)
      options.data_dir = argv[++i];
    else if (argument == "--repeat" && i + 1 < argc)
      options.repeat = std::stoi(argv[++i]);
    else if (argument == "--compare-cpu")
      options.compare_cpu = true;
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (options.model.empty() || options.plugin.empty() ||
      options.kernel_elf.empty() || options.data_dir.empty()) {
    throw std::invalid_argument(
        "--model, --plugin, --kernel-elf and --data-dir are required");
  }
  if (options.repeat < 1)
    throw std::invalid_argument("--repeat must be positive");
  return options;
}

std::vector<float> ReadFloats(const std::filesystem::path &path,
                              size_t expected = 0) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open data file: " + path.string());
  const auto bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
    throw std::runtime_error("invalid float data file: " + path.string());
  }
  std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char *>(values.data()), bytes)) {
    throw std::runtime_error("cannot read data file: " + path.string());
  }
  if (expected != 0 && values.size() != expected) {
    throw std::runtime_error("unexpected element count in " + path.string());
  }
  return values;
}

struct Outputs {
  std::vector<float> output;
  std::vector<float> flat;
};

Outputs Run(Ort::Session &session, std::vector<float> &input) {
  const std::array<int64_t, 4> input_shape{1, 1, 10, 10};
  Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_value =
      Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                      input_shape.data(), input_shape.size());
  const std::array<const char *, 1> input_names{"input"};
  const std::array<const char *, 2> output_names{"output", "flat_out"};
  auto values =
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_value, 1,
                  output_names.data(), output_names.size());
  if (values.size() != 2)
    throw std::runtime_error("session returned unexpected output count");

  Outputs result;
  for (size_t index = 0; index < values.size(); ++index) {
    const auto info = values[index].GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    const float *data = values[index].GetTensorData<float>();
    auto &destination = index == 0 ? result.output : result.flat;
    destination.assign(data, data + count);
  }
  return result;
}

float Compare(const std::vector<float> &actual,
              const std::vector<float> &expected, float absolute,
              float relative, const std::string &label) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(label + " output size mismatch");
  }
  float maximum = 0.0f;
  for (size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]))
      throw std::runtime_error(label + " has non-finite output");
    const float difference = std::abs(actual[index] - expected[index]);
    maximum = std::max(maximum, difference);
    if (difference > absolute + relative * std::abs(expected[index])) {
      throw std::runtime_error(label + " mismatch at element " +
                               std::to_string(index) +
                               " got=" + std::to_string(actual[index]) +
                               " expected=" + std::to_string(expected[index]) +
                               " diff=" + std::to_string(difference));
    }
  }
  return maximum;
}

void CheckAssignment(const Ort::Session &session) {
  bool saw_rvnpu_matmul = false;
  for (const auto &subgraph : session.GetEpGraphAssignmentInfo()) {
    const std::string ep_name = subgraph.GetEpName();
    for (const auto &node : subgraph.GetNodes()) {
      const std::string op = node.GetOperatorType();
      std::cout << "assignment " << op << " -> " << ep_name << std::endl;
      if (op == "MatMul") {
        if (ep_name != kEpName)
          throw std::runtime_error("MatMul was not assigned to RVNPU EP");
        saw_rvnpu_matmul = true;
      } else if (ep_name == kEpName) {
        throw std::runtime_error("RVNPU EP claimed unsupported operator " + op);
      }
    }
  }
  if (!saw_rvnpu_matmul)
    throw std::runtime_error("assignment info has no RVNPU MatMul");
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = Parse(argc, argv);
    std::vector<float> input =
        ReadFloats(std::filesystem::path(options.data_dir) / "input.bin", 100);
    const auto reference = ReadFloats(
        std::filesystem::path(options.data_dir) / "reference_output.bin", 32);
    const auto reference_flat = ReadFloats(
        std::filesystem::path(options.data_dir) / "reference_flat.bin", 32);

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "rvnpu-ep-runner");
    Outputs cpu;
    if (options.compare_cpu) {
      Ort::SessionOptions cpu_options;
      cpu_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
      Ort::Session cpu_session(env, options.model.c_str(), cpu_options);
      cpu = Run(cpu_session, input);
      std::cout << "CPU vs ReferenceEvaluator max_abs_diff="
                << Compare(cpu.output, reference, 1e-6f, 0.0f, "CPU/reference")
                << std::endl;
      Compare(cpu.flat, reference_flat, 1e-6f, 0.0f, "CPU/reference flat_out");
    }

    env.RegisterExecutionProviderLibrary(kRegistrationName, options.plugin);
    std::vector<Ort::ConstEpDevice> selected;
    for (const auto &device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == kEpName)
        selected.push_back(device);
    }
    if (selected.size() != 1) {
      throw std::runtime_error("expected exactly one RVNPU EP device, found " +
                               std::to_string(selected.size()));
    }

    Outputs previous;
    {
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

      for (int iteration = 0; iteration < options.repeat; ++iteration) {
        Outputs current = Run(session, input);
        const auto &golden = options.compare_cpu ? cpu.output : reference;
        const auto &flat_golden =
            options.compare_cpu ? cpu.flat : reference_flat;
        std::cout << "RVNPU vs golden iteration=" << iteration
                  << " max_abs_diff="
                  << Compare(current.output, golden, 1e-4f, 1e-4f,
                             "RVNPU/golden")
                  << std::endl;
        Compare(current.flat, flat_golden, 1e-6f, 0.0f, "RVNPU flat_out");
        if (iteration > 0) {
          Compare(current.output, previous.output, 0.0f, 0.0f,
                  "repeat determinism");
        }
        previous = std::move(current);
      }
    }
    env.UnregisterExecutionProviderLibrary(kRegistrationName);
    std::cout << "PASS" << std::endl;
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "FAIL: " << ex.what() << std::endl;
    return 1;
  }
}
