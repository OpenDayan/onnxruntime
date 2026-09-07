#include "runtime/device_context.h"
#include "runtime/tensor_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string kernel_elf;
  int m = 1;
  int n = 32;
  int k = 32;
  std::string a_file;
  std::string b_file;
  bool benchmark = false;
  int warmup = 1;
  int iterations = 5;
};

Options Parse(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--kernel-elf" && i + 1 < argc)
      options.kernel_elf = argv[++i];
    else if (argument == "--m" && i + 1 < argc)
      options.m = std::stoi(argv[++i]);
    else if (argument == "--n" && i + 1 < argc)
      options.n = std::stoi(argv[++i]);
    else if (argument == "--k" && i + 1 < argc)
      options.k = std::stoi(argv[++i]);
    else if (argument == "--a-file" && i + 1 < argc)
      options.a_file = argv[++i];
    else if (argument == "--b-file" && i + 1 < argc)
      options.b_file = argv[++i];
    else if (argument == "--benchmark")
      options.benchmark = true;
    else if (argument == "--warmup" && i + 1 < argc)
      options.warmup = std::stoi(argv[++i]);
    else if (argument == "--iterations" && i + 1 < argc)
      options.iterations = std::stoi(argv[++i]);
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (options.kernel_elf.empty())
    throw std::invalid_argument("--kernel-elf is required");
  if (options.m <= 0 || options.n <= 0 || options.k <= 0)
    throw std::invalid_argument("--m, --n and --k must be positive");
  if (options.warmup < 0)
    throw std::invalid_argument("--warmup must not be negative");
  if (options.iterations <= 0)
    throw std::invalid_argument("--iterations must be positive");
  return options;
}

float AValue(int m, int k) {
  return static_cast<float>(((m * 17 + k * 13) % 29) - 14) / 127.0f;
}

float BValue(int k, int n) {
  return static_cast<float>(((k * 7 + n * 19) % 31) - 15) / 113.0f;
}

std::vector<float> ReadFloats(const std::string &path, size_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open input data: " + path);
  const auto bytes = stream.tellg();
  if (bytes != static_cast<std::streamoff>(expected * sizeof(float)))
    throw std::runtime_error("unexpected input data size: " + path);
  std::vector<float> values(expected);
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char *>(values.data()), bytes))
    throw std::runtime_error("cannot read input data: " + path);
  return values;
}

std::vector<float> ReferenceMatMul(const std::vector<float> &a,
                                   const std::vector<float> &b,
                                   const rvnpu::MatMulShape &shape) {
  std::vector<float> reference(shape.y_elements, 0.0f);
  for (int m = 0; m < shape.m; ++m)
    for (int n = 0; n < shape.n; ++n)
      for (int k = 0; k < shape.k; ++k)
        reference[static_cast<size_t>(m) * shape.n + n] +=
            a[static_cast<size_t>(m) * shape.k + k] *
            b[static_cast<size_t>(k) * shape.n + n];
  return reference;
}

struct ValidationStats {
  float max_abs_diff = 0.0f;
  double output_l1 = 0.0;
};

ValidationStats ValidateOutput(const std::vector<float> &output,
                               const std::vector<float> &reference,
                               const std::string &label) {
  ValidationStats stats;
  bool output_has_nonzero = false;
  bool reference_has_nonzero = false;
  for (size_t i = 0; i < output.size(); ++i) {
    if (!std::isfinite(output[i]))
      throw std::runtime_error(label + ": non-finite output at element " +
                               std::to_string(i));
    output_has_nonzero = output_has_nonzero || output[i] != 0.0f;
    reference_has_nonzero = reference_has_nonzero || reference[i] != 0.0f;
    stats.output_l1 += std::abs(static_cast<double>(output[i]));
  }
  if (reference_has_nonzero && !output_has_nonzero) {
    throw std::runtime_error(
        label + ": all-zero output; possible MAT no-op/protocol mismatch");
  }
  for (size_t i = 0; i < output.size(); ++i) {
    const float difference = std::abs(output[i] - reference[i]);
    stats.max_abs_diff = std::max(stats.max_abs_diff, difference);
    const float tolerance = 2e-2f + 2e-2f * std::abs(reference[i]);
    if (difference > tolerance) {
      throw std::runtime_error(label + ": numeric mismatch at element " +
                               std::to_string(i));
    }
  }
  return stats;
}

double Median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t middle = samples.size() / 2;
  if (samples.size() % 2 != 0)
    return samples[middle];
  return (samples[middle - 1] + samples[middle]) / 2.0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = Parse(argc, argv);
    const auto shape = rvnpu::ValidateMatMulShapes({options.m, options.k},
                                                   {options.k, options.n});
    std::vector<float> a(shape.a_elements);
    std::vector<float> b(shape.b_elements);
    std::vector<float> output(shape.y_elements, 0.0f);

    if (!options.a_file.empty()) {
      a = ReadFloats(options.a_file, shape.a_elements);
    } else {
      for (int m = 0; m < shape.m; ++m)
        for (int k = 0; k < shape.k; ++k)
          a[static_cast<size_t>(m) * shape.k + k] = AValue(m, k);
    }
    if (!options.b_file.empty()) {
      b = ReadFloats(options.b_file, shape.b_elements);
    } else {
      for (int k = 0; k < shape.k; ++k)
        for (int n = 0; n < shape.n; ++n)
          b[static_cast<size_t>(k) * shape.n + n] = BValue(k, n);
    }
    const auto reference = ReferenceMatMul(a, b, shape);

    using Clock = std::chrono::steady_clock;
    const auto setup_begin = Clock::now();
    rvnpu::DeviceContext context(options.kernel_elf);
    const double setup_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - setup_begin)
            .count();

    if (!options.benchmark) {
      std::fill(output.begin(), output.end(),
                std::numeric_limits<float>::quiet_NaN());
      context.RunMatMul(a.data(), b.data(), output.data(), shape);
      const auto stats = ValidateOutput(output, reference, "MatMul");
      std::cout << "PASS M=" << shape.m << " N=" << shape.n
                << " K=" << shape.k
                << " max_abs_diff=" << stats.max_abs_diff << std::endl;
      return 0;
    }

    std::vector<float> alternate_a = a;
    for (float &value : alternate_a)
      value = -value;
    std::vector<float> alternate_reference = reference;
    for (float &value : alternate_reference)
      value = -value;

    float max_abs_diff = 0.0f;
    double output_l1 = 0.0;
    auto run_checked = [&](const std::vector<float> &input,
                           const std::vector<float> &expected,
                           const std::string &label) {
      std::fill(output.begin(), output.end(),
                std::numeric_limits<float>::quiet_NaN());
      const auto begin = Clock::now();
      context.RunMatMul(input.data(), b.data(), output.data(), shape);
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - begin)
              .count();
      const auto stats = ValidateOutput(output, expected, label);
      max_abs_diff = std::max(max_abs_diff, stats.max_abs_diff);
      output_l1 = stats.output_l1;
      return elapsed_ms;
    };

    static_cast<void>(run_checked(a, reference, "smoke-A"));
    static_cast<void>(
        run_checked(alternate_a, alternate_reference, "smoke-B"));
    for (int i = 0; i < options.warmup; ++i) {
      const bool alternate = i % 2 != 0;
      static_cast<void>(run_checked(alternate ? alternate_a : a,
                                    alternate ? alternate_reference : reference,
                                    "warmup-" + std::to_string(i)));
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(options.iterations));
    for (int i = 0; i < options.iterations; ++i) {
      const bool alternate = i % 2 != 0;
      samples_ms.push_back(run_checked(
          alternate ? alternate_a : a,
          alternate ? alternate_reference : reference,
          "iteration-" + std::to_string(i)));
    }

    const double total_ms =
        std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    const double median_ms = Median(samples_ms);
    const double operations = 2.0 * static_cast<double>(shape.m) *
                              static_cast<double>(shape.n) *
                              static_cast<double>(shape.k);
    const double effective_gflops =
        median_ms > 0.0 ? operations / (median_ms * 1.0e6) : 0.0;

    std::cout << std::setprecision(9) << "PASS smoke M=" << shape.m
              << " N=" << shape.n << " K=" << shape.k
              << " max_abs_diff=" << max_abs_diff
              << " output_l1=" << output_l1 << '\n';
    std::cout << std::fixed << std::setprecision(3)
              << "MatMul 性能结果（模拟器端到端）\n"
              << "  初始化耗时: " << setup_ms << " ms\n"
              << "  预热次数: " << options.warmup << "\n"
              << "  计时次数: " << options.iterations << "\n"
              << "  计时总耗时: " << total_ms << " ms\n"
              << "  有效吞吐率: " << effective_gflops << " GFLOPS\n";
    std::cout << std::endl;
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "FAIL: " << ex.what() << std::endl;
    return 1;
  }
}
