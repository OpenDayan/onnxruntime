#include "runtime/device_context.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/device_buffer.h"
#include "runtime/kernel_catalog.h"

namespace rvnpu {

namespace {
void RequireSuccess(ssize_t result, const std::string &stage) {
  if (result != NPU_SUCCESS) {
    throw std::runtime_error(stage +
                             " failed: runtime code=" + std::to_string(result));
  }
}

std::vector<uint8_t> ReadFile(const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open kernel ELF: " + path);
  }
  const auto end = stream.tellg();
  if (end <= 0) {
    throw std::runtime_error("kernel ELF is empty: " + path);
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char *>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("cannot read kernel ELF: " + path);
  }
  return bytes;
}

} // namespace

DeviceContext::DeviceContext(std::string kernel_elf)
    : kernel_elf_(std::move(kernel_elf)) {
  if (kernel_elf_.empty() || !std::filesystem::is_regular_file(kernel_elf_)) {
    throw std::invalid_argument(
        "kernel_elf must name an existing regular file: " + kernel_elf_);
  }

  RequireSuccess(npu_runtime_init(&runtime_), "npu_runtime_init");
  initialized_ = true;
  try {
    const auto image = ReadFile(kernel_elf_);
    RequireSuccess(npu_runtime_load_kernelset_from_memory(
                       &runtime_, image.data(), image.size()),
                   "npu_runtime_load_kernelset_from_memory");

    NPUKernelDesc *tx_init = RequireKernel(kernels::kTxInit);
    int kernel_result = -1;
    RequireSuccess(
        npu_runtime_kernel_start(&runtime_, tx_init, &kernel_result, 0),
        "__tx_init launch");
    if (kernel_result != 0) {
      throw std::runtime_error("__tx_init returned " +
                               std::to_string(kernel_result));
    }
    static_cast<void>(RequireKernel(kernels::kMatMulFp32));
  } catch (...) {
    npu_runtime_deinit(&runtime_);
    initialized_ = false;
    throw;
  }
}

DeviceContext::~DeviceContext() {
  if (initialized_) {
    npu_runtime_deinit(&runtime_);
  }
}

NPUKernelDesc *DeviceContext::RequireKernel(const char *name) {
  const auto found = kernels_.find(name);
  if (found != kernels_.end()) {
    return found->second;
  }
  NPUKernelDesc *descriptor = nullptr;
  RequireSuccess(npu_runtime_get_kernel_desc(&runtime_, name, &descriptor),
                 std::string("kernel lookup ") + name);
  if (descriptor == nullptr || descriptor->entry == 0) {
    throw std::runtime_error(std::string("kernel has no entry: ") + name);
  }
  kernels_.emplace(name, descriptor);
  return descriptor;
}

void DeviceContext::RunMatMul(const float *a, const float *b, float *y,
                              const MatMulShape &shape) {
  if (a == nullptr || b == nullptr || y == nullptr) {
    throw std::invalid_argument("RVNPU MatMul received a null tensor pointer");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const size_t a_bytes = shape.a_elements * sizeof(float);
  const size_t b_bytes = shape.b_elements * sizeof(float);
  const size_t y_bytes = shape.y_elements * sizeof(float);
  DeviceBuffer device_a(runtime_, a_bytes);
  DeviceBuffer device_b(runtime_, b_bytes);
  DeviceBuffer device_y(runtime_, y_bytes);
  device_a.CopyFromHost(a, a_bytes);
  device_b.CopyFromHost(b, b_bytes);

  const std::string details =
      " M=" + std::to_string(shape.m) + " N=" + std::to_string(shape.n) +
      " K=" + std::to_string(shape.k);
  LaunchKernelLocked("RVNPU MatMul", kernels::kMatMulFp32, details,
                     device_a.address(), device_b.address(),
                     device_y.address(), shape.m, shape.n, shape.k, shape.k,
                     1, shape.n, 1, shape.n, 1);
  device_y.CopyToHost(y, y_bytes);

}

void DeviceContext::RunBinaryElementwise(
    const char *operator_name, const char *kernel_name, const float *lhs,
    const float *rhs, float *output, const BinaryElementwiseShape &shape) {
  if (lhs == nullptr || rhs == nullptr || output == nullptr) {
    throw std::invalid_argument(std::string(operator_name) +
                                " received a null tensor pointer");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const float *device_a_source = shape.swap_inputs ? rhs : lhs;
  const float *device_b_source = shape.swap_inputs ? lhs : rhs;
  const size_t a_elements = shape.output_elements;
  const size_t b_elements =
      shape.swap_inputs ? shape.lhs_elements : shape.rhs_elements;
  const size_t a_bytes = a_elements * sizeof(float);
  const size_t b_bytes = b_elements * sizeof(float);
  const size_t output_bytes = shape.output_elements * sizeof(float);

  DeviceBuffer device_a(runtime_, a_bytes);
  DeviceBuffer device_b(runtime_, b_bytes);
  DeviceBuffer device_output(runtime_, output_bytes);
  device_a.CopyFromHost(device_a_source, a_bytes);
  device_b.CopyFromHost(device_b_source, b_bytes);

  const auto &a = shape.output_dims;
  const auto &b = shape.broadcast_dims;
  const std::string details =
      " elements=" + std::to_string(shape.output_elements);
  LaunchKernelLocked(operator_name, kernel_name, details, device_a.address(),
                     device_b.address(), device_output.address(), a[0], a[1],
                     a[2], a[3], b[0], b[1], b[2], b[3]);
  device_output.CopyToHost(output, output_bytes);

}

void DeviceContext::RunAdd(const float *lhs, const float *rhs, float *output,
                           const BinaryElementwiseShape &shape) {
  RunBinaryElementwise("RVNPU Add", kernels::kAddFp32, lhs, rhs, output,
                       shape);
}

void DeviceContext::RunMul(const float *lhs, const float *rhs, float *output,
                           const BinaryElementwiseShape &shape) {
  RunBinaryElementwise("RVNPU Mul", kernels::kMulFp32, lhs, rhs, output,
                       shape);
}

void DeviceContext::RunSigmoid(const float *input, float *output,
                               const UnaryTensorShape &shape) {
  if (input == nullptr || output == nullptr) {
    throw std::invalid_argument("RVNPU Sigmoid received a null tensor pointer");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const size_t bytes = shape.elements * sizeof(float);
  DeviceBuffer device_input(runtime_, bytes);
  DeviceBuffer device_output(runtime_, bytes);
  device_input.CopyFromHost(input, bytes);
  const auto &dims = shape.dims;
  const std::string details = " elements=" + std::to_string(shape.elements);
  LaunchKernelLocked("RVNPU Sigmoid", kernels::kSigmoidFp32, details,
                     device_input.address(), device_output.address(), dims[3],
                     dims[2], dims[1], dims[0]);
  device_output.CopyToHost(output, bytes);

}

void DeviceContext::RunConv(const float *input, const float *weight,
                            const float *bias, float *output,
                            const ConvShape &shape) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      (shape.has_bias && bias == nullptr)) {
    throw std::invalid_argument("RVNPU Conv received a null tensor pointer");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const size_t input_bytes = shape.input_elements * sizeof(float);
  const size_t weight_bytes = shape.weight_elements * sizeof(float);
  const size_t column_bytes = shape.column_elements * sizeof(float);
  const size_t output_bytes = shape.output_elements * sizeof(float);
  std::vector<float> transposed_weight(shape.weight_elements);
  for (int32_t output_channel = 0;
       output_channel < shape.output_channels; ++output_channel) {
    for (int32_t reduction = 0; reduction < shape.matmul_k; ++reduction) {
      transposed_weight[static_cast<size_t>(reduction) *
                            shape.output_channels +
                        output_channel] =
          weight[static_cast<size_t>(output_channel) * shape.matmul_k +
                 reduction];
    }
  }
  DeviceBuffer device_input(runtime_, input_bytes);
  DeviceBuffer device_weight(runtime_, weight_bytes);
  DeviceBuffer device_columns(runtime_, column_bytes);
  DeviceBuffer device_output(runtime_, output_bytes);
  device_input.CopyFromHost(input, input_bytes);
  device_weight.CopyFromHost(transposed_weight.data(), weight_bytes);

  const std::string details =
      " N=1 C=" + std::to_string(shape.input_channels) +
      " H=" + std::to_string(shape.input_h) +
      " W=" + std::to_string(shape.input_w) +
      " M=" + std::to_string(shape.output_channels) +
      " KH=" + std::to_string(shape.kernel_h) +
      " KW=" + std::to_string(shape.kernel_w);
  LaunchKernelLocked(
      "RVNPU Conv im2col", kernels::kIm2ColNchwFp32, details,
      device_input.address(), device_columns.address(), shape.input_channels,
      shape.input_h, shape.input_w, shape.output_h, shape.output_w,
      shape.kernel_h, shape.kernel_w, shape.stride_h, shape.stride_w,
      shape.pad_h, shape.pad_w);
  LaunchKernelLocked(
      "RVNPU Conv MatMul", kernels::kMatMulFp32, details,
      device_columns.address(), device_weight.address(),
      device_output.address(), shape.matmul_m, shape.matmul_n,
      shape.matmul_k, shape.matmul_k, 1, shape.matmul_n, 1, 1,
      shape.matmul_m);

  if (shape.has_bias) {
    const size_t bias_bytes = shape.bias_elements * sizeof(float);
    DeviceBuffer device_bias(runtime_, bias_bytes);
    device_bias.CopyFromHost(bias, bias_bytes);
    LaunchKernelLocked(
        "RVNPU Conv bias", kernels::kAddFp32, details,
        device_output.address(), device_bias.address(),
        device_output.address(), shape.output_w, shape.output_h,
        shape.output_channels, 1, 1, 1, shape.output_channels, 1);
  }

  device_output.CopyToHost(output, output_bytes);
}

void DeviceContext::RunMaxPool(const float *input, float *output,
                               const MaxPoolShape &shape) {
  if (input == nullptr || output == nullptr) {
    throw std::invalid_argument(
        "RVNPU MaxPool received a null tensor pointer");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t input_bytes = shape.input_elements * sizeof(float);
  const size_t output_bytes = shape.output_elements * sizeof(float);
  DeviceBuffer device_input(runtime_, input_bytes);
  DeviceBuffer device_output(runtime_, output_bytes);
  device_input.CopyFromHost(input, input_bytes);

  const std::string details =
      " N=" + std::to_string(shape.batch) +
      " C=" + std::to_string(shape.channels) +
      " H=" + std::to_string(shape.input_h) +
      " W=" + std::to_string(shape.input_w) +
      " KH=" + std::to_string(shape.kernel_h) +
      " KW=" + std::to_string(shape.kernel_w);
  LaunchKernelLocked(
      "RVNPU MaxPool", kernels::kMaxPool2dNchwFp32, details,
      device_input.address(), device_output.address(), shape.batch,
      shape.channels, shape.input_h, shape.input_w, shape.output_h,
      shape.output_w, shape.kernel_h, shape.kernel_w, shape.stride_h,
      shape.stride_w, shape.pad_h, shape.pad_w);
  device_output.CopyToHost(output, output_bytes);

}

void DeviceContext::RunTopK(const float *input, float *values,
                            int64_t *indices, const TopKShape &shape) {
  if (input == nullptr || values == nullptr || indices == nullptr) {
    throw std::invalid_argument("RVNPU TopK received a null tensor pointer");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t input_bytes = shape.input_elements * sizeof(float);
  const size_t values_bytes = shape.output_elements * sizeof(float);
  const size_t indices_bytes = shape.output_elements * sizeof(int64_t);
  DeviceBuffer device_input(runtime_, input_bytes);
  DeviceBuffer device_values(runtime_, values_bytes);
  DeviceBuffer device_indices(runtime_, indices_bytes);
  device_input.CopyFromHost(input, input_bytes);

  const std::string details =
      " rows=" + std::to_string(shape.rows) +
      " columns=" + std::to_string(shape.columns) +
      " K=" + std::to_string(shape.k);
  LaunchKernelLocked("RVNPU TopK", kernels::kTopKLastDimFp32, details,
                     device_input.address(), device_values.address(),
                     device_indices.address(), shape.rows, shape.columns,
                     shape.k);
  device_values.CopyToHost(values, values_bytes);
  device_indices.CopyToHost(indices, indices_bytes);
}

std::vector<int64_t>
DeviceContext::RunNms(const float *boxes, const float *scores,
                      float iou_threshold, float score_threshold,
                      const NmsShape &shape) {
  if (boxes == nullptr || scores == nullptr) {
    throw std::invalid_argument("RVNPU NMS received a null tensor pointer");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t boxes_bytes = shape.boxes_elements * sizeof(float);
  const size_t scores_bytes = shape.scores_elements * sizeof(float);
  const size_t output_elements = shape.max_selected * 3;
  const size_t output_bytes = output_elements * sizeof(int64_t);
  DeviceBuffer device_boxes(runtime_, boxes_bytes);
  DeviceBuffer device_scores(runtime_, scores_bytes);
  DeviceBuffer device_output(runtime_, output_bytes);
  DeviceBuffer device_count(runtime_, sizeof(uint64_t));
  device_boxes.CopyFromHost(boxes, boxes_bytes);
  device_scores.CopyFromHost(scores, scores_bytes);
  const uint64_t zero = 0;
  device_count.CopyFromHost(&zero, sizeof(zero));

  const std::string details =
      " boxes=" + std::to_string(shape.num_boxes) +
      " classes=" + std::to_string(shape.num_classes) +
      " max_output=" + std::to_string(shape.max_output_per_class);
  const uint32_t iou_bits = [&] {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(iou_threshold));
    std::memcpy(&bits, &iou_threshold, sizeof(bits));
    return bits;
  }();
  const uint32_t score_bits = [&] {
    uint32_t bits = 0;
    std::memcpy(&bits, &score_threshold, sizeof(bits));
    return bits;
  }();
  LaunchKernelLocked(
      "RVNPU NonMaxSuppression", kernels::kNonMaxSuppressionFp32, details,
      device_boxes.address(), device_scores.address(), device_output.address(),
      device_count.address(), shape.num_boxes, shape.num_classes,
      shape.max_output_per_class, iou_bits, score_bits);

  uint64_t selected = 0;
  device_count.CopyToHost(&selected, sizeof(selected));
  if (selected > shape.max_selected) {
    throw std::runtime_error("RVNPU NMS returned an invalid output count");
  }
  std::vector<int64_t> output(static_cast<size_t>(selected) * 3);
  if (!output.empty())
    device_output.CopyToHost(output.data(), output.size() * sizeof(int64_t));
  return output;
}

} // namespace rvnpu
