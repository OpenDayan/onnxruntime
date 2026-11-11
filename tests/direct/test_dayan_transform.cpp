#include "runtime/device_context.h"
#include "runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string elf;
  std::string op;
};

Options Parse(int argc, char **argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--kernel-elf" && index + 1 < argc)
      result.elf = argv[++index];
    else if (argument == "--op" && index + 1 < argc)
      result.op = argv[++index];
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (result.elf.empty() ||
      (result.op != "resize" && result.op != "transpose" &&
       result.op != "reduce_max" && result.op != "softmax"))
    throw std::invalid_argument("--kernel-elf and --op are required");
  return result;
}

void Compare(const std::vector<float> &actual,
             const std::vector<float> &reference) {
  if (actual.size() != reference.size())
    throw std::runtime_error("output size mismatch");
  float maximum = 0.0f;
  for (size_t index = 0; index < actual.size(); ++index) {
    maximum = std::max(maximum, std::abs(actual[index] - reference[index]));
    if (actual[index] != reference[index])
      throw std::runtime_error("mismatch at element " + std::to_string(index));
  }
  std::cout << "PASS max_abs_diff=" << maximum << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    dayan::DeviceContext context(options.elf);
    if (options.op == "resize") {
      const auto shape = dayan::ValidateResizeShape({1, 2, 4, 4}, {1, 2, 8, 8});
      std::vector<float> input(shape.input_elements);
      std::vector<float> output(shape.output_elements);
      std::vector<float> reference(shape.output_elements);
      for (size_t index = 0; index < input.size(); ++index)
        input[index] = static_cast<float>(index);
      for (int32_t c = 0; c < 2; ++c)
        for (int32_t h = 0; h < 8; ++h)
          for (int32_t w = 0; w < 8; ++w)
            reference[(c * 8 + h) * 8 + w] = input[(c * 4 + h / 2) * 4 + w / 2];
      context.RunResize(input.data(), output.data(), shape);
      std::cout << "Resize [1,2,4,4] -> [1,2,8,8] ";
      Compare(output, reference);
    } else if (options.op == "transpose") {
      const auto run = [&](const std::vector<int64_t> &input_shape,
                           const std::vector<int64_t> &perm) {
        const auto shape = dayan::ValidateTransposeShape(input_shape, perm);
        std::vector<float> input(shape.elements), output(shape.elements),
            reference(shape.elements);
        for (size_t index = 0; index < input.size(); ++index)
          input[index] = static_cast<float>(index);
        const int32_t d0 = shape.dims[0];
        const int32_t d1 = shape.dims[1];
        const int32_t d2 = shape.dims[2];
        const int32_t d3 = shape.dims[3];
        for (int32_t i0 = 0; i0 < d0; ++i0)
          for (int32_t i1 = 0; i1 < d1; ++i1)
            for (int32_t i2 = 0; i2 < d2; ++i2)
              for (int32_t i3 = 0; i3 < d3; ++i3) {
                const size_t source =
                    ((static_cast<size_t>(i0) * d1 + i1) * d2 + i2) * d3 + i3;
                const size_t destination =
                    ((static_cast<size_t>(i0) * d2 + i2) * d1 + i1) * d3 + i3;
                reference[destination] = input[source];
              }
        context.RunTranspose(input.data(), output.data(), shape);
        std::cout << "Transpose rank=" << shape.rank << " ";
        Compare(output, reference);
      };
      run({1, 84, 32}, {0, 2, 1});
      run({1, 4, 2, 16}, {0, 2, 1, 3});
    } else if (options.op == "reduce_max") {
      const auto shape =
          dayan::ValidateReduceMaxShape({1, 32, 80}, {-1}, 0);
      std::vector<float> input(shape.input_elements),
          output(shape.output_elements), reference(shape.output_elements);
      for (size_t index = 0; index < input.size(); ++index)
        input[index] =
            static_cast<float>(static_cast<int>(index % 97) - 48);
      for (int32_t row = 0; row < shape.rows; ++row) {
        float maximum = input[static_cast<size_t>(row) * shape.columns];
        for (int32_t column = 1; column < shape.columns; ++column)
          maximum = std::max(
              maximum, input[static_cast<size_t>(row) * shape.columns + column]);
        reference[row] = maximum;
      }
      context.RunReduceMax(input.data(), output.data(), shape);
      std::cout << "ReduceMax shape=[1,32,80] axis=-1 ";
      Compare(output, reference);
    } else {
      const auto shape = dayan::ValidateSoftmaxAxis1Shape({1, 16, 2, 16}, 1);
      std::vector<float> input(shape.elements), output(shape.elements),
          reference(shape.elements);
      for (size_t index = 0; index < input.size(); ++index)
        input[index] =
            static_cast<float>(static_cast<int>(index % 13) - 6) / 4.0f;
      for (int32_t inner = 0; inner < shape.inner; ++inner) {
        float maximum = input[inner];
        for (int32_t axis = 1; axis < shape.axis; ++axis)
          maximum = std::max(maximum, input[axis * shape.inner + inner]);
        float sum = 0.0f;
        for (int32_t axis = 0; axis < shape.axis; ++axis) {
          const size_t index = axis * shape.inner + inner;
          reference[index] = std::exp(input[index] - maximum);
          sum += reference[index];
        }
        for (int32_t axis = 0; axis < shape.axis; ++axis)
          reference[axis * shape.inner + inner] /= sum;
      }
      context.RunSoftmax(input.data(), output.data(), shape);
      float maximum = 0.0f;
      for (size_t index = 0; index < output.size(); ++index) {
        maximum = std::max(maximum, std::abs(output[index] - reference[index]));
        if (!std::isfinite(output[index]) ||
            std::abs(output[index] - reference[index]) > 2e-3f) {
          throw std::runtime_error(
              "Softmax mismatch at element " + std::to_string(index) +
              ": got=" + std::to_string(output[index]) +
              " reference=" + std::to_string(reference[index]));
        }
      }
      std::cout << "Softmax shape=[1,16,2,16] axis=1 PASS max_abs_diff="
                << maximum << std::endl;
    }
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
