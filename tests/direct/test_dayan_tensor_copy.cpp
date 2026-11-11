/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * onnxruntime is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include "runtime/device_context.h"
#include "runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string KernelElf(int argc, char **argv, std::string &op) {
  std::string elf;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--kernel-elf" && index + 1 < argc)
      elf = argv[++index];
    else if (argument == "--op" && index + 1 < argc)
      op = argv[++index];
    else
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
  }
  if (elf.empty() || (op != "concat" && op != "split" && op != "slice"))
    throw std::invalid_argument("--kernel-elf and --op are required");
  return elf;
}

std::vector<float> Values(size_t count, int salt) {
  std::vector<float> values(count);
  for (size_t index = 0; index < count; ++index)
    values[index] = static_cast<float>(index + salt * 100);
  return values;
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
    std::string op;
    dayan::DeviceContext context(KernelElf(argc, argv, op));
    if (op == "concat") {
      const auto shape =
          dayan::ValidateConcatShapes({{1, 2, 2, 16}, {1, 1, 2, 16}}, 1);
      const auto a = Values(64, 1);
      const auto b = Values(32, 2);
      std::vector<float> output(shape.output_elements);
      std::vector<float> reference;
      reference.reserve(output.size());
      reference.insert(reference.end(), a.begin(), a.end());
      reference.insert(reference.end(), b.begin(), b.end());
      context.RunConcat({a.data(), b.data()}, output.data(), shape);
      std::cout << "Concat axis=1 output=[1,3,2,16] ";
      Compare(output, reference);
    } else if (op == "split") {
      const auto shape = dayan::ValidateSplitShape({1, 4, 2, 16}, {2, 2}, 1, 2);
      const auto input = Values(128, 1);
      std::vector<float> first(64), second(64);
      context.RunSplit(input.data(), {first.data(), second.data()}, shape);
      std::vector<float> output = first;
      output.insert(output.end(), second.begin(), second.end());
      std::cout << "Split axis=1 parts=[2,2] ";
      Compare(output, input);
    } else {
      const auto shape = dayan::ValidateSliceShape({1, 4, 16}, 1, 3, 1, 1);
      const auto input = Values(64, 1);
      std::vector<float> output(shape.output_elements);
      const std::vector<float> reference(input.begin() + 16,
                                         input.begin() + 48);
      context.RunSlice(input.data(), output.data(), shape);
      std::cout << "Slice axis=1 range=[1,3) ";
      Compare(output, reference);
    }
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
