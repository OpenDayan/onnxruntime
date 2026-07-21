#include "runtime/device_context.h"
#include "runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TestCase {
  std::vector<float> boxes;
  std::vector<float> scores;
  int64_t num_boxes{};
  int64_t num_classes{};
  int64_t max_output{};
  float iou_threshold{};
  float score_threshold{};
};

float Iou(const float *lhs, const float *rhs) {
  const float ly1 = std::min(lhs[0], lhs[2]);
  const float lx1 = std::min(lhs[1], lhs[3]);
  const float ly2 = std::max(lhs[0], lhs[2]);
  const float lx2 = std::max(lhs[1], lhs[3]);
  const float ry1 = std::min(rhs[0], rhs[2]);
  const float rx1 = std::min(rhs[1], rhs[3]);
  const float ry2 = std::max(rhs[0], rhs[2]);
  const float rx2 = std::max(rhs[1], rhs[3]);
  const float intersection =
      std::max(0.0f, std::min(ly2, ry2) - std::max(ly1, ry1)) *
      std::max(0.0f, std::min(lx2, rx2) - std::max(lx1, rx1));
  const float area_l = (ly2 - ly1) * (lx2 - lx1);
  const float area_r = (ry2 - ry1) * (rx2 - rx1);
  const float union_area = area_l + area_r - intersection;
  return union_area <= 0.0f ? 0.0f : intersection / union_area;
}

std::vector<int64_t> Reference(const TestCase &test) {
  std::vector<int64_t> output;
  for (int64_t class_index = 0; class_index < test.num_classes;
       ++class_index) {
    std::vector<int64_t> candidates(test.num_boxes);
    for (int64_t index = 0; index < test.num_boxes; ++index)
      candidates[index] = index;
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&](int64_t lhs, int64_t rhs) {
                       return test.scores[class_index * test.num_boxes + lhs] >
                              test.scores[class_index * test.num_boxes + rhs];
                     });
    std::vector<int64_t> selected;
    for (const int64_t candidate : candidates) {
      if (test.scores[class_index * test.num_boxes + candidate] <=
          test.score_threshold)
        continue;
      bool suppressed = false;
      for (const int64_t previous : selected) {
        if (Iou(test.boxes.data() + candidate * 4,
                test.boxes.data() + previous * 4) > test.iou_threshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        output.insert(output.end(), {0, class_index, candidate});
        selected.push_back(candidate);
        if (selected.size() == static_cast<size_t>(test.max_output))
          break;
      }
    }
  }
  return output;
}

TestCase MakeCase(const std::string &name) {
  const std::vector<float> boxes = {
      0.0f, 0.0f, 2.0f, 2.0f, 0.1f, 0.1f, 2.1f, 2.1f,
      3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 0.0f, 6.0f, 1.0f};
  if (name == "nonoverlap") {
    return {{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5},
            {0.9f, 0.8f, 0.7f}, 3, 1, 3, 0.5f, 0.0f};
  }
  if (name == "overlap")
    return {boxes, {0.9f, 0.8f, 0.7f, 0.6f}, 4, 1, 3, 0.5f, 0.0f};
  if (name == "multiclass") {
    return {boxes,
            {0.9f, 0.8f, 0.7f, 0.6f, 0.1f, 0.95f, 0.5f, 0.85f},
            4, 2, 2, 0.5f, 0.0f};
  }
  if (name == "empty")
    return {boxes, {0.9f, 0.8f, 0.7f, 0.6f}, 4, 1, 3, 0.5f, 0.95f};
  if (name == "boundary") {
    return {{0.0f, 0.0f, 1.0f, 1.0f,
             0.0f, 0.0f, 1.0f, 2.0f},
            {0.9f, 0.8f}, 2, 1, 2, 0.5f, 0.0f};
  }
  throw std::invalid_argument("unknown NMS case: " + name);
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string kernel_elf;
    std::string case_name;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--kernel-elf" && index + 1 < argc)
        kernel_elf = argv[++index];
      else if (argument == "--case" && index + 1 < argc)
        case_name = argv[++index];
      else
        throw std::invalid_argument("unknown or incomplete argument: " +
                                    argument);
    }
    if (kernel_elf.empty() || case_name.empty())
      throw std::invalid_argument("--kernel-elf and --case are required");
    const TestCase test = MakeCase(case_name);
    const auto shape = rvnpu::ValidateNmsShapes(
        {1, test.num_boxes, 4}, {1, test.num_classes, test.num_boxes},
        test.max_output, {});
    rvnpu::DeviceContext context(kernel_elf);
    const auto output =
        context.RunNms(test.boxes.data(), test.scores.data(),
                       test.iou_threshold, test.score_threshold, shape);
    if (output != Reference(test))
      throw std::runtime_error("NMS output mismatch");
    std::cout << "PASS case=" << case_name
              << " boxes=" << test.num_boxes
              << " classes=" << test.num_classes
              << " selected=" << output.size() / 3 << std::endl;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << std::endl;
    return 1;
  }
}
