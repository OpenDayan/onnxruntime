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
  int64_t center_point_box{};
};

TestCase MakeProfileCase(int64_t boxes_count, int64_t classes_count,
                         int64_t max_output, float iou_threshold,
                         float score_threshold) {
  TestCase test;
  test.num_boxes = boxes_count;
  test.num_classes = classes_count;
  test.max_output = max_output;
  test.iou_threshold = iou_threshold;
  test.score_threshold = score_threshold;
  test.center_point_box = 0;
  test.boxes.resize(static_cast<size_t>(boxes_count) * 4);
  test.scores.resize(static_cast<size_t>(classes_count) * boxes_count);
  for (int64_t index = 0; index < boxes_count; ++index) {
    test.boxes[static_cast<size_t>(index) * 4 + 0] =
        static_cast<float>(index % 64);
    test.boxes[static_cast<size_t>(index) * 4 + 1] =
        static_cast<float>(index % 64);
    test.boxes[static_cast<size_t>(index) * 4 + 2] =
        static_cast<float>(index % 64) + 1.0f;
    test.boxes[static_cast<size_t>(index) * 4 + 3] =
        static_cast<float>(index % 64) + 1.0f;
  }
  for (size_t index = 0; index < test.scores.size(); ++index)
    test.scores[index] = 0.1f + static_cast<float>(index % 17) * 0.05f;
  return test;
}

void Corners(const float *box, int64_t center_point_box, float &y1,
             float &x1, float &y2, float &x2) {
  if (center_point_box == 1) {
    x1 = box[0] - box[2] * 0.5f;
    y1 = box[1] - box[3] * 0.5f;
    x2 = box[0] + box[2] * 0.5f;
    y2 = box[1] + box[3] * 0.5f;
  } else {
    y1 = std::min(box[0], box[2]);
    x1 = std::min(box[1], box[3]);
    y2 = std::max(box[0], box[2]);
    x2 = std::max(box[1], box[3]);
  }
}

float Iou(const float *lhs, const float *rhs, int64_t center_point_box) {
  float ly1 = 0.0f, lx1 = 0.0f, ly2 = 0.0f, lx2 = 0.0f;
  float ry1 = 0.0f, rx1 = 0.0f, ry2 = 0.0f, rx2 = 0.0f;
  Corners(lhs, center_point_box, ly1, lx1, ly2, lx2);
  Corners(rhs, center_point_box, ry1, rx1, ry2, rx2);
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
                test.boxes.data() + previous * 4,
                test.center_point_box) > test.iou_threshold) {
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
            {0.9f, 0.8f, 0.7f}, 3, 1, 3, 0.5f, 0.0f, 0};
  }
  if (name == "overlap")
    return {boxes, {0.9f, 0.8f, 0.7f, 0.6f}, 4, 1, 3, 0.5f, 0.0f, 0};
  if (name == "multiclass") {
    return {boxes,
            {0.9f, 0.8f, 0.7f, 0.6f, 0.1f, 0.95f, 0.5f, 0.85f},
            4, 2, 2, 0.5f, 0.0f, 0};
  }
  if (name == "empty")
    return {boxes, {0.9f, 0.8f, 0.7f, 0.6f}, 4, 1, 3, 0.5f, 0.95f, 0};
  if (name == "boundary") {
    return {{0.0f, 0.0f, 1.0f, 1.0f,
             0.0f, 0.0f, 1.0f, 2.0f},
            {0.9f, 0.8f}, 2, 1, 2, 0.5f, 0.0f, 0};
  }
  if (name == "center") {
    return {{1.0f, 1.0f, 2.0f, 2.0f,
             1.1f, 1.1f, 2.0f, 2.0f,
             3.5f, 3.5f, 1.0f, 1.0f,
             0.5f, 5.5f, 1.0f, 1.0f},
            {0.9f, 0.8f, 0.7f, 0.6f}, 4, 1, 3, 0.5f, 0.0f, 1};
  }
  throw std::invalid_argument("unknown NMS case: " + name);
}

void TestGatherNd(dayan::DeviceContext &context) {
  const std::vector<float> candidates = {
      0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23};
  const std::vector<int64_t> candidate_indices = {2, 0};
  std::vector<float> selected_candidates(8);
  const auto candidate_shape =
      dayan::ValidateGatherNdShape({3, 4}, {2, 1}, 0);
  context.RunGatherNd(candidates.data(), candidate_indices.data(),
                      selected_candidates.data(), candidate_shape);
  if (selected_candidates !=
      std::vector<float>({20, 21, 22, 23, 0, 1, 2, 3}))
    throw std::runtime_error("GatherND candidate filter mismatch");

  const std::vector<float> boxes = {
      0, 0, 1, 1, 2, 2, 3, 3,
      4, 4, 5, 5, 6, 6, 7, 7};
  const std::vector<int64_t> box_indices = {0, 2, 0, 0};
  std::vector<float> selected_boxes(8);
  const auto box_shape =
      dayan::ValidateGatherNdShape({1, 4, 4}, {2, 2}, 0);
  context.RunGatherNd(boxes.data(), box_indices.data(),
                      selected_boxes.data(), box_shape);
  if (selected_boxes != std::vector<float>({4, 4, 5, 5, 0, 0, 1, 1}))
    throw std::runtime_error("GatherND boxes mismatch");

  std::vector<float> scores(32);
  for (size_t index = 0; index < scores.size(); ++index)
    scores[index] = static_cast<float>(index);
  const std::vector<int64_t> score_indices = {0, 1, 15, 0, 0, 0};
  std::vector<float> selected_scores(2);
  const auto score_shape =
      dayan::ValidateGatherNdShape({1, 2, 16}, {2, 3}, 0);
  context.RunGatherNd(scores.data(), score_indices.data(),
                      selected_scores.data(), score_shape);
  if (selected_scores != std::vector<float>({scores[31], scores[0]}))
    throw std::runtime_error("GatherND scores mismatch");
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
      throw std::invalid_argument("");
    dayan::DeviceContext context(kernel_elf);
    if (case_name == "gathernd") {
      TestGatherNd(context);
      std::cout << "PASS case=gathernd candidates=2 boxes=2 scores=2"
                << std::endl;
      return 0;
    }
    const TestCase test = MakeCase(case_name);
    const auto shape = dayan::ValidateNmsShapes(
        {1, test.num_boxes, 4}, {1, test.num_classes, test.num_boxes},
        test.max_output, {test.center_point_box});
    const auto output =
        context.RunNms(test.boxes.data(), test.scores.data(),
                       test.iou_threshold, test.score_threshold, shape);
    if (case_name == "profile") {
      std::cout << "PASS case=profile boxes=" << test.num_boxes
                << " classes=" << test.num_classes
                << " max_output=" << test.max_output
                << " selected=" << output.size() / 3 << std::endl;
      return 0;
    }
    const auto expected = Reference(test);
    if (output != expected) {
      auto print = [](const char *name, const std::vector<int64_t> &values) {
        std::cerr << name << "=[";
        for (size_t index = 0; index < values.size(); ++index)
          std::cerr << (index ? "," : "") << values[index];
        std::cerr << "] ";
      };
      print("actual", output);
      print("expected", expected);
      std::cerr << "\n";
      throw std::runtime_error("NMS output mismatch");
    }
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
