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

#include "ops/resize/resize.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace dayan {
namespace {

std::string NodeString(const Ort::ConstNode &node, const char *name,
                       const char *default_value) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName(name, attribute);
  if (!status.IsOK())
    throw std::invalid_argument(std::string("cannot read Resize ") + name);
  if (attribute == nullptr)
    return default_value;
  std::string result;
  const Ort::Status read_status = attribute.GetValue<std::string>(result);
  if (!read_status.IsOK())
    throw std::invalid_argument(std::string("invalid Resize ") + name);
  return result;
}

std::string KernelString(const OrtKernelInfo *info, const char *name,
                         const char *default_value) {
  try {
    return Ort::ConstKernelInfo{info}.GetAttribute<std::string>(name);
  } catch (const Ort::Exception &) {
    return default_value;
  }
}

ResizeAttributes ReadResizeAttributes(const OrtKernelInfo *info) {
  return {KernelString(info, "coordinate_transformation_mode", "half_pixel"),
          KernelString(info, "mode", "nearest"),
          KernelString(info, "nearest_mode", "round_prefer_floor")};
}

} // namespace

ResizeAttributes ReadResizeAttributes(const Ort::ConstNode &node) {
  return {NodeString(node, "coordinate_transformation_mode", "half_pixel"),
          NodeString(node, "mode", "nearest"),
          NodeString(node, "nearest_mode", "round_prefer_floor")};
}

bool IsSupportedResizeAttributes(const ResizeAttributes &attributes) {
  return attributes.coordinate_transformation_mode == "asymmetric" &&
         attributes.mode == "nearest" && attributes.nearest_mode == "floor";
}

DAYAN_DEFINE_KERNEL(Resize, kOnnxDomain, 13,
                    (Ort::KernelDefBuilder().AddTypeConstraint(
                        "T1", TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
                    ResizeKernel)

ResizeKernel::ResizeKernel(const OrtKernelInfo *info, DeviceContext *context)
    : OrtKernelImpl{}, attributes_(ReadResizeAttributes(info)),
      context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *ResizeKernel::Create(const OrtKernelInfo *info, void *state,
                                OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan Resize has no DeviceContext");
  output = new ResizeKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL ResizeKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<ResizeKernel *>(kernel);
  if (!IsSupportedResizeAttributes(self.attributes_))
    throw std::invalid_argument(
        "Dayan Resize requires nearest/asymmetric/floor");
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const auto input_info = input.GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      kernel_context.GetInputCount() < 3)
    throw std::invalid_argument("Dayan Resize requires FP32 input and scales");
  const Ort::ConstValue scales = kernel_context.GetInput(2);
  if (scales == nullptr)
    throw std::invalid_argument("Dayan Resize requires scales");
  const auto scale_info = scales.GetTensorTypeAndShapeInfo();
  if (scale_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      scale_info.GetElementCount() != 4)
    throw std::invalid_argument(
        "Dayan Resize scales must contain four FP32 values");
  const float *scale = scales.GetTensorData<float>();
  const auto input_shape = input_info.GetShape();
  if (input_shape.size() != 4 || scale[0] != 1.0f || scale[1] != 1.0f)
    throw std::invalid_argument("Dayan Resize only changes H/W");
  std::vector<int64_t> output_shape = input_shape;
  output_shape[2] = static_cast<int64_t>(std::floor(input_shape[2] * scale[2]));
  output_shape[3] = static_cast<int64_t>(std::floor(input_shape[3] * scale[3]));
  const auto shape = ValidateResizeShape(input_shape, output_shape);
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  self.context_->RunResize(input.GetTensorData<float>(),
                           output.GetTensorMutableData<float>(), shape);
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL ResizeKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<ResizeKernel *>(kernel);
}

} // namespace dayan
