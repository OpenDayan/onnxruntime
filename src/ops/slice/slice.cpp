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

#include "ops/slice/slice.h"

#include <stdexcept>
#include <vector>

namespace dayan {
namespace {

int64_t ScalarInt64(const Ort::KernelContext &context, size_t index,
                    int64_t default_value, bool required) {
  if (index >= context.GetInputCount()) {
    if (required)
      throw std::invalid_argument("Dayan Slice is missing an input");
    return default_value;
  }
  const Ort::ConstValue value = context.GetInput(index);
  if (value == nullptr) {
    if (required)
      throw std::invalid_argument("Dayan Slice is missing an input");
    return default_value;
  }
  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      info.GetElementCount() != 1)
    throw std::invalid_argument("Dayan Slice controls must be scalar INT64");
  return *value.GetTensorData<int64_t>();
}

} // namespace

DAYAN_DEFINE_KERNEL(
    Slice, kOnnxDomain, 13,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint(
             "T", std::vector<const OrtDataType *>{
                      TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
                      TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)})
         .AddTypeConstraint("Tind",
                            TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))),
    SliceKernel)

SliceKernel::SliceKernel(DeviceContext *context)
    : OrtKernelImpl{}, context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *SliceKernel::Create(const OrtKernelInfo *, void *state,
                               OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan Slice has no DeviceContext");
  output = new SliceKernel(static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL SliceKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<SliceKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue input = kernel_context.GetInput(0);
  const auto input_info = input.GetTensorTypeAndShapeInfo();
  const auto input_type = input_info.GetElementType();
  if (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::invalid_argument("Dayan Slice requires FP32 or INT64 input");
  }
  const int64_t start = ScalarInt64(kernel_context, 1, 0, true);
  const int64_t end = ScalarInt64(kernel_context, 2, 0, true);
  const int64_t axis = ScalarInt64(kernel_context, 3, 0, false);
  const int64_t step = ScalarInt64(kernel_context, 4, 1, false);
  const auto shape =
      ValidateSliceShape(input_info.GetShape(), start, end, axis, step);
  Ort::UnownedValue output = kernel_context.GetOutput(0, shape.output_shape);
  if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    self.context_->RunSlice(input.GetTensorData<float>(),
                            output.GetTensorMutableData<float>(), shape);
  } else {
    self.context_->RunSlice(input.GetTensorData<int64_t>(),
                            output.GetTensorMutableData<int64_t>(), shape);
  }
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL SliceKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<SliceKernel *>(kernel);
}

} // namespace dayan
