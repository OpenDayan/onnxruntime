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

#include "ops/gather_nd/gather_nd.h"

#include <stdexcept>
#include <vector>

namespace dayan {
namespace {

int64_t KernelBatchDims(const OrtKernelInfo *info) {
  try {
    return Ort::ConstKernelInfo{info}.GetAttribute<int64_t>("batch_dims");
  } catch (const Ort::Exception &) {
    return 0;
  }
}

} // namespace

int64_t ReadGatherNdBatchDims(const Ort::ConstNode &node) {
  Ort::ConstOpAttr attribute;
  const Ort::Status status = node.GetAttributeByName("batch_dims", attribute);
  if (!status.IsOK() || attribute == nullptr)
    return 0;
  int64_t result = 0;
  const Ort::Status read_status = attribute.GetValue<int64_t>(result);
  if (!read_status.IsOK())
    throw std::invalid_argument("Dayan GatherND batch_dims is invalid");
  return result;
}

DAYAN_DEFINE_KERNEL(
    GatherND, kOnnxDomain, 13,
    (Ort::KernelDefBuilder().AddTypeConstraint(
        "T", std::vector<const OrtDataType *>{
                 TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
                 TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)})),
    GatherNdKernel)

GatherNdKernel::GatherNdKernel(const OrtKernelInfo *info,
                               DeviceContext *context)
    : OrtKernelImpl{}, batch_dims_(KernelBatchDims(info)), context_(context) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus *GatherNdKernel::Create(const OrtKernelInfo *info, void *state,
                                  OrtKernelImpl *&output) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "Dayan GatherND has no DeviceContext");
  output = new GatherNdKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL GatherNdKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  DAYAN_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<GatherNdKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue data = kernel_context.GetInput(0);
  const Ort::ConstValue indices = kernel_context.GetInput(1);
  const auto data_info = data.GetTensorTypeAndShapeInfo();
  const auto indices_info = indices.GetTensorTypeAndShapeInfo();
  const auto data_type = data_info.GetElementType();
  if ((data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
       data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      indices_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    throw std::invalid_argument(
        "Dayan GatherND requires FP32 or INT64 data and INT64 indices");
  const auto shape = ValidateGatherNdShape(
      data_info.GetShape(), indices_info.GetShape(), self.batch_dims_,
      data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? sizeof(int64_t)
                                                       : sizeof(float));
  Ort::UnownedValue output =
      kernel_context.GetOutput(0, shape.output_shape);
  if (shape.output_elements != 0 &&
      data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    self.context_->RunGatherNd(data.GetTensorData<float>(),
                               indices.GetTensorData<int64_t>(),
                               output.GetTensorMutableData<float>(), shape);
  } else if (shape.output_elements != 0) {
    self.context_->RunGatherNd(data.GetTensorData<int64_t>(),
                               indices.GetTensorData<int64_t>(),
                               output.GetTensorMutableData<int64_t>(), shape);
  }
  return nullptr;
  DAYAN_EXCEPTION_GUARD_END
}

void ORT_API_CALL GatherNdKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<GatherNdKernel *>(kernel);
}

} // namespace dayan
