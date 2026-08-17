#include "ops/gather_nd/gather_nd.h"

#include <stdexcept>

namespace rvnpu {
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
    throw std::invalid_argument("RVNPU GatherND batch_dims is invalid");
  return result;
}

RVNPU_DEFINE_KERNEL(
    GatherND, kOnnxDomain, 13,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T",
                            TensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
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
  RVNPU_EXCEPTION_GUARD_BEGIN
  if (state == nullptr)
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "RVNPU GatherND has no DeviceContext");
  output = new GatherNdKernel(info, static_cast<DeviceContext *>(state));
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

OrtStatus *ORT_API_CALL GatherNdKernel::ComputeImpl(
    OrtKernelImpl *kernel, OrtKernelContext *context) noexcept {
  RVNPU_EXCEPTION_GUARD_BEGIN
  auto &self = *static_cast<GatherNdKernel *>(kernel);
  Ort::KernelContext kernel_context(context);
  const Ort::ConstValue data = kernel_context.GetInput(0);
  const Ort::ConstValue indices = kernel_context.GetInput(1);
  const auto data_info = data.GetTensorTypeAndShapeInfo();
  const auto indices_info = indices.GetTensorTypeAndShapeInfo();
  if (data_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      indices_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::invalid_argument(
        "RVNPU GatherND requires FP32 data and INT64 indices");
  }
  const auto shape = ValidateGatherNdShape(
      data_info.GetShape(), indices_info.GetShape(), self.batch_dims_);
  Ort::UnownedValue output =
      kernel_context.GetOutput(0, shape.output_shape);
  if (shape.output_elements != 0) {
    self.context_->RunGatherNd(data.GetTensorData<float>(),
                               indices.GetTensorData<int64_t>(),
                               output.GetTensorMutableData<float>(), shape);
  }
  return nullptr;
  RVNPU_EXCEPTION_GUARD_END
}

void ORT_API_CALL GatherNdKernel::ReleaseImpl(OrtKernelImpl *kernel) noexcept {
  delete static_cast<GatherNdKernel *>(kernel);
}

} // namespace rvnpu
