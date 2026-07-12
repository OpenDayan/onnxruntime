// Portions follow ONNX Runtime's example Plugin EP, licensed under MIT.
#pragma once

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#define RVNPU_RETURN_IF_ERROR(expr)                                            \
  do {                                                                         \
    Ort::Status status{(expr)};                                                \
    if (!status.IsOK()) {                                                      \
      return status.release();                                                 \
    }                                                                          \
  } while (false)

#define RVNPU_IGNORE_STATUS(expr)                                              \
  do {                                                                         \
    Ort::Status ignored{(expr)};                                               \
  } while (false)

#define RVNPU_EXCEPTION_GUARD_BEGIN try {
#define RVNPU_EXCEPTION_GUARD_END                                              \
  }                                                                            \
  catch (const Ort::Exception &ex) {                                           \
    return Ort::Status(ex).release();                                          \
  }                                                                            \
  catch (const std::exception &ex) {                                           \
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();                      \
  }                                                                            \
  catch (...) {                                                                \
    return Ort::Status("unknown RVNPU EP exception", ORT_EP_FAIL).release();   \
  }

inline std::string GetSessionOption(const OrtSessionOptions &options,
                                    const std::string &key,
                                    const std::string &fallback = {}) {
  return Ort::ConstSessionOptions{&options}.GetConfigEntryOrDefault(key.c_str(),
                                                                    fallback);
}

inline std::optional<std::vector<int64_t>>
TensorShape(Ort::ConstValueInfo info) {
  const auto type_info = info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return std::nullopt;
  }
  return type_info.GetTensorTypeAndShapeInfo().GetShape();
}

inline bool IsFloatTensor(Ort::ConstValueInfo info) {
  const auto type_info = info.TypeInfo();
  return type_info.GetONNXType() == ONNX_TYPE_TENSOR &&
         type_info.GetTensorTypeAndShapeInfo().GetElementType() ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
