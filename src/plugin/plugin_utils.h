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

#define DAYAN_RETURN_IF_ERROR(expr)                                            \
  do {                                                                         \
    Ort::Status status{(expr)};                                                \
    if (!status.IsOK()) {                                                      \
      return status.release();                                                 \
    }                                                                          \
  } while (false)

#define DAYAN_IGNORE_STATUS(expr)                                              \
  do {                                                                         \
    Ort::Status ignored{(expr)};                                               \
  } while (false)

#define DAYAN_EXCEPTION_GUARD_BEGIN try {
#define DAYAN_EXCEPTION_GUARD_END                                              \
  }                                                                            \
  catch (const Ort::Exception &ex) {                                           \
    return Ort::Status(ex).release();                                          \
  }                                                                            \
  catch (const std::exception &ex) {                                           \
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();                      \
  }                                                                            \
  catch (...) {                                                                \
    return Ort::Status("unknown Dayan EP exception", ORT_EP_FAIL).release();   \
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

inline bool IsTensorType(Ort::ConstValueInfo info,
                         ONNXTensorElementDataType element_type) {
  const auto type_info = info.TypeInfo();
  return type_info.GetONNXType() == ONNX_TYPE_TENSOR &&
         type_info.GetTensorTypeAndShapeInfo().GetElementType() ==
             element_type;
}

inline bool IsFloatTensor(Ort::ConstValueInfo info) {
  return IsTensorType(info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}
