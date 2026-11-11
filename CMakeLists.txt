#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# onnxruntime is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

cmake_minimum_required(VERSION 3.20)
project(onnxruntime VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime source or SDK root")
set(ONNXRUNTIME_LIB_DIR "" CACHE PATH "Directory containing libonnxruntime")
set(DAYAN_ROOT "" CACHE PATH "dayan-base repository root")
set(DAYAN_RUNTIME_ROOT "${DAYAN_ROOT}" CACHE PATH "dayan-base runtime repository root")
set(DAYAN_KERNEL_ROOT "${DAYAN_ROOT}" CACHE PATH "dayan-base kernel repository root")
set(DAYAN_KERNEL_OBJECT_DIR "${DAYAN_KERNEL_ROOT}/build/demo/onnx" CACHE PATH "dycc-generated ONNX kernel objects")
file(GLOB DAYAN_KERNEL_OBJECTS "${DAYAN_KERNEL_OBJECT_DIR}/*.o")

foreach(required_var ONNXRUNTIME_ROOT ONNXRUNTIME_LIB_DIR DAYAN_RUNTIME_ROOT DAYAN_KERNEL_ROOT)
  if(NOT ${required_var})
    message(FATAL_ERROR "${required_var} is required; set DAYAN_ROOT or the individual paths")
  endif()
endforeach()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS
    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
    "${ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}" NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "Could not locate ONNX Runtime headers/library under the supplied paths")
endif()

if(NOT DAYAN_KERNEL_OBJECTS)
  message(FATAL_ERROR "No dycc-generated Dayan ONNX kernel objects in ${DAYAN_KERNEL_OBJECT_DIR}")
endif()

add_library(dayan_runtime_vendor STATIC
  ${DAYAN_KERNEL_OBJECTS})
target_include_directories(dayan_runtime_vendor PUBLIC
  "${DAYAN_RUNTIME_ROOT}/host/include")
target_link_libraries(dayan_runtime_vendor PUBLIC
  "${DAYAN_RUNTIME_ROOT}/build/host/driver/libDayanRuntime.a"
  ${CMAKE_DL_LIBS} pthread)
set_target_properties(dayan_runtime_vendor PROPERTIES LINKER_LANGUAGE CXX)

add_library(dayan_runtime_bridge STATIC
  src/runtime/device_context.cpp
  src/runtime/device_buffer.cpp
  src/runtime/dayan_context_impl.cpp)
target_include_directories(dayan_runtime_bridge PUBLIC
  src "${DAYAN_RUNTIME_ROOT}/host/include")
target_link_libraries(dayan_runtime_bridge PUBLIC dayan_runtime_vendor)


add_library(dayan_ep SHARED
  src/plugin/plugin_entry.cpp
  src/plugin/ep_factory.cpp
  src/plugin/ep.cpp
  src/plugin/memory.cpp
  src/plugin/kernel_registry.cpp
  src/ops/argmax/argmax.cpp
  src/ops/concat/concat.cpp
  src/ops/conv/conv.cpp
  src/ops/elementwise/elementwise.cpp
  src/ops/gather_nd/gather_nd.cpp
  src/ops/matmul/matmul.cpp
  src/ops/maxpool/maxpool.cpp
  src/ops/topk/topk.cpp
  src/ops/nms/nms.cpp)
target_sources(dayan_ep PRIVATE
  src/ops/resize/resize.cpp
  src/ops/reduce_max/reduce_max.cpp
  src/ops/slice/slice.cpp
  src/ops/softmax/softmax.cpp
  src/ops/split/split.cpp
  src/ops/transpose/transpose.cpp)
target_include_directories(dayan_ep PRIVATE src "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(dayan_ep PRIVATE dayan_runtime_bridge "${ONNXRUNTIME_LIBRARY}")
set_target_properties(dayan_ep PROPERTIES
  OUTPUT_NAME dayan_ep
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")
target_link_options(dayan_ep PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/dayan_ep.lds")

add_executable(test_dayan_tiny_cnn tests/integration/test_dayan_tiny_cnn.cpp)
target_include_directories(test_dayan_tiny_cnn PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_dayan_tiny_cnn PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_dayan_tiny_cnn PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_dayan_onnx_model tests/integration/test_dayan_onnx_model.cpp)
target_include_directories(test_dayan_onnx_model PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_dayan_onnx_model PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_dayan_onnx_model PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_dayan_matmul tests/direct/test_dayan_matmul.cpp)
target_include_directories(test_dayan_matmul PRIVATE src)
target_link_libraries(test_dayan_matmul PRIVATE dayan_runtime_bridge)

add_executable(test_dayan_elementwise tests/direct/test_dayan_elementwise.cpp)
target_include_directories(test_dayan_elementwise PRIVATE src)
target_link_libraries(test_dayan_elementwise PRIVATE dayan_runtime_bridge)

add_executable(test_dayan_tensor_copy tests/direct/test_dayan_tensor_copy.cpp)
target_include_directories(test_dayan_tensor_copy PRIVATE src)
target_link_libraries(test_dayan_tensor_copy PRIVATE dayan_runtime_bridge)

add_executable(test_dayan_transform tests/direct/test_dayan_transform.cpp)
target_include_directories(test_dayan_transform PRIVATE src)
target_link_libraries(test_dayan_transform PRIVATE dayan_runtime_bridge)

add_executable(test_dayan_conv tests/direct/test_dayan_conv.cpp)
target_include_directories(test_dayan_conv PRIVATE src)
target_link_libraries(test_dayan_conv PRIVATE dayan_runtime_bridge)

add_executable(test_dayan_maxpool tests/direct/test_dayan_maxpool.cpp)
target_include_directories(test_dayan_maxpool PRIVATE src)
target_link_libraries(test_dayan_maxpool PRIVATE dayan_runtime_bridge)

add_executable(test_dayan_topk tests/direct/test_dayan_topk.cpp)
target_include_directories(test_dayan_topk PRIVATE src)
target_link_libraries(test_dayan_topk PRIVATE dayan_runtime_bridge)

add_executable(test_dayan_topk_onnx tests/integration/test_dayan_topk_onnx.cpp)
target_include_directories(test_dayan_topk_onnx PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_dayan_topk_onnx PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_dayan_topk_onnx PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_dayan_nms tests/direct/test_dayan_nms.cpp)
target_include_directories(test_dayan_nms PRIVATE src)
target_link_libraries(test_dayan_nms PRIVATE dayan_runtime_bridge)

add_executable(test_dayan_nms_onnx tests/integration/test_dayan_nms_onnx.cpp)
target_include_directories(test_dayan_nms_onnx PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_dayan_nms_onnx PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_dayan_nms_onnx PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_dayan_yolo_postprocess tests/integration/test_dayan_yolo_postprocess.cpp)
target_include_directories(test_dayan_yolo_postprocess PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_dayan_yolo_postprocess PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_dayan_yolo_postprocess PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_dayan_yolov8n tests/integration/test_dayan_yolov8n.cpp)
target_include_directories(test_dayan_yolov8n PRIVATE src "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_dayan_yolov8n PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_dayan_yolov8n PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

enable_testing()
add_executable(tensor_utils_test tests/unit/tensor_utils_test.cpp)
target_include_directories(tensor_utils_test PRIVATE src)
add_test(NAME tensor_utils_test COMMAND tensor_utils_test)

add_executable(cpu_reference_conv_test tests/unit/cpu_reference_conv_test.cpp)
target_include_directories(cpu_reference_conv_test PRIVATE src)
add_test(NAME cpu_reference_conv_test COMMAND cpu_reference_conv_test)

