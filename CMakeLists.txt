cmake_minimum_required(VERSION 3.20)
project(onnxruntime_rvnpu VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime source or SDK root")
set(ONNXRUNTIME_LIB_DIR "" CACHE PATH "Directory containing libonnxruntime")
set(NPU_RUNTIME_ROOT "" CACHE PATH "npu-runtime repository root")
set(RVNPU_KERNEL_PACKED_C "" CACHE FILEPATH "hw-lab packed kernelset C source")

foreach(required_var ONNXRUNTIME_ROOT ONNXRUNTIME_LIB_DIR NPU_RUNTIME_ROOT)
  if(NOT ${required_var})
    message(FATAL_ERROR "${required_var} is required")
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

add_library(rvnpu_runtime_vendor STATIC
  "${NPU_RUNTIME_ROOT}/src/NPUDriver.c"
  "${NPU_RUNTIME_ROOT}/src/NPURuntime.c"
  "${NPU_RUNTIME_ROOT}/src/NPUMemoryManagement.cpp")
option(RVNPU_KERNEL_PROFILING "Enable NPU kernel execution cycle profiling" OFF)
if(RVNPU_KERNEL_PROFILING)
  target_compile_definitions(rvnpu_runtime_vendor PRIVATE NPU_ENABLE_KERNEL_PROFILING=1)
endif()
if(RVNPU_KERNEL_PACKED_C)
  target_sources(rvnpu_runtime_vendor PRIVATE "${RVNPU_KERNEL_PACKED_C}")
endif()
target_include_directories(rvnpu_runtime_vendor PUBLIC "${NPU_RUNTIME_ROOT}/include")
target_link_libraries(rvnpu_runtime_vendor PUBLIC ${CMAKE_DL_LIBS} pthread)

add_library(rvnpu_runtime_bridge STATIC
  src/runtime/device_context.cpp
  src/runtime/device_buffer.cpp)
target_include_directories(rvnpu_runtime_bridge PUBLIC src "${NPU_RUNTIME_ROOT}/include")
target_link_libraries(rvnpu_runtime_bridge PUBLIC rvnpu_runtime_vendor)

add_library(rvnpu_ep SHARED
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
target_sources(rvnpu_ep PRIVATE
  src/ops/resize/resize.cpp
  src/ops/reduce_max/reduce_max.cpp
  src/ops/slice/slice.cpp
  src/ops/softmax/softmax.cpp
  src/ops/split/split.cpp
  src/ops/transpose/transpose.cpp)
target_include_directories(rvnpu_ep PRIVATE src "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(rvnpu_ep PRIVATE rvnpu_runtime_bridge "${ONNXRUNTIME_LIBRARY}")
set_target_properties(rvnpu_ep PROPERTIES
  OUTPUT_NAME rvnpu_ep
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")
target_link_options(rvnpu_ep PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/rvnpu_ep.lds")

add_executable(test_rvnpu_tiny_cnn tools/test_rvnpu_tiny_cnn.cpp)
target_include_directories(test_rvnpu_tiny_cnn PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_rvnpu_tiny_cnn PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_rvnpu_tiny_cnn PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_rvnpu_onnx_model tools/test_rvnpu_onnx_model.cpp)
target_include_directories(test_rvnpu_onnx_model PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_rvnpu_onnx_model PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_rvnpu_onnx_model PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_rvnpu_matmul tools/test_rvnpu_matmul.cpp)
target_include_directories(test_rvnpu_matmul PRIVATE src)
target_link_libraries(test_rvnpu_matmul PRIVATE rvnpu_runtime_bridge)

add_executable(test_rvnpu_elementwise tools/test_rvnpu_elementwise.cpp)
target_include_directories(test_rvnpu_elementwise PRIVATE src)
target_link_libraries(test_rvnpu_elementwise PRIVATE rvnpu_runtime_bridge)

add_executable(test_rvnpu_tensor_copy tools/test_rvnpu_tensor_copy.cpp)
target_include_directories(test_rvnpu_tensor_copy PRIVATE src)
target_link_libraries(test_rvnpu_tensor_copy PRIVATE rvnpu_runtime_bridge)

add_executable(test_rvnpu_transform tools/test_rvnpu_transform.cpp)
target_include_directories(test_rvnpu_transform PRIVATE src)
target_link_libraries(test_rvnpu_transform PRIVATE rvnpu_runtime_bridge)

add_executable(test_rvnpu_conv_maxpool tools/test_rvnpu_conv_maxpool.cpp)
target_include_directories(test_rvnpu_conv_maxpool PRIVATE src)
target_link_libraries(test_rvnpu_conv_maxpool PRIVATE rvnpu_runtime_bridge)

add_executable(test_rvnpu_topk tools/test_rvnpu_topk.cpp)
target_include_directories(test_rvnpu_topk PRIVATE src)
target_link_libraries(test_rvnpu_topk PRIVATE rvnpu_runtime_bridge)

add_executable(test_rvnpu_topk_onnx tools/test_rvnpu_topk_onnx.cpp)
target_include_directories(test_rvnpu_topk_onnx PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_rvnpu_topk_onnx PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_rvnpu_topk_onnx PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_rvnpu_nms tools/test_rvnpu_nms.cpp)
target_include_directories(test_rvnpu_nms PRIVATE src)
target_link_libraries(test_rvnpu_nms PRIVATE rvnpu_runtime_bridge)

add_executable(test_rvnpu_nms_onnx tools/test_rvnpu_nms_onnx.cpp)
target_include_directories(test_rvnpu_nms_onnx PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_rvnpu_nms_onnx PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_rvnpu_nms_onnx PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_rvnpu_yolo_postprocess tools/test_rvnpu_yolo_postprocess.cpp)
target_include_directories(test_rvnpu_yolo_postprocess PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_rvnpu_yolo_postprocess PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_rvnpu_yolo_postprocess PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(test_rvnpu_yolov8n tools/test_rvnpu_yolov8n.cpp)
add_executable(test_rvnpu_yolo_perf tools/test_rvnpu_yolo_perf.cpp)
target_include_directories(test_rvnpu_yolo_perf PRIVATE src "${NPU_RUNTIME_ROOT}/include")
target_link_libraries(test_rvnpu_yolo_perf PRIVATE rvnpu_runtime_bridge)
target_include_directories(test_rvnpu_yolov8n PRIVATE src "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(test_rvnpu_yolov8n PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(test_rvnpu_yolov8n PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

enable_testing()
add_executable(tensor_utils_test tests/tensor_utils_test.cpp)
target_include_directories(tensor_utils_test PRIVATE src)
add_test(NAME tensor_utils_test COMMAND tensor_utils_test)

add_executable(cpu_reference_conv_test tests/cpu_reference_conv_test.cpp)
target_include_directories(cpu_reference_conv_test PRIVATE src)
add_test(NAME cpu_reference_conv_test COMMAND cpu_reference_conv_test)
