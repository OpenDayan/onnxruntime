cmake_minimum_required(VERSION 3.20)
project(onnxruntime_rvnpu VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime source or SDK root")
set(ONNXRUNTIME_LIB_DIR "" CACHE PATH "Directory containing libonnxruntime")
set(NPU_RUNTIME_ROOT "" CACHE PATH "npu-runtime repository root")

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
  src/ops/matmul/matmul.cpp)
target_include_directories(rvnpu_ep PRIVATE src "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(rvnpu_ep PRIVATE rvnpu_runtime_bridge "${ONNXRUNTIME_LIBRARY}")
set_target_properties(rvnpu_ep PROPERTIES
  OUTPUT_NAME rvnpu_ep
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")
target_link_options(rvnpu_ep PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/rvnpu_ep.lds")

add_executable(rvnpu_ort_runner tools/rvnpu_ort_runner.cpp)
target_include_directories(rvnpu_ort_runner PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(rvnpu_ort_runner PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(rvnpu_ort_runner PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

add_executable(rvnpu_matmul_probe tools/rvnpu_matmul_probe.cpp)
target_include_directories(rvnpu_matmul_probe PRIVATE src)
target_link_libraries(rvnpu_matmul_probe PRIVATE rvnpu_runtime_bridge)

enable_testing()
add_executable(tensor_utils_test tests/tensor_utils_test.cpp)
target_include_directories(tensor_utils_test PRIVATE src)
add_test(NAME tensor_utils_test COMMAND tensor_utils_test)
