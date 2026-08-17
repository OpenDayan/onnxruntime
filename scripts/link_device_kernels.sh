#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPU_KERNELS_ROOT="${NPU_KERNELS_ROOT:?NPU_KERNELS_ROOT is required}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:?LLVM_BUILD_DIR is required}"
DEVICE_BUILD_DIR="${DEVICE_BUILD_DIR:-${ROOT_DIR}/build/device/device_build}"

CLANG="${LLVM_BUILD_DIR}/bin/clang"
CLANGXX="${LLVM_BUILD_DIR}/bin/clang++"
LLVM_OBJCOPY="${LLVM_BUILD_DIR}/bin/llvm-objcopy"
OUTPUT_ELF="${DEVICE_BUILD_DIR}/rvnpu_kernels.elf"
OUTPUT_BIN="${DEVICE_BUILD_DIR}/rvnpu_kernels.bin"

for path in "${CLANG}" "${CLANGXX}" "${LLVM_OBJCOPY}"; do
  [[ -x "${path}" ]] || { echo "缺少 LLVM 工具：${path}" >&2; exit 2; }
done

for object in crt.o trap.o common_kernel.o syscalls.o \
              matmul_kernel_from_triton_ll.o triton_matmul_bridge.o; do
  [[ -f "${DEVICE_BUILD_DIR}/${object}" ]] || {
    echo "缺少基础 device object：${DEVICE_BUILD_DIR}/${object}" >&2
    exit 2
  }
done

target_flags=(
  -g -O3 -mcmodel=medany
  --target=riscv64-unknown-elf
  -march=rv64g -mabi=lp64d
  -fno-common -fno-builtin-printf -fno-pic -mno-relax
  -ffunction-sections -fdata-sections
  -Xclang -target-feature -Xclang +txlsu
  -Xclang -target-feature -Xclang +txmat
  -Xclang -target-feature -Xclang +txvec
  -Xclang -target-feature -Xclang +txbase
  -Xclang -target-feature -Xclang +txmacro
)

include_flags=(
  -I"${NPU_KERNELS_ROOT}/include"
  -I"${NPU_KERNELS_ROOT}/libs"
  -I"${NPU_KERNELS_ROOT}/kernel"
)

resource_dir="$("${CLANG}" -print-resource-dir)"
[[ -d "${resource_dir}/include" ]] &&
  include_flags+=(-idirafter "${resource_dir}/include")

for path in /usr/lib/picolibc/riscv64-unknown-elf/include \
            /usr/riscv64-linux-gnu/include \
            /opt/riscv-tools/riscv64-unknown-elf/include; do
  [[ -d "${path}" ]] && include_flags+=(-isystem "${path}")
done

cxx_include_flags=()
if [[ -d /usr/riscv64-linux-gnu/include/c++/12 ]]; then
  cxx_include_flags+=(
    -isystem /usr/riscv64-linux-gnu/include/c++/12
    -isystem /usr/riscv64-linux-gnu/include/c++/12/riscv64-linux-gnu
    -isystem /usr/riscv64-linux-gnu/include/c++/12/backward
  )
else
  for candidate in /opt/riscv-tools/riscv64-unknown-elf/include/c++/*; do
    if [[ -d "${candidate}" ]]; then
      cxx_include_flags+=(-isystem "${candidate}")
      [[ -d "${candidate}/riscv64-unknown-elf" ]] &&
        cxx_include_flags+=(-isystem "${candidate}/riscv64-unknown-elf")
      [[ -d "${candidate}/backward" ]] &&
        cxx_include_flags+=(-isystem "${candidate}/backward")
      break
    fi
  done
fi

"${CLANG}" -std=gnu99 "${target_flags[@]}" "${include_flags[@]}" \
  -c "${NPU_KERNELS_ROOT}/libs/math.c" \
  -o "${DEVICE_BUILD_DIR}/math.o"

kernel_sources=(
  kernel_elementwise.cpp
  kernel_silu.cpp
  kernel_sigmoid.cpp
  kernel_concat.cpp
  kernel_split.cpp
  kernel_slice.cpp
  kernel_resize.cpp
  kernel_transpose.cpp
  kernel_softmax.cpp
  kernel_reduce_max.cpp
  kernel_im2col.cpp
  kernel_maxpool.cpp
  kernel_topk.cpp
  kernel_nms.cpp
  kernel_gather_nd.cpp
)

kernel_objects=()
for source in "${kernel_sources[@]}"; do
  object="${DEVICE_BUILD_DIR}/${source%.cpp}.o"
  "${CLANGXX}" -std=c++17 "${target_flags[@]}" \
    -fno-threadsafe-statics -fno-exceptions -fno-rtti \
    "${include_flags[@]}" "${cxx_include_flags[@]}" \
    -c "${NPU_KERNELS_ROOT}/kernel/${source}" -o "${object}"
  kernel_objects+=("${object}")
done

"${CLANG}" "${target_flags[@]}" -fuse-ld=lld \
  -static -nostartfiles -nostdlib -nodefaultlibs -Wl,--gc-sections \
  -T "${NPU_KERNELS_ROOT}/libs/test.ld" \
  -o "${OUTPUT_ELF}" \
  "${DEVICE_BUILD_DIR}/crt.o" \
  "${DEVICE_BUILD_DIR}/trap.o" \
  "${DEVICE_BUILD_DIR}/common_kernel.o" \
  "${DEVICE_BUILD_DIR}/syscalls.o" \
  "${DEVICE_BUILD_DIR}/math.o" \
  "${DEVICE_BUILD_DIR}/matmul_kernel_from_triton_ll.o" \
  "${DEVICE_BUILD_DIR}/triton_matmul_bridge.o" \
  "${kernel_objects[@]}"

"${LLVM_OBJCOPY}" -O binary "${OUTPUT_ELF}" "${OUTPUT_BIN}"
echo "device ELF：${OUTPUT_ELF}"
