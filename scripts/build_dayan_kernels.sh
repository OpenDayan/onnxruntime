#!/usr/bin/env bash
# 把 demo/onnx 的迁移内核编成 .o，供 onnxruntime-rvnpu 的 CMake GLOB
# （DAYAN_KERNEL_OBJECT_DIR）链接。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAYAN_ROOT="${DAYAN_ROOT:-${ROOT_DIR}/../dayan-base}"
OUT="${DAYAN_ROOT}/build/demo/onnx"
DYCC="${DAYAN_ROOT}/host/dycc/dycc"

mkdir -p "$OUT"

for src in "$DAYAN_ROOT"/demo/onnx/kernel_*.cpp; do
    name="$(basename "$src" .cpp)"
    "$DYCC" --save-temps="$OUT/$name-temps" -c "$src" -o "$OUT/$name.o"
    echo "built $name.o"
done

echo "done: $(ls "$OUT"/*.o | wc -l) objects in $OUT"
