#!/usr/bin/env bash
set -euo pipefail

: "${DAYAN_ROOT:?set DAYAN_ROOT to the dayan-base repository root}"
OUT="${DAYAN_KERNEL_OBJECT_DIR:-${DAYAN_ROOT}/build/demo/onnx}"
DYCC="${DAYAN_ROOT}/host/dycc/dycc"

mkdir -p "$OUT"

for src in "$DAYAN_ROOT"/demo/onnx/kernel_*.cpp; do
    name="$(basename "$src" .cpp)"
    "$DYCC" -I"$(dirname "$src")" --save-temps="$OUT/$name-temps" \
        -c "$src" -o "$OUT/$name.o"
    echo "built $name.o"
done

echo "done: $(ls "$OUT"/*.o | wc -l) objects in $OUT"
