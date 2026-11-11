#!/usr/bin/env bash
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
