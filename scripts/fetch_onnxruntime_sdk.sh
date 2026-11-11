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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="1.26.0"
COMMIT="8c546c37b43caaca1fa25db430dab94b901cf277"
ARCHIVE_SHA256="1254da24fb389cf39dc0ff3451ab48301740ffbfcbaf646849df92f80ee92c57"
ARCHIVE="${ROOT_DIR}/build/downloads/onnxruntime-linux-x64-${VERSION}.tgz"
SDK_DIR="${ROOT_DIR}/build/ort-sdk"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-x64-${VERSION}.tgz"

mkdir -p "$(dirname "${ARCHIVE}")"
if [[ ! -f "${ARCHIVE}" ]]; then
  curl -L --fail --retry 3 --output "${ARCHIVE}" "${URL}"
fi
echo "${ARCHIVE_SHA256}  ${ARCHIVE}" | sha256sum --check

if [[ ! -f "${SDK_DIR}/lib/libonnxruntime.so" ]]; then
  mkdir -p "${SDK_DIR}"
  tar -xzf "${ARCHIVE}" -C "${SDK_DIR}" --strip-components=1
fi

[[ "$(<"${SDK_DIR}/VERSION_NUMBER")" == "${VERSION}" ]] || {
  echo "unexpected ONNX Runtime SDK version" >&2
  exit 1
}
[[ "$(<"${SDK_DIR}/GIT_COMMIT_ID")" == "${COMMIT}" ]] || {
  echo "unexpected ONNX Runtime SDK commit" >&2
  exit 1
}
echo "ONNX Runtime SDK ${VERSION}@${COMMIT} is ready at ${SDK_DIR}"

