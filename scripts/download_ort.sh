#!/usr/bin/env bash
set -euo pipefail

# Download ONNX Runtime GPU 1.22.0 into deployment/, matching CMakeLists.txt.
# Paper path: CUDA GPU bundle. CPU-only CMake (-DUSE_CUDA=OFF) still expects this
# directory layout today; treat CPU builds as best-effort.

ORT_VERSION="1.22.0"
ORT_NAME="onnxruntime-linux-x64-gpu-${ORT_VERSION}"
ORT_TGZ="${ORT_NAME}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TGZ}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REPO_ROOT}/deployment"
mkdir -p "${DEST}"
cd "${DEST}"

if [[ -d "${ORT_NAME}" ]]; then
  echo "[ok] ${ORT_NAME} already present under ${DEST}"
  exit 0
fi

echo "[download] ${ORT_URL}"
curl -L --fail -o "${ORT_TGZ}" "${ORT_URL}"
tar -xzf "${ORT_TGZ}"
rm -f "${ORT_TGZ}"
echo "[ok] extracted ${ORT_NAME}"
