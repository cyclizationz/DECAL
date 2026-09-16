# DECAL

DECAL is segment-and-regenerate streaming: the server segments recurring
objects, replaces them with an encoder-friendly fill, and sends tens of bytes
of reconstruction metadata (RMD) in-band. The client composites cached RGBA
templates onto the decoded frame. A recover-only gate masks a region only
when a resident template can restore it; otherwise the region stays ordinary
video.

This tree is a **minimum, anonymous** source package for confidential review.
Models, clips, and template pools are **not** included. See `DRIVE_CONTENTS.md`.

Live network transport is **not** implemented. The production-quality path is
the offline evaluator (`decal_offline`). Online binaries are scaffolds for
artifact-contract and thin-client experiments.

## Quick start (about 15 minutes after inputs are in place)

Host class: x86-64 Linux. Paper measurements use CUDA. CPU is best-effort
(`-DUSE_CUDA=OFF` plus `--cpu`).

### 1. Packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake g++ libopencv-dev libssl-dev \
  nlohmann-json3-dev ffmpeg pkg-config curl
# Optional quality metric (not required to run the encoder):
#   sudo apt-get install -y libvmaf-dev
#   or build libvmaf and export VMAF_BIN=/path/to/vmaf
```

NVIDIA driver + CUDA toolkit are required for the default GPU ONNX Runtime
bundle.

### 2. Inputs

Unpack Drive contents to `$DECAL_DATA` (default: `./data`):

```text
$DECAL_DATA/models/fc5_seg.onnx
$DECAL_DATA/clips/fc5_00.mp4
```

```bash
export DECAL_ROOT="$PWD"
export DECAL_DATA="${DECAL_DATA:-$DECAL_ROOT/data}"
```

### 3. Build

```bash
bash scripts/download_ort.sh
cmake -S deployment -B deployment/build
cmake --build deployment/build --target decal_offline -j
```

### 4. Smoke test

```bash
mkdir -p outputs/smoke
./deployment/build/decal_offline \
  -i "$DECAL_DATA/clips/fc5_00.mp4" \
  -m "$DECAL_DATA/models/fc5_seg.onnx" \
  -o outputs/smoke \
  --latent-key --latent-thr 0.95 \
  --yolo-heal-only \
  --yolo-heal-iou 0.995 --yolo-heal-extra 0.005 \
  --yolo-heal-min-coverage 0.15 --yolo-heal-mse 700 \
  --mask-color dominant --mask-color-period 200 \
  --fill-mode solid --feather-px 4 \
  --enc-codec libx264 --enc-crf 23 --enc-preset medium \
  --enc-tune none --enc-profile none --enc-level none \
  --enc-open-gop-defaults \
  --max-frames 120 \
  --cuda
```

Outputs under `outputs/smoke/`:

| File | Meaning |
|------|---------|
| `segmented_output.mp4` | masked stream sent to the encoder |
| `msk1_payloads.bin` | RMD sidecar (SEI payload in file form) |
| `recovered_output.mp4` | client reconstruction |
| `dict/` | minted templates |
| `report.json` | byte accounting, quality, timing |

Paper-default encoder flags and the full knob list: `docs/PARAMETERS.md`.
Offline vs online split: `docs/DEPLOYMENT.md`.
Regenerating paper suites: `docs/EXPERIMENTS.md`.
