# Deployment

DECAL is **segment-and-regenerate streaming** (§3). It sits above an unmodified
codec: the server removes recurring objects before encode; the client pastes
cached RGBA templates after decode. Reconstruction metadata (RMD) rides
**in-band** in the frame’s H.264 SEI. Templates travel **once**, out of band,
and the server masks a region only after the client acknowledges the template
(recover-only invariant, §1 and §4).

This package’s runnable path is the **offline evaluator** used in §7: recorded
gameplay, both arms encoded with the same libx264 settings. **Live network
streaming to players is not evaluated** (§9). Online binaries here are
scaffolds for the artifact contract and thin-client experiments.

## Deployment model (§1, §5, §9)

Each title is prepared **once**, offline:

1. Label gameplay and train a per-title YOLOv12n-seg model; export ONNX.
2. Seed the template dictionary from the same objects (pixel titles skip the
   neural model and keep NCC templates).
3. Install model + dictionary alongside the title on the server.
4. Optional `--build-index` pass over recorded play: detect and identify
   without masking or encoding, to populate the pool and latent-key bank.

At session start the provider pushes the dictionary’s most-reused templates
to the client during the loading screen (**warm cache**). Mid-session mints
follow on the asset channel; a region stays **Raw** until that template is
ACK’d. The server mirrors the client’s LRU cache.

## Two paths (Figure 3)

| Path | Titles | Front end | After the gate |
|------|--------|-----------|----------------|
| **Photoreal** | FC5, FM6, SC | YOLOv12n-seg: mask + 32-d latent key, one CUDA inference pass | identical |
| **Pixel** | Mario | NCC template matching + Kalman + optical flow | identical |

Later stages on both paths: reliability gate → dominant-color fill + 4 px
feather → libx264 with RMD in SEI → client decode, RMD parse, one
alpha-composite per region. The client has **no learned decoder and no GPU
requirement** (R3).

A **Rec** frame contains masked regions and region records. A **Raw** frame
is displayed as decoded. RMD layout is Figure 4 (header + 14-byte region
records; pixel path adds grid offsets). Offline runs dump the same payload
as `msk1_payloads.bin`.

## Binaries

Build directory: `deployment/build`.

| Target | Role |
|--------|------|
| `decal_offline` | Offline evaluator (server + client in one process). **§7 paper path.** |
| `decal_baseline` | Unmasked baseline encoder (masking and RMD off) |
| `decal_duo` | Two-channel comparison evaluator |
| `decal_server` | Scaffold: evaluator with server artifact layout |
| `decal_client` | Scaffold: validates Rec/Raw artifacts and cache policy |
| `decal_template_server` | Localhost out-of-band template bytes (thin-client Exp6) |

```bash
bash scripts/download_ort.sh
cmake -S deployment -B deployment/build          # CUDA / paper
# cmake -S deployment -B deployment/build -DUSE_CUDA=OFF   # best-effort CPU
cmake --build deployment/build -j
```

```bash
bash deployment/client/setup_env.sh
```

Paper implementation notes (§6): FFmpeg + libx264; ONNX Runtime with CUDA
streams at **pipeline depth 4**; `--build-index` for dictionary population.
Pass `--server-pipeline 1 --server-pipeline-depth 4` on `decal_server` when
matching the timed server path.

## Offline evaluator (supported, §7)

`decal_offline` runs the full pipeline on a file and writes `report.json`.
Required: `-i <video>`. Photoreal needs `-m <onnx> --cuda`. Pixel path: `-p -T`.

**Constant-quality command** (paper shipped flags; see `docs/PARAMETERS.md`):

```bash
./deployment/build/decal_offline \
  -i "$DECAL_DATA/clips/fc5_00.mp4" \
  -m "$DECAL_DATA/models/fc5_seg.onnx" \
  -o outputs/cq \
  --latent-key --latent-thr 0.95 \
  --yolo-heal-only \
  --yolo-heal-iou 0.995 --yolo-heal-extra 0.005 \
  --yolo-heal-min-coverage 0.15 --yolo-heal-mse 700 \
  --mask-color dominant --mask-color-period 200 \
  --fill-mode solid --feather-px 4 \
  --enc-codec libx264 --enc-crf 23 --enc-preset medium \
  --enc-tune none --enc-profile none --enc-level none \
  --enc-open-gop-defaults \
  --cuda
```

Rate-capped regime: same DECAL flags, replace CRF with
`--enc-bitrate-mbps {8,12,16,20,24}` (`maxrate = bitrate`, `bufsize = 2×`).

Outputs:

| File | Meaning |
|------|---------|
| `segmented_output.mp4` | what the encoder sees (fill where Rec) |
| `msk1_payloads.bin` | RMD sidecar (SEI contents in file form) |
| `recovered_output.mp4` | client reconstruction |
| `original_output.mp4` | baseline-style unmasked encode of the source |
| `dict/` | templates (RGBA) |
| `report.json` | bytes, Rec/Raw, gate, timing |

Net delivered bytes = masked video + RMD + delivered template PNGs.
Warm-cache accounting omits the template term (§8.1).

## Online scaffolds (not a live testbed)

`decal_server` writes the same artifacts. `decal_client` checks Rec/Raw
policy. `decal_template_server` is the out-of-band asset channel on
localhost.

There is **no** live RTMP/FLV testbed, in-network ACK loop, or stall
measurement in this package. §9 states that DECAL has not yet streamed to
players over a live network; that is future work. Exp6 uses these scaffolds
plus `/usr/bin/time` and `nvidia-smi` on 300-frame CRF 23 clips.

```bash
./deployment/build/decal_server --input CLIP --model MODEL --output OUT \
  --latent-key --yolo-heal-only --cache-mode warm \
  --server-pipeline 1 --server-pipeline-depth 4 ...
./deployment/build/decal_template_server --dict OUT/dict --port 19061
./deployment/build/decal_client \
  --input CLIENT_INPUT --cache-mode warm \
  --template-server 127.0.0.1:19061
```

`tools/experiments/run_exp6_thin_client.py` orchestrates the pair.

## Dependencies

- CMake ≥ 3.5, g++ (C++17), OpenCV 4, OpenSSL, FFmpeg/ffprobe with libx264
- `nlohmann/json` (`nlohmann-json3-dev`)
- ONNX Runtime **GPU 1.22.0** (`scripts/download_ort.sh`) and CUDA for the
  photoreal paper path (~5 ms/frame detector core, §5)
- Optional: `libvmaf` (`VMAF_BIN`) for VMAF; SSIM still runs via OpenCV

## Failure mode

If identity matching, the gate, or template residency fails, the region is
left unmasked (**Raw**). Compression may suffer; the displayed frame is still
ordinary decoded video (R2).
