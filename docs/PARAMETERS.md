# Environment and parameters

CLI binaries ship conservative C++ defaults. **Paper results use the shipped
configuration below**, which you must pass explicitly. Section numbers refer
to the DECAL paper.

## Paper shipped configuration

Photoreal path (FC5, FM6, SC): YOLOv12n-seg → latent-key identity → recover-only
reliability gate → dominant-color fill with a 4 px feather → libx264.
Pixel path (Mario): NCC template matching + Kalman + Lucas–Kanade flow; later
stages are the same.

```bash
# Photoreal, constant-quality regime (§7)
./deployment/build/decal_offline \
  -i CLIP -m MODEL -o OUT \
  --latent-key --latent-thr 0.95 \
  --yolo-heal-only \
  --yolo-heal-iou 0.995 --yolo-heal-extra 0.005 \
  --yolo-heal-min-coverage 0.15 --yolo-heal-mse 700 \
  --yolo-heal-max-shift 8 \
  --mask-color dominant --mask-color-period 200 \
  --fill-mode solid --feather-px 4 \
  --enc-codec libx264 --enc-crf 23 --enc-preset medium \
  --enc-tune none --enc-profile none --enc-level none \
  --enc-open-gop-defaults \
  --cuda
```

SC cockpit compound regions add `--multipart-object --multipart-class 0`
`--multipart-min-components 2 --multipart-geometry-tol 0.12`.

Headline byte savings in the paper use a **warm cache**: referenced templates
are already resident, so net BSP does not charge first-delivery template
bytes (§8.1).

## Two encoder regimes (§7)

The regimes answer different questions and are never mixed into one bandwidth
number.

| Regime | Encoder | What it measures |
|--------|---------|------------------|
| **Constant quality** | libx264 CRF 23, preset `medium`, open GOP, **zero B-frames**, no fixed GOP length | fewer delivered bytes at a quality target |
| **Rate-capped (VBV)** | 8, 12, 16, 20, 24 Mbps; `maxrate = bitrate`, `bufsize = 2× bitrate` | whether masking buys quality when both arms fill a cap |

Observed baseline keyframe intervals under CRF 23 are the evaluation GOPs
(about 25–214 frames on FC5, 25–250 on FM6 and Mario).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `DECAL_ROOT` | directory containing `deployment/` | repository root |
| `DECAL_DATA` | `$DECAL_ROOT/data` | models, clips, banks, templates |
| `DECAL_EVAL` | `$DECAL_ROOT/eval` | experiment outputs |
| `DECAL_MODEL` | `$DECAL_DATA/models/seg.onnx` | default ONNX path for Python drivers |
| `DECAL_PYTHON` | `python3` or `$DECAL_ROOT/.venv/bin/python` | interpreter for drivers |
| `TM_TEMPLATE_PATH` | unset | pixel-path template file; directory is used if `-T` is omitted |
| `VMAF_BIN` | `vmaf` on `PATH` | libvmaf CLI |
| `VMAF_TMP` | `/tmp/decal_vmaf` | y4m scratch (use a disk path, not tmpfs, for long clips) |
| `LD_LIBRARY_PATH` | — | include `deployment/onnxruntime-linux-x64-gpu-1.22.0/lib` if needed |

Python drivers resolve roots from these variables (`tools/experiments/common.py`).

## Identity matching (§4, Figure 5)

Latent key: 32-d L2-normalized mask-coefficient vector from one YOLOv12n-seg
pass. Cosine similarity against the same-class bank. Temporal fast path reuses
the previous same-class instance when box IoU ≥ 0.7 and cosine ≥ τcos − 0.01.

| Quantity | Paper | Flag / notes |
|----------|-------|----------------|
| τcos (reuse) | **0.95** | `--latent-thr 0.95` with `--latent-key` |
| Temporal IoU | 0.7 | implementation |
| Motion mint-boost | box IoU < 0.6, center shift > 20 px, or sharp area-ratio change | `--latent-motion-iou 0.6`, `--latent-motion-center 20`, `--latent-motion-scale 0.25` |
| Mint-boost window | a few frames | `--latent-motion-boost 6` |
| Periodic refresh | every N frames | `--latent-period 2` |
| τskip, τmerge | τskip > τmerge > τcos | `--latent-period-skip`, `--latent-merge` (C++ leaves these disabled unless set) |

Ablation-only matchers (`--yolo-matcher phash|iou_only|rgb_hist`): Table 2.
The reliability gate still sends original pixels when they fail.

## Reliability gate (§4, Figure 6)

`--yolo-heal-only` is the recover-only invariant: the server masks a region
only if a **resident** template can restore it. Failures become **Raw**
(original pixels), never a broken picture.

Four checks, in order:

| Check | Paper threshold | Flag | C++ CLI default |
|-------|-----------------|------|-----------------|
| 1 Resident | template ACK’d on the client | implied by heal-only / cache mode | — |
| 2 Appearance | aligned masked RGB MSE ≤ τmse | `--yolo-heal-mse` | 700 |
| 3 Coverage | aligned alpha covers ≥ τcov of the detected mask | `--yolo-heal-min-coverage` | 0.15 |
| 4 Geometry | IoU ≥ **0.995**, spill ≤ **0.005** | `--yolo-heal-iou`, `--yolo-heal-extra` | 0.0 (off), 0.02 |

Pass `--yolo-heal-iou 0.995 --yolo-heal-extra 0.005` to match the paper.
Alignment search: `--yolo-heal-max-shift 8` (px). If the primary template
fails, the server tries other same-class bank entries above τcos, then the
most similar template used in roughly the past GOP (`--yolo-heal-fallback-window 24`,
`--yolo-heal-fallback-minsim 0.85`).

A frame with no passing region is sent **Raw**. Rec frames carry region
records in the RMD.

## Mask, fill, feather (§4, Table 3)

Shipped fill is **dominant color**, re-estimated every **200** frames. Fill
is written only inside the mask. A **4 px** feather band inside the mask
edge is the Table 3 default (0/8/16 px are ablations). Black fill and
inpainting are ablation-only.

| Flag | C++ default | Paper shipped |
|------|-------------|----------------|
| `--mask-color` | green | **dominant** |
| `--mask-color-period` | 200 | **200** |
| `--fill-mode` | solid | **solid** (with dominant color) |
| `--feather-px` | 0 | **4** |
| `--yolo-force-mask-all` | off | off (breaks recovery; do not use) |

## SC compound regions (§4)

SC’s cockpit arrives as several same-class components. When at least two
confident components are present they merge into one compound region (union
box, alpha-union mask, area-weighted latent key). Enabled only for SC’s
cockpit class.

| Flag | Paper |
|------|-------|
| `--multipart-object` | on for SC |
| `--multipart-class` | **0** |
| `--multipart-min-components` | **2** |
| `--multipart-geometry-tol` | **0.12** RMS |

## Encoder

| Flag | C++ default | Constant-quality (paper) | Rate-capped (paper) |
|------|-------------|--------------------------|---------------------|
| `--enc-codec` | libx264 | libx264 | libx264 |
| `--enc-crf` | 18 | **23** | unused |
| `--enc-bitrate-mbps` | 0 | unused | **8 / 12 / 16 / 20 / 24** |
| `--enc-maxrate-mbps` | = bitrate | — | = bitrate |
| `--enc-bufsize-mbits` | 2× bitrate | — | 2× bitrate |
| `--enc-preset` | superfast | **medium** | **medium** |
| `--enc-tune` | zerolatency | **none** | **none** |
| B-frames | `encBFrames=0` (no CLI flag) | **0** | **0** |
| `--enc-open-gop-defaults` | off | **on** (no fixed GOP length; B-frames stay 0) | as used in the VBV protocol |
| `--enc-profile` / `--enc-level` | baseline / 4.2 | **none** | **none** |

`--no-ffmpeg-enc` is a debug fallback (OpenCV writer), not used in the paper.

## Pixel path (Mario, §4)

Replaces segmentation only. NCC against the dictionary every few frames;
Kalman smooths boxes; Lucas–Kanade flow carries boxes between matching
passes. Identity is the dictionary template (no bank mint). Repeated tiles
use one `template_id` plus integer grid offsets (pixel-path RMD).

| Flag | C++ default | Paper Mario |
|------|-------------|-------------|
| `-p` / `--pixel` | off | **on** |
| `-T` | `TM_TEMPLATE_PATH` or `.` | dictionary directory |
| `--pixel-force-scale` | 0 (auto) | **1.0** (unscaled stamps) |
| `--pixel-kalman` / `--pixel-flow` | 1 | **on** |
| `--pixel-grid-header` | off | **on** |
| `--pixel-max-peaks` | 30 | 90 |
| `--pixel-bootstrap` | 30 | 15 |
| `--pixel-thr-k` | 0.85 | 0.80 |

Full Mario flag list: `tools/experiments/pixel_mario_defaults.py` and
`deployment/docs/pixel_kalman_flow.md`.

## Workloads (§7)

Five clips per title, identical across experiments.

| Title | Recurring object | Geometry / rate | Notes |
|-------|------------------|-----------------|-------|
| FC5 | weapon | 960×540 crop, 3000 frames, 60 fps | photoreal |
| FM6 | dashboard | 1920×1080, 3965 frames, 60 fps | photoreal |
| Mario | sprite and bricks | 1280×720, 3000 frames, 60 fps | pixel path |
| SC | cockpit | 1280×720, 3000 frames, 30.303 fps | compound regions |

Mean mask area: FC5 18 %, FM6 38 %, Mario 12 %, SC 31 %.

## Accounting and quality (§7)

- **BSP** = 100 (1 − B/B0). Net BSP charges masked video + measured RMD bytes
  + delivered template PNG sizes.
- Warm-cache runs charge zero template-delivery bytes.
- SSIM and VMAF score **every fifth frame** at the **540-p** VMAF scale,
  both arms against the original pre-encode source.
- Offline evaluation uses a sidecar dump of the RMD (`msk1_payloads.bin`)
  in place of in-band SEI; the wire layout is Figure 4.

## Client substitution (§6)

On a rare cache miss the client may substitute its newest same-`class_id`
template if that template’s last box overlaps the masked box with IoU ≥ 0.7.
Photoreal templates are resized to `(w, h)` by nearest-neighbor. Pixel-path
templates are stamped unscaled at grid offsets.

## Server pipeline (implementation §6, timing §8.5)

Paper implementation: ONNX Runtime CUDA, overlapping preprocess / infer /
post at **pipeline depth 4**. Timing clips are 300 frames at CRF 23.

| Flag | C++ default | Paper implementation |
|------|-------------|----------------------|
| `--server-pipeline` | off | **on** for timed server runs |
| `--server-pipeline-depth` | 2 | **4** |
| `--server-infer-workers` | 1 | 1 (sweep is ablation) |
| `--cache-mode` | cold | **warm** for headline BSP; `partial-warm` for delivery experiments |

`--yolo-force-mask-all`, matcher/fill/feather sweeps, and infer-worker
counts are ablation-only.

## I/O flags

| Flag | Default | Notes |
|------|---------|-------|
| `-i` | required | input video |
| `-o` | `../outputs` | run directory |
| `-m` | `models/seg.onnx` | ONNX model (omit on pixel path) |
| `--max-frames` | unlimited | 300 for paper timing; cap for smokes |
| `-d` / `--cuda` | off in C++ CLI | **on** for photoreal paper path |
| `--cpu` | — | CPU ONNX (not the paper path) |
| `--build-index` | off | populate dictionary + latent bank, no mask/encode (§6) |
| `--latent-bank` | `<dict>/latent_bank.json` if present | load a prepared bank |
| `--latent-no-mint` | off | match-only after `--build-index` |
| `--timing` | off | per-frame timing in `report.json` |
