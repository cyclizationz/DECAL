# Protocol notes

Anonymous rewrite of the §7 evaluation recipe. Hardware is described by class.

## Host class

x86-64 Linux, CUDA GPU for photoreal ONNX, FFmpeg/libx264. Quality: SSIM and
VMAF on every fifth frame at the 540-p VMAF scale. Timing: 300-frame CRF 23
clips, `/usr/bin/time` and `nvidia-smi`.

## Encoder regimes (§7)

Constant quality:

```text
libx264  CRF 23  preset medium  open GOP  B-frames 0  no fixed GOP length
```

Rate-capped VBV: 8 / 12 / 16 / 20 / 24 Mbps, `maxrate = bitrate`,
`bufsize = 2× bitrate`. Do not combine the two regimes into one BSP.

## Shipped photoreal flags

`--latent-key --latent-thr 0.95 --yolo-heal-only --yolo-heal-iou 0.995
--yolo-heal-extra 0.005 --mask-color dominant --mask-color-period 200
--fill-mode solid --feather-px 4`. SC adds `--multipart-object
--multipart-class 0`. Mario uses the pixel path (`-p`), not YOLO.

## Workloads

Five clips per title. FC5: 960×540 weapon crop, 3000 frames, 60 fps. FM6:
1920×1080, 3965 frames, 60 fps. Mario: 1280×720, 3000 frames, 60 fps. SC:
1280×720, 3000 frames, 30.303 fps.

## Accounting

Baseline: identical source and encoder with masking and RMD off.
DECAL net bytes: masked video + RMD + delivered template PNGs.
Warm cache: template-delivery term is zero. Quality is scored against the
original pre-encode source.

## What this package excludes

Build trees, generated videos, ONNX/ORT blobs, template pools, live transport.
GRACE comparison harnesses are not in this min-repro. Drive inputs are listed
in `DRIVE_CONTENTS.md`.
