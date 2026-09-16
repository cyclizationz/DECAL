# Drive inputs (not in this source zip)

Place these under `$DECAL_DATA` after download. Do **not** put a personal
cloud URL in the paper. For double-blind review, attach `decal-src.zip` as
HotCRP auxiliary material and keep this Drive folder private.

Gameplay footage is typically publisher-copyrighted. Keep it private. Do not
publish clips on a public Community Award URL without permission.

## Recommended layout

```text
$DECAL_DATA/
  models/
    fc5_seg.onnx
    fm6_seg.onnx
    spaceflight_seg.onnx          # paper-table only
  clips/
    fc5_00.mp4 ... fc5_04.mp4
    fm6_00.mp4 ... fm6_04.mp4
    mario_00.mp4 ... mario_04.mp4
    spaceflight_*.mp4             # optional
  templates/mario/
    brick_large_brown.png         # Mario pixel mode
  banks/                          # optional; else --build-index
    fc5/latent_bank.json
    fm6/latent_bank.json
    spaceflight/latent_bank.json
```

Relative clip paths in `configs/manifest.example.json` resolve against
`$DECAL_DATA`.

## Minimum smoke (run `decal_offline` on one learned clip)

| Drive path | Approx. size | Role |
|------------|--------------|------|
| `models/fc5_seg.onnx` | 11 MB | FC5 segmentation |
| `models/fm6_seg.onnx` | 11 MB | FM6 segmentation |
| `clips/fc5_00.mp4` | 55 MB | 30 s 1920×1080@60 |
| `clips/fm6_00.mp4` | 496 MB | 100 s 1920×1080@60 |

`fm6_00` is large; for a tiny smoke you can run `--max-frames 120` on `fc5_00`
alone.

## Paper-table regeneration

Normalized evaluation clips (1920×1080@60 learned; Mario native 960×720@30):

| Drive path | Approx. size |
|------------|--------------|
| `clips/fc5_00.mp4` … `fc5_04.mp4` | 55, 82, 85, 60, 50 MB |
| `clips/fm6_00.mp4` … `fm6_04.mp4` | 496, 469, 506, 411, 87 MB |
| `clips/mario_00.mp4` … `mario_04.mp4` | 16, 12, 22, 16, 19 MB |
| `models/spaceflight_seg.onnx` | 11 MB |
| `templates/mario/brick_large_brown.png` | 4 KB |

Optional latent-bank JSON (no PNG dump required if you `--build-index`):

| Drive path | Approx. size |
|------------|--------------|
| `banks/fc5/latent_bank.json` | 4.7 MB |
| `banks/fm6/latent_bank.json` | 97 KB |
| `banks/spaceflight/latent_bank.json` | 3.3 MB |

Optional: `onnxruntime-linux-x64-gpu-1.22.0.tgz` so reviewers need not hit GitHub.

## Do not upload

- Full experiment `record/` trees (hundreds of GB)
- Training datasets / `.pt` weights
- Pixel template `outputs/` dumps (multi-GB)
- Comparison-system dumps
- Anything with author paths, git history, or venvs
