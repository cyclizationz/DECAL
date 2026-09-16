# Experiment drivers

Run from `$DECAL_ROOT` after building `decal_offline` (and online targets for
Exp6). Install `requirements-experiments.txt`. Point `$DECAL_DATA` at the Drive
unpack. Manifests may use paths relative to `$DECAL_DATA`.

```bash
export DECAL_ROOT="$PWD"
export DECAL_DATA="${DECAL_DATA:-$DECAL_ROOT/data}"
export DECAL_EVAL="${DECAL_EVAL:-$DECAL_ROOT/eval}"
python3 -m pip install -r requirements-experiments.txt
```

Copy `configs/manifest.example.json` to `$DECAL_EVAL/manifest/offline_manifest.json`
or generate one with `tools/experiments/build_manifest.py` after placing source
videos under `$DECAL_DATA/sources` and `$DECAL_DATA/clips`.

VMAF is optional. If `VMAF_BIN` is missing, pass `--skip-video-metrics` on RD
or install libvmaf. SSIM/PSNR still run via OpenCV.

## Exp1 — rate-distortion (fixed VBV)

```bash
python3 tools/experiments/run_rd_suite.py \
  --manifest "$DECAL_EVAL/manifest/offline_manifest.json" \
  --out-dir "$DECAL_EVAL/exp1" \
  --bitrate-mbps 8 12 16 20 24 \
  --clip-ids fc5_00 fm6_00
```

Reference arm: pure streaming (`encode_pure_streaming_baseline.py`).
DECAL arm: `segmented_output.mp4` + `msk1_payloads.bin`. Extra DECAL flags:
`--decal-extra-args`.

## Exp2 — template / control overhead

```bash
python3 tools/experiments/analyze_overhead.py \
  --rd-dir "$DECAL_EVAL/exp1" \
  --out-dir "$DECAL_EVAL/exp2"
```

Replays `msk1_payloads.bin` and accounts for cache/template delivery.

## Exp3 — ablations (matching, fill, feather)

```bash
python3 tools/experiments/run_ablation_suite.py \
  --manifest "$DECAL_EVAL/manifest/offline_manifest.json" \
  --out-dir "$DECAL_EVAL/exp3" \
  --clip-ids fc5_00 fm6_00 \
  --config-kinds matching \
  --exp35-encoder \
  --use-normalized-input
```

Feather sweep uses `--feather-px` 0/4/8/16. Fill and matching kinds are
selected with `--config-kinds`.

## Exp4 — generality ratios

```bash
python3 tools/experiments/analyze_generality.py \
  --manifest "$DECAL_EVAL/manifest/offline_manifest.json" \
  --rd-points "$DECAL_EVAL/exp1/rd_suite_points.csv" \
  --out-dir "$DECAL_EVAL/exp5"
```

## Exp5 — GOP / CRF23 analysis

```bash
python3 tools/experiments/run_gop_analysis_crf23.py \
  --manifest "$DECAL_EVAL/manifest/offline_manifest.json" \
  --out-root "$DECAL_EVAL/gop_analysis" \
  --clip-ids fc5_00

python3 tools/experiments/build_exp35_points_csv.py \
  --out-dir "$DECAL_EVAL/exp35_crf"

python3 tools/experiments/plot_mario_exp5_cdf.py
```

## Exp6 — resource / thin client

Build `decal_server`, `decal_client`, `decal_template_server`.

```bash
python3 tools/experiments/run_exp6_resource_curve.py
python3 tools/experiments/run_exp6_infer_worker_sweep.py
python3 tools/experiments/run_exp6_thin_client.py --out-dir "$DECAL_EVAL/exp6/thin_client"
python3 tools/experiments/build_exp6_resource_summary.py
python3 tools/experiments/measure_client_decode_baseline.py
```

Resource scripts sample `nvidia-smi` and `/usr/bin/time`. They expect CUDA.

## Spaceflight multipart

```bash
./deployment/build/decal_offline \
  -i "$DECAL_DATA/clips/spaceflight_all.mp4" \
  -m "$DECAL_DATA/models/spaceflight_seg.onnx" \
  -o "$DECAL_EVAL/spaceflight/banks/compound" \
  --latent-key --build-index --multipart-object --multipart-class 0

python3 tools/experiments/run_rd_suite.py \
  --manifest "$DECAL_EVAL/spaceflight/offline_manifest.json" \
  --out-dir "$DECAL_EVAL/spaceflight/rd" \
  --spaceflight-multipart
```

Table rebuild from completed runs: `build_evaluation_revision.py`,
`build_spaceflight_comparison.py`, `run_spaceflight_resource_benchmark.py`.

## Building a latent bank instead of uploading one

```bash
./deployment/build/decal_offline \
  -i "$DECAL_DATA/clips/fc5_00.mp4" \
  -m "$DECAL_DATA/models/fc5_seg.onnx" \
  -o "$DECAL_EVAL/banks/fc5" \
  --latent-key --build-index
```

Then pass `--latent-bank "$DECAL_EVAL/banks/fc5/dict/latent_bank.json"` and
`--latent-no-mint` on subsequent runs.
