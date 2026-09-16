#!/usr/bin/env python3
"""Assemble Exp3/Exp5 intake CSV from CRF23/open-GOP exploratory runs (not Exp1 VBV)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import REPO_ROOT, EVAL_DIR, ensure_dir, load_manifest, python_bin

CRF_RATE_LABEL = 23.0


@dataclass(frozen=True)
class IntakeSpec:
    clip_id: str
    game: str
    rate_point_mbps: float
    decal_dir: Path
    pure_dir: Path
    encoder_note: str


def ffprobe_duration_s(video: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(video),
        ],
        text=True,
    ).strip()
    return float(out or 0.0)


def bitrate_bps(video: Path) -> float:
    dur = ffprobe_duration_s(video)
    return (video.stat().st_size * 8.0) / dur if dur > 1e-9 else 0.0


def delivered_bps(run_dir: Path) -> tuple[float, float, float]:
    video = run_dir / "segmented_output.mp4"
    meta = run_dir / "msk1_payloads.bin"
    video_bps = bitrate_bps(video) if video.exists() else 0.0
    meta_bps = 0.0
    if meta.exists() and video.exists():
        dur = ffprobe_duration_s(video)
        if dur > 1e-9:
            meta_bps = (meta.stat().st_size * 8.0) / dur
    return video_bps, meta_bps, video_bps + meta_bps


def run_build_per_frame(out_dir: Path) -> None:
    if not (out_dir / "report.json").exists():
        raise FileNotFoundError(f"missing report.json in {out_dir}")
    subprocess.run(
        [
            python_bin(),
            str(REPO_ROOT / "tools/metrics/build_per_frame_csv.py"),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )


def ensure_pure_from_original(decal_dir: Path, pure_dir: Path) -> None:
    """Pure baseline = original encode only (no masking, empty MSK1)."""
    ensure_dir(pure_dir)
    orig = decal_dir / "original_output.mp4"
    if not orig.exists():
        raise FileNotFoundError(f"missing {orig}")
    for name in ("original_output.mp4", "segmented_output.mp4", "recovered_output.mp4"):
        dst = pure_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(orig.resolve(), dst)
    msk1 = pure_dir / "msk1_payloads.bin"
    msk1.write_bytes(b"")
    rep_src = decal_dir / "report.json"
    rep = json.loads(rep_src.read_text())
    rep["pure_streaming_baseline"] = True
    (pure_dir / "report.json").write_text(json.dumps(rep), encoding="utf-8")


def ensure_per_frame(decal_dir: Path, pure_dir: Path) -> None:
    if not (decal_dir / "per_frame_metrics.csv").exists():
        run_build_per_frame(decal_dir)
    ensure_pure_from_original(decal_dir, pure_dir)
    if not (pure_dir / "per_frame_metrics.csv").exists():
        run_build_per_frame(pure_dir)


def fc5_specs() -> list[IntakeSpec]:
    out: list[IntakeSpec] = []
    for i in range(5):
        clip_id = f"fc5_{i:02d}"
        decal = EVAL_DIR / "gop_analysis" / f"{clip_id}_offline_open_x264_crf23"
        pure = EVAL_DIR / "exp35_crf" / clip_id / "crf23/pure_streaming"
        out.append(
            IntakeSpec(
                clip_id=clip_id,
                game="fc5",
                rate_point_mbps=CRF_RATE_LABEL,
                decal_dir=decal,
                pure_dir=pure,
                encoder_note="CRF23 open x264 (gop_analysis)",
            )
        )
    return out


def mario_specs() -> list[IntakeSpec]:
    out: list[IntakeSpec] = []
    for i in range(5):
        clip_id = f"mario_{i:02d}"
        decal = EVAL_DIR / "gop_analysis" / f"{clip_id}_offline_open_x264_crf23"
        pure = EVAL_DIR / "exp35_crf" / clip_id / "crf23/pure_streaming"
        out.append(
            IntakeSpec(
                clip_id=clip_id,
                game="mario",
                rate_point_mbps=CRF_RATE_LABEL,
                decal_dir=decal,
                pure_dir=pure,
                encoder_note="CRF23 open x264 pixel pipeline (gop_analysis)",
            )
        )
    return out


def fm6_specs(
    *,
    decal_root: Path = EVAL_DIR / "gop_analysis",
    pure_root: Path = EVAL_DIR / "exp35_crf",
) -> list[IntakeSpec]:
    out: list[IntakeSpec] = []
    for i in range(5):
        clip_id = f"fm6_{i:02d}"
        decal = decal_root / f"{clip_id}_offline_open_x264_crf23"
        pure = pure_root / clip_id / "crf23/pure_streaming"
        out.append(
            IntakeSpec(
                clip_id=clip_id,
                game="fm6",
                rate_point_mbps=CRF_RATE_LABEL,
                decal_dir=decal,
                pure_dir=pure,
                encoder_note="CRF23 open x264 (gop_analysis)",
            )
        )
    return out


def spaceflight_specs(*, decal_root: Path, pure_root: Path) -> list[IntakeSpec]:
    return [
        IntakeSpec(
            clip_id=f"spaceflight_{i:02d}",
            game="spaceflight",
            rate_point_mbps=CRF_RATE_LABEL,
            decal_dir=decal_root / f"spaceflight_{i:02d}_offline_open_x264_crf23",
            pure_dir=pure_root / f"spaceflight_{i:02d}" / "crf23/pure_streaming",
            encoder_note="CRF23 open x264, heal-only multipart class 0 (gop_analysis)",
        )
        for i in range(5)
    ]


def row_for(
    spec: IntakeSpec,
    *,
    variant: str,
    pure_bps: tuple[float, float, float],
    decal_bps: tuple[float, float, float],
    run_dir: Path,
    pure_dir: Path,
    decal_dir: Path,
) -> dict[str, Any]:
    pb_v, pb_m, pb_t = pure_bps
    rs_v, rs_m, rs_t = decal_bps
    if variant == "pure_streaming":
        vo, mo, tot = pb_v, pb_m, pb_t
    else:
        vo, mo, tot = rs_v, rs_m, rs_t
    return {
        "clip_id": spec.clip_id,
        "game": spec.game,
        "rate_point_mbps": spec.rate_point_mbps,
        "variant": variant,
        "achieved_bps": tot,
        "achieved_mbps": tot / 1_000_000.0,
        "video_only_bps": vo,
        "meta_bps": mo,
        "vmaf_mean": "",
        "vmaf_p10": "",
        "ssim_mean": "",
        "psnr_mean": "",
        "roi_vmaf_mean": "",
        "roi_ssim_mean": "",
        "roi_psnr_mean": "",
        "roi_frame_count": "",
        "baseline_achieved_bps": pb_t,
        "decal_achieved_bps": rs_t,
        "baseline_vmaf_mean": "",
        "decal_vmaf_mean": "",
        "baseline_ssim_mean": "",
        "decal_ssim_mean": "",
        "baseline_psnr_mean": "",
        "decal_psnr_mean": "",
        "baseline_run_dir": str(pure_dir),
        "decal_run_dir": str(decal_dir),
        "stitch_full_vmaf_mean": "",
        "stitch_full_ssim_mean": "",
        "stitch_full_psnr_mean": "",
        "stitch_roi_vmaf_mean": "",
        "stitch_roi_ssim_mean": "",
        "stitch_roi_psnr_mean": "",
        "frame_mode_hysteresis_pre_mean_run_length": "",
        "frame_mode_hysteresis_post_mean_run_length": "",
        "frame_mode_hysteresis_pre_switch_fraction": "",
        "frame_mode_hysteresis_post_switch_fraction": "",
        "run_dir": str(run_dir),
        "encoder_note": spec.encoder_note,
    }


def build_specs(
    *,
    include_fm6: bool,
    skip_fc5: bool,
    skip_mario: bool,
    fm6_decal_root: Path,
    fm6_pure_root: Path,
    include_spaceflight: bool,
    spaceflight_decal_root: Path,
    spaceflight_pure_root: Path,
) -> list[IntakeSpec]:
    specs: list[IntakeSpec] = []
    if not skip_fc5:
        specs.extend(fc5_specs())
    if not skip_mario:
        specs.extend(mario_specs())
    if include_fm6:
        specs.extend(
            fm6_specs(
                decal_root=fm6_decal_root,
                pure_root=fm6_pure_root,
            )
        )
    if include_spaceflight:
        specs.extend(
            spaceflight_specs(
                decal_root=spaceflight_decal_root,
                pure_root=spaceflight_pure_root,
            )
        )
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Exp3/Exp5 CRF exploratory intake CSV.")
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR / "exp35_crf")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--include-fm6", action="store_true", help="Include FM6 gop_analysis CRF23 dirs.")
    ap.add_argument("--include-spaceflight", action="store_true")
    ap.add_argument("--manifest", type=Path, default=EVAL_DIR / "manifest/offline_manifest.json")
    ap.add_argument("--extra-manifest", type=Path, action="append", default=[])
    ap.add_argument("--spaceflight-decal-root", type=Path, default=EVAL_DIR / "spaceflight_v2/gop_analysis")
    ap.add_argument("--spaceflight-pure-root", type=Path, default=None)
    ap.add_argument(
        "--fm6-decal-root",
        type=Path,
        default=EVAL_DIR / "gop_analysis",
    )
    ap.add_argument("--fm6-pure-root", type=Path, default=None)
    ap.add_argument("--skip-fc5", action="store_true")
    ap.add_argument("--skip-mario", action="store_true")
    ap.add_argument("--require-complete", action="store_true", help="Fail if any decal dir is missing.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    out_csv = args.out_csv or (args.out_dir / "exp35_points.csv")
    manifest = {c.clip_id: c for c in load_manifest(args.manifest)}
    for manifest_path in args.extra_manifest:
        manifest.update({c.clip_id: c for c in load_manifest(manifest_path)})

    rows: list[dict[str, Any]] = []
    for spec in build_specs(
        include_fm6=args.include_fm6,
        skip_fc5=args.skip_fc5,
        skip_mario=args.skip_mario,
        fm6_decal_root=args.fm6_decal_root,
        fm6_pure_root=args.fm6_pure_root or args.out_dir,
        include_spaceflight=args.include_spaceflight,
        spaceflight_decal_root=args.spaceflight_decal_root,
        spaceflight_pure_root=args.spaceflight_pure_root or args.out_dir,
    ):
        if spec.clip_id not in manifest:
            print(f"[skip] unknown clip {spec.clip_id}", file=sys.stderr)
            continue
        if not spec.decal_dir.is_dir():
            msg = f"missing decal dir {spec.decal_dir}"
            if args.require_complete:
                raise FileNotFoundError(msg)
            print(f"[skip] {msg}", file=sys.stderr)
            continue
        if spec.game == "mario" and not spec.pure_dir.is_dir() and not spec.decal_dir.is_dir():
            msg = f"missing mario decal dir {spec.decal_dir}"
            if args.require_complete:
                raise FileNotFoundError(msg)
            print(f"[skip] {msg}", file=sys.stderr)
            continue

        if spec.game in ("fc5", "fm6", "mario", "spaceflight"):
            ensure_per_frame(spec.decal_dir, spec.pure_dir)
        else:
            if not (spec.decal_dir / "per_frame_metrics.csv").exists():
                run_build_per_frame(spec.decal_dir)
            if not (spec.pure_dir / "per_frame_metrics.csv").exists():
                run_build_per_frame(spec.pure_dir)

        pure_bps = delivered_bps(spec.pure_dir)
        decal_bps = delivered_bps(spec.decal_dir)
        rows.append(
            row_for(
                spec,
                variant="pure_streaming",
                pure_bps=pure_bps,
                decal_bps=decal_bps,
                run_dir=spec.pure_dir,
                pure_dir=spec.pure_dir,
                decal_dir=spec.decal_dir,
            )
        )
        rows.append(
            row_for(
                spec,
                variant="decal",
                pure_bps=pure_bps,
                decal_bps=decal_bps,
                run_dir=spec.decal_dir,
                pure_dir=spec.pure_dir,
                decal_dir=spec.decal_dir,
            )
        )
        print(f"[ok] {spec.clip_id} pure={pure_bps[2]/1e6:.2f} Mbps decal={decal_bps[2]/1e6:.2f} Mbps")

    if not rows:
        raise SystemExit("no intake rows written")

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    protocol = args.out_dir / "exp35_protocol.md"
    protocol.write_text(
        "\n".join(
            [
                "# Exp3/Exp5 intake protocol (CRF / exploratory)",
                "",
                "- Exp1 RD suite remains fixed VBV under `eval/exp1/`.",
                "- This CSV feeds Exp3 overhead and Exp5 generality only.",
                "",
                "## Sources",
                f"- FC5: `gop_analysis/fc5_XX_offline_open_x264_crf23` (CRF 23, open GOP)",
                "- Mario: `gop_analysis/mario_XX_offline_open_x264_crf23` (CRF 23, open GOP, pixel pipeline)",
                "- FM6: `gop_analysis/fm6_XX_offline_open_x264_crf23` (CRF 23, open GOP, heal-only)",
                "- Spaceflight: `spaceflight_v2/gop_analysis/spaceflight_XX_offline_open_x264_crf23` (CRF 23, open GOP, heal-only multipart class 0)",
                "",
                f"- Points CSV: `{out_csv}`",
                f"- Rows: {len(rows)} ({len(rows)//2} clip pairs)",
                "",
            ]
        )
        + "\n"
    )
    print(json.dumps({"points_csv": str(out_csv), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
