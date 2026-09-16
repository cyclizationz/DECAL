#!/usr/bin/env python3
"""Run offline open-x264 CRF23 exploratory encodes for gop_analysis."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import (
    DEFAULT_MODEL,
    EVAL_DIR,
    FC5_LATENT_BANK,
    FC5_MODEL,
    FM6_LATENT_BANK,
    FM6_MODEL,
    OFFLINE_BIN,
    REPO_ROOT,
    SPACEFLIGHT_LATENT_BANK,
    SPACEFLIGHT_MODEL,
    ensure_dir,
    load_manifest,
    python_bin,
)
from pixel_mario_defaults import mario_pixel_args

SPACEFLIGHT_V2_STANDARD_BANK = EVAL_DIR / "spaceflight_v2/banks/standard/dict/latent_bank.json"
SPACEFLIGHT_V2_COMPOUND_BANK = EVAL_DIR / "spaceflight_v2/banks/compound/dict/latent_bank.json"

ENCODER_ARGS = [
    "--enc-codec",
    "libx264",
    "--enc-crf",
    "23",
    "--enc-preset",
    "medium",
    "--enc-tune",
    "none",
    "--enc-profile",
    "none",
    "--enc-level",
    "none",
    "--enc-open-gop-defaults",
    "--enc-aud",
    "1",
    "--enc-repeat-headers",
    "0",
]

MOTION_ARGS = [
    "--latent-motion-iou",
    "0.6",
    "--latent-motion-center",
    "20",
    "--latent-motion-scale",
    "0.25",
    "--latent-motion-boost",
    "6",
]

MASK_ARGS = [
    "--mask-color",
    "dominant",
    "--mask-color-period",
    "200",
    "--fill-mode",
    "solid",
    "--feather-px",
    "4",
]


def model_for_game(game: str) -> Path:
    if game == "fc5":
        return FC5_MODEL
    if game == "fm6":
        return FM6_MODEL
    if game == "spaceflight":
        return SPACEFLIGHT_MODEL
    return Path(DEFAULT_MODEL)


def game_args(
    game: str,
    *,
    spaceflight_multipart: bool = False,
    spaceflight_latent_bank: Path | None = None,
) -> list[str]:
    if game == "fc5":
        return [
            "--latent-key",
            "--latent-bank",
            str(FC5_LATENT_BANK),
            "--latent-thr",
            "0.85",
            *MOTION_ARGS,
            *MASK_ARGS,
        ]
    if game == "fm6":
        return [
            "--latent-key",
            "--yolo-heal-only",
            "--latent-bank",
            str(FM6_LATENT_BANK),
            "--latent-thr",
            "0.86",
            *MOTION_ARGS,
            *MASK_ARGS,
        ]
    if game == "spaceflight":
        bank = spaceflight_latent_bank or (
            SPACEFLIGHT_V2_COMPOUND_BANK if spaceflight_multipart else SPACEFLIGHT_V2_STANDARD_BANK
        )
        args = [
            "--latent-key",
            "--yolo-heal-only",
            "--latent-bank",
            str(bank),
            "--latent-thr",
            "0.86",
            *MOTION_ARGS,
            *MASK_ARGS,
        ]
        if spaceflight_multipart:
            args += ["--multipart-object", "--multipart-class", "0"]
        return args
    if game == "mario":
        return list(mario_pixel_args())
    raise ValueError(f"unsupported game {game!r}")


def run_clip(
    *,
    clip_id: str,
    game: str,
    input_path: Path,
    out_dir: Path,
    use_cuda: bool,
    force: bool,
    spaceflight_multipart: bool,
    spaceflight_latent_bank: Path | None,
) -> None:
    if (out_dir / "report.json").exists() and not force:
        print(f"[skip] complete {out_dir}")
        return
    ensure_dir(out_dir)
    cmd = [str(OFFLINE_BIN)]
    if use_cuda:
        cmd.append("-d")
    cmd += [
        "-i",
        str(input_path),
        "-o",
        str(out_dir),
        "-m",
        str(model_for_game(game)),
        *ENCODER_ARGS,
        *game_args(
            game,
            spaceflight_multipart=spaceflight_multipart,
            spaceflight_latent_bank=spaceflight_latent_bank,
        ),
    ]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    subprocess.run(
        [
            python_bin(),
            str(REPO_ROOT / "tools/metrics/build_per_frame_csv.py"),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run gop_analysis CRF23 offline encodes.")
    ap.add_argument("--manifest", type=Path, default=EVAL_DIR / "manifest/offline_manifest.json")
    ap.add_argument("--out-root", type=Path, default=EVAL_DIR / "gop_analysis")
    ap.add_argument("--game", choices=["fc5", "fm6", "mario", "spaceflight"], default="fm6")
    ap.add_argument("--clip-ids", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--spaceflight-multipart", action="store_true")
    ap.add_argument("--spaceflight-latent-bank", type=Path, default=None)
    args = ap.parse_args()

    manifest = {c.clip_id: c for c in load_manifest(args.manifest)}
    if args.clip_ids:
        clip_ids = list(args.clip_ids)
    elif args.game == "fm6":
        clip_ids = [f"fm6_{i:02d}" for i in range(5)]
    elif args.game == "mario":
        clip_ids = [f"mario_{i:02d}" for i in range(5)]
    elif args.game == "spaceflight":
        clip_ids = [f"spaceflight_{i:02d}" for i in range(5)]
    else:
        clip_ids = [f"fc5_{i:02d}" for i in range(5)]

    for clip_id in clip_ids:
        clip = manifest.get(clip_id)
        if clip is None:
            print(f"[skip] missing manifest clip {clip_id}", file=sys.stderr)
            continue
        if clip.game != args.game:
            print(f"[skip] {clip_id} game mismatch", file=sys.stderr)
            continue
        out_dir = args.out_root / f"{clip_id}_offline_open_x264_crf23"
        run_clip(
            clip_id=clip_id,
            game=clip.game,
            input_path=Path(clip.normalized_path),
            out_dir=out_dir,
            use_cuda=not args.no_cuda,
            force=args.force,
            spaceflight_multipart=args.spaceflight_multipart,
            spaceflight_latent_bank=args.spaceflight_latent_bank,
        )


if __name__ == "__main__":
    main()
