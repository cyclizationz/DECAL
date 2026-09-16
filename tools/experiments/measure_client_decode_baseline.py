#!/usr/bin/env python3
"""Measure decode-only client cost for Table 9 (stitch vs ordinary decode).

Times FFmpeg software decode of the same baseline H.264 MP4s used by the
Exp6 / spaceflight resource runs. No stitch, no YOLO, no extra filters.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "eval"
DEFAULT_OUT = ROOT / "exp6" / "client_decode_baseline"

CLIPS = [
    {
        "game": "FC5",
        "clip_id": "fc5_00",
        "path": ROOT / "exp6" / "resource_runs" / "fc5_00" / "original_output.mp4",
        "kind": "baseline",
    },
    {
        "game": "FM6",
        "clip_id": "fm6_00",
        "path": ROOT / "exp6" / "resource_runs" / "fm6_00" / "original_output.mp4",
        "kind": "baseline",
    },
    {
        "game": "Mario",
        "clip_id": "mario_00",
        "path": ROOT / "exp6" / "resource_runs" / "mario_00" / "original_output.mp4",
        "kind": "baseline",
    },
] + [
    {
        "game": "SC",
        "clip_id": f"spaceflight_{idx:02d}",
        "path": ROOT / "spaceflight_v2" / "resource" / f"spaceflight_{idx:02d}" / "original_output.mp4",
        "kind": "baseline",
    }
    for idx in range(5)
]


def host_info() -> dict[str, str]:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor() or "unknown"
    gpu = "unknown"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            gpu = out.splitlines()[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        try:
            lspci = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
            for line in lspci.splitlines():
                if "NVIDIA" in line and ("VGA" in line or "3D" in line):
                    gpu = line.split(": ", 1)[-1].strip()
                    break
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass
    return {
        "cpu": cpu,
        "gpu": gpu,
        "decode_impl": "ffmpeg libavcodec software H.264 to -f null (no NVDEC)",
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    }


def probe(path: Path) -> dict[str, str]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate,avg_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    stream = json.loads(out)["streams"][0]
    num, den = (stream.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {
        "codec": stream.get("codec_name", ""),
        "width": stream.get("width", ""),
        "height": stream.get("height", ""),
        "nb_frames": int(stream.get("nb_frames") or 0),
        "duration_s": float(stream.get("duration") or 0.0),
        "fps": fps,
    }


def run_ffmpeg_null(path: Path) -> tuple[float, float, str]:
    """Return (wall_s, ffmpeg_rtime_s, stderr)."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-benchmark",
        "-i",
        str(path),
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    wall_s = time.perf_counter() - started
    match = re.search(r"bench:\s+utime=([0-9.]+)s\s+stime=([0-9.]+)s\s+rtime=([0-9.]+)s", proc.stderr)
    rtime_s = float(match.group(3)) if match else wall_s
    return wall_s, rtime_s, proc.stderr


def measure_clip(clip: dict[str, object], repeats: int, warmup: bool) -> dict[str, object]:
    path = Path(clip["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    meta = probe(path)
    frames = int(meta["nb_frames"])
    if warmup:
        run_ffmpeg_null(path)
    walls: list[float] = []
    rtimes: list[float] = []
    for _ in range(repeats):
        wall_s, rtime_s, _ = run_ffmpeg_null(path)
        walls.append(wall_s)
        rtimes.append(rtime_s)
    wall_ms = [1000.0 * t / frames for t in walls]
    rtime_ms = [1000.0 * t / frames for t in rtimes]
    return {
        "game": clip["game"],
        "clip_id": clip["clip_id"],
        "kind": clip["kind"],
        "path": str(path),
        "codec": meta["codec"],
        "width": meta["width"],
        "height": meta["height"],
        "fps": f"{meta['fps']:.3f}",
        "frames": frames,
        "repeats": repeats,
        "mean_decode_ms": statistics.mean(rtime_ms),
        "median_decode_ms": statistics.median(rtime_ms),
        "p95_decode_ms": sorted(rtime_ms)[min(len(rtime_ms) - 1, int(0.95 * len(rtime_ms)))],
        "min_decode_ms": min(rtime_ms),
        "max_decode_ms": max(rtime_ms),
        "mean_wall_ms": statistics.mean(wall_ms),
        "median_wall_ms": statistics.median(wall_ms),
        "run_rtime_ms": rtime_ms,
        "run_wall_ms": wall_ms,
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["game"]), []).append(row)
    summary = []
    for game in ("FC5", "FM6", "Mario", "SC"):
        values = grouped[game]
        weights = [int(row["frames"]) for row in values]
        total_frames = sum(weights)
        mean_ms = sum(float(row["mean_decode_ms"]) * n for row, n in zip(values, weights)) / total_frames
        median_ms = statistics.mean(float(row["median_decode_ms"]) for row in values)
        summary.append(
            {
                "game": game,
                "clips": ",".join(str(row["clip_id"]) for row in values),
                "clip_count": len(values),
                "frames": total_frames,
                "width": values[0]["width"],
                "height": values[0]["height"],
                "fps": values[0]["fps"],
                "mean_decode_ms": mean_ms,
                "median_decode_ms": median_ms,
                "per_clip_mean_ms": ";".join(f"{row['clip_id']}={float(row['mean_decode_ms']):.4f}" for row in values),
            }
        )
    return summary


def write_md(summary: list[dict[str, object]], rows: list[dict[str, object]], host: dict[str, str], path: Path) -> None:
    lines = [
        "# Decode-only client baseline",
        "",
        "FFmpeg software decode of the same CRF23 baseline H.264 MP4s used by Table 9",
        "(Exp6 `original_output.mp4` for FC5/FM6/Mario clip 00; five-clip SC resource originals).",
        "Command: `ffmpeg -benchmark -i <mp4> -an -sn -dn -f null -`. No stitch, no YOLO.",
        "",
        f"- CPU: `{host['cpu']}`",
        f"- GPU: `{host['gpu']}` (present; this measurement does not use NVDEC)",
        f"- Decoder: `{host['decode_impl']}`",
        f"- FFmpeg: `{host['ffmpeg']}`",
        "",
        "| Game | Clips | Frames | Resolution | Mean ms/frame | Median ms/frame |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['game']} | {row['clips']} | {row['frames']} | "
            f"{row['width']}x{row['height']}@{row['fps']} | "
            f"{float(row['mean_decode_ms']):.3f} | {float(row['median_decode_ms']):.3f} |"
        )
    lines += ["", "## Per-clip runs", ""]
    for row in rows:
        runs = ", ".join(f"{ms:.3f}" for ms in row["run_rtime_ms"])
        lines.append(
            f"- {row['game']} `{row['clip_id']}`: mean {float(row['mean_decode_ms']):.3f} ms "
            f"(runs: {runs})"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    host = host_info()
    rows = [measure_clip(clip, args.repeats, warmup=not args.no_warmup) for clip in CLIPS]
    summary = summarize(rows)
    csv_path = args.out_dir / "decode_baseline_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k in summary[0] if k != "per_clip_mean_ms"] + ["per_clip_mean_ms"])
        writer.writeheader()
        writer.writerows(summary)
    (args.out_dir / "decode_baseline_per_clip.json").write_text(json.dumps({"host": host, "clips": rows}, indent=2) + "\n")
    write_md(summary, rows, host, args.out_dir / "decode_baseline_summary.md")
    print(json.dumps({"out_dir": str(args.out_dir), "summary": summary, "host": host}, indent=2))


if __name__ == "__main__":
    main()
