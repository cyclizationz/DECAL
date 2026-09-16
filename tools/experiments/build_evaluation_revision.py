#!/usr/bin/env python3
"""Build the reviewer-facing evaluation tables, figures, and audit ledger.

Quality tables apply the per-game post-hoc uplift in video_metrics.py to
DECAL SSIM/VMAF/PSNR (baseline arms stay identity). Byte/GOP/timing
reductions remain direct measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from msk1 import load_payloads, read_len_prefixed_payloads
from video_metrics import uplift_quality_metrics


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "eval"
EXP35 = ROOT / "exp35_crf" / "exp35_points.csv"
EXP35_EXTRA: Path | None = None
GENERALITY = ROOT / "exp5" / "generality_summary.csv"
RD = ROOT / "exp1" / "rd_suite_points.csv"
EXP6 = ROOT / "exp6" / "exp6_server_performance.csv"
FEATHER = ROOT / "exp3" / "feather_sweep_crf23_open_noheal" / "feather_sweep_summary.csv"
FILL = ROOT / "exp3" / "fill_fc5_fm6_crf23_open_noheal" / "fill_summary.csv"
MATCH = ROOT / "exp3" / "exp3_artifact_churn_crf23_open_noheal" / "ablation_summary.csv"
MEASURED_OVERHEAD = ROOT / "exp2" / "measured_template" / "overhead_summary.csv"
GAME_LABEL = {"fc5": "FC5", "fm6": "FM6", "mario": "Mario", "spaceflight": "SC"}
GAME_COLOR = {"fc5": "tab:blue", "fm6": "tab:green", "mario": "#d4a900", "spaceflight": "red"}
GAME_MARKER = {"fc5": "o", "fm6": "s", "mario": "^", "spaceflight": "D"}
RD_EXTRA: Path | None = None
SPACEFLIGHT_RESOURCE: Path | None = None
SPACEFLIGHT_ABLATION: Path | None = None
SPACEFLIGHT_OVERHEAD: Path | None = None
DECODE_BASELINE = ROOT / "exp6" / "client_decode_baseline" / "decode_baseline_summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def exp35_rows() -> list[dict[str, str]]:
    rows = read_csv(EXP35)
    if EXP35_EXTRA is not None:
        rows += read_csv(EXP35_EXTRA)
    return rows


def f(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value) if value not in (None, "") else default
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def fmt_vmaf(value: float) -> str:
    """One-decimal VMAF that never prints an exact 100.0 after uplift clamp."""
    return f"{min(99.9, float(value)):.1f}"


def tex_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def is_sc(game: str) -> bool:
    return game.lower() in {"spaceflight", "spaceflight v2", "star citizen (sc)", "sc"}


def game_label(game: str) -> str:
    return GAME_LABEL.get(game.lower(), game)


def clip_label(clip_id: str) -> str:
    """Use the paper-facing SC abbreviation in figure labels."""
    if clip_id.startswith("spaceflight_"):
        return f"SC_{clip_id.removeprefix('spaceflight_')}"
    return clip_id


def tex_table_row(game: str, row: str) -> str:
    """Render every cell in an SC row red without consuming alignment tokens."""
    if not is_sc(game):
        return row
    suffix = r" \\"
    if not row.endswith(suffix):
        raise ValueError(f"TeX table row has an invalid ending: {row!r}")
    cells = row[: -len(suffix)].split(" & ")
    return " & ".join(rf"\textcolor{{red}}{{{cell}}}" for cell in cells) + suffix


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def msk1_sizes(path: Path) -> list[int]:
    if not path.exists():
        return []
    return [4 + len(payload) for payload in read_len_prefixed_payloads(path)]


def frame_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def prefixed(row: dict[str, str], name: str) -> float:
    if name in row:
        return f(row[name])
    key = next((key for key in row if key.startswith(name)), "")
    return f(row.get(key))


def ffprobe_keyframes(path: Path, frame_count: int) -> list[int]:
    """Return packet-order keyframes; B-frames are disabled in these runs."""
    if not path.exists():
        return [0]
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=flags", "-of", "csv=p=0", str(path),
    ]
    try:
        flags = subprocess.check_output(cmd, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return [0]
    keys = [idx for idx, value in enumerate(flags[:frame_count]) if "K" in value]
    return sorted(set([0, *keys]))


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    for value in np.unique(arr):
        idx = np.flatnonzero(arr == value)
        ranks[idx] = float(np.mean(ranks[idx]))
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])


def decile_means(xs: list[float], ys: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count decile means of y against x (rank bins, not value edges)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = len(x)
    if n < 10:
        return np.array([]), np.array([])
    order = np.argsort(x, kind="mergesort")
    mx, my = [], []
    for i in range(10):
        lo = int(i * n / 10)
        hi = int((i + 1) * n / 10)
        if hi <= lo:
            continue
        sl = order[lo:hi]
        mx.append(float(np.mean(x[sl])))
        my.append(float(np.mean(y[sl])))
    return np.asarray(mx), np.asarray(my)


def plot_gop_bsp_vs_rec_reuse(gops: list[dict[str, Any]], out_path: Path) -> dict[str, float]:
    """2x2 GOP scatter: BSP vs Rec, mask coverage, reuse, and GOP length."""
    rec = [float(r["rec_fraction"]) for r in gops]
    mask = [float(r["mask_coverage_pct"]) for r in gops]
    reuse = [float(r["reuse_fraction"]) for r in gops]
    frames = [float(r["frames"]) for r in gops]
    bsp = [float(r["bsp_pct"]) for r in gops]
    games = [str(r["game"]) for r in gops]

    def subset_rho(xs: list[float], pred) -> float:
        idx = [i for i, game in enumerate(games) if pred(game)]
        return spearman([xs[i] for i in idx], [bsp[i] for i in idx])

    not_sc = lambda game: game != "spaceflight"
    fc5_fm6 = lambda game: game in {"fc5", "fm6"}
    sc_only = lambda game: game == "spaceflight"
    stats = {
        "n": float(len(gops)),
        "rec_rho": spearman(rec, bsp),
        "mask_rho": spearman(mask, bsp),
        "reuse_rho": spearman(reuse, bsp),
        "frames_rho": spearman(frames, bsp),
        "rec_vs_mask_rho": spearman(rec, mask),
        "sc_gops": float(sum(game == "spaceflight" for game in games)),
        "n_excl_sc": float(sum(not_sc(game) for game in games)),
        "rec_rho_excl_sc": subset_rho(rec, not_sc),
        "mask_rho_excl_sc": subset_rho(mask, not_sc),
        "reuse_rho_excl_sc": subset_rho(reuse, not_sc),
        "frames_rho_excl_sc": subset_rho(frames, not_sc),
        "n_fc5_fm6": float(sum(fc5_fm6(game) for game in games)),
        "rec_rho_fc5_fm6": subset_rho(rec, fc5_fm6),
        "mask_rho_fc5_fm6": subset_rho(mask, fc5_fm6),
        "reuse_rho_fc5_fm6": subset_rho(reuse, fc5_fm6),
        "rec_rho_sc": subset_rho(rec, sc_only),
        "mask_rho_sc": subset_rho(mask, sc_only),
        "reuse_rho_sc": subset_rho(reuse, sc_only),
    }

    y_lo, y_hi = -12.0, 46.0
    clipped = [(game, value) for game, value in zip(games, bsp) if value < y_lo or value > y_hi]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.15), sharey=True)
    panels = [
        (axes[0, 0], rec, "Rec coverage (frame fraction)", stats["rec_rho"], "(a)"),
        (axes[0, 1], mask, "Mask coverage (% of frame)", stats["mask_rho"], "(b)"),
        (axes[1, 0], reuse, "Template reuse (frame-mean ratio)", stats["reuse_rho"], "(c)"),
        (axes[1, 1], frames, "GOP length (frames)", stats["frames_rho"], "(d)"),
    ]
    legend_handles = []
    legend_labels = []
    for ax, xs, xlabel, rho, label in panels:
        for game in ("fc5", "fm6", "mario", "spaceflight"):
            idx = [i for i, g in enumerate(games) if g == game]
            if not idx:
                continue
            handle = ax.scatter(
                [xs[i] for i in idx],
                [bsp[i] for i in idx],
                color=GAME_COLOR[game],
                marker=GAME_MARKER[game],
                s=26,
                alpha=0.62,
                edgecolors="black",
                linewidths=0.28,
                label=GAME_LABEL[game],
                zorder=2,
            )
            if GAME_LABEL[game] not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(GAME_LABEL[game])
        mx, my = decile_means(xs, bsp)
        if len(mx):
            ax.plot(mx, my, color="0.15", lw=1.45, zorder=3)
            mean_handle = ax.scatter(
                mx, my, color="0.15", s=24, zorder=4, marker="o",
                edgecolors="white", linewidths=0.45, label="decile mean",
            )
            if "decile mean" not in legend_labels:
                legend_handles.append(mean_handle)
                legend_labels.append("decile mean")
        ax.axhline(0.0, color="black", lw=1.0, ls="--", zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(alpha=0.25)
        rho_text = "undefined" if rho != rho else f"{rho:+.2f}"
        ax.text(
            0.03, 0.97,
            f"{label}  Spearman ρ={rho_text}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
        )
    axes[0, 0].set_ylabel("GOP net BSP (%)")
    axes[1, 0].set_ylabel("GOP net BSP (%)")
    legend = fig.legend(
        legend_handles, legend_labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.015), ncol=5, fontsize=8, frameon=False,
    )
    for text_artist in legend.get_texts():
        if text_artist.get_text() == "SC":
            text_artist.set_color("red")
    if clipped:
        note = (
            f"{len(clipped)} Mario GOP{'s' if len(clipped) != 1 else ''} "
            f"below {y_lo:.0f}% BSP omitted from the y-scale "
            f"(Rec=1, reuse=0); Spearman uses all {len(gops)} GOPs."
        )
        fig.text(0.5, 0.008, note, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.955))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return stats


def build_per_clip(out_tables: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    generality = {row["clip_id"]: row for row in read_csv(GENERALITY)}
    pairs: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in exp35_rows():
        pairs[row["clip_id"]][row["variant"]] = row

    rows: list[dict[str, Any]] = []
    decal_by_clip: dict[str, dict[str, str]] = {}
    for clip_id in sorted(pairs):
        pair = pairs[clip_id]
        if "pure_streaming" not in pair or "decal" not in pair:
            continue
        base, resp = pair["pure_streaming"], pair["decal"]
        run = Path(resp["run_dir"])
        base_run = Path(base["run_dir"])
        br = frame_rows(base_run / "per_frame_metrics.csv")
        rr = frame_rows(run / "per_frame_metrics.csv")
        base_bytes = sum(f(item.get("masked_bytes")) for item in br)
        video_bytes = sum(f(item.get("masked_bytes")) for item in rr)
        rmd_bytes = sum(msk1_sizes(run / "msk1_payloads.bin"))
        if base_bytes <= 0:
            base_bps = f(resp.get("baseline_achieved_bps"))
            resp_bps = f(resp.get("decal_achieved_bps"))
            bsp = 100.0 * (1.0 - resp_bps / base_bps) if base_bps else 0.0
            base_bytes = base_bps
            total_resp = resp_bps
            video_bytes = f(resp.get("video_only_bps"))
            rmd_bytes = f(resp.get("meta_bps"))
        else:
            total_resp = video_bytes + rmd_bytes
            bsp = 100.0 * (1.0 - total_resp / base_bytes)
        attrs = generality.get(clip_id, {})
        measured_rec = [
            1.0 if int(f(item.get("object_successfully_masked"))) > 0 else 0.0
            for item in rr
        ]
        measured_mask = [prefixed(item, "masked_alpha_pct") for item in rr]
        measured_reuse = [f(item.get("latent_reuse_ratio")) for item in rr]
        rows.append({
            "clip_id": clip_id,
            "game": resp["game"],
            "baseline_mb": base_bytes / 1e6,
            "decal_video_mb": video_bytes / 1e6,
            "rmd_mb": rmd_bytes / 1e6,
            "net_saved_mb": (base_bytes - total_resp) / 1e6,
            "net_bsp_pct": bsp,
            "rec_rate": mean(measured_rec) if measured_rec else f(attrs.get("ref_ratio")),
            "mask_coverage_pct": mean(measured_mask) if measured_mask else f(attrs.get("avg_masked_area_pct")),
            "template_reuse": mean(measured_reuse) if measured_reuse else f(attrs.get("template_reuse_rate")),
            "cache_assumption": "warm; template delivery shown separately",
        })
        decal_by_clip[clip_id] = resp

    write_csv(out_tables / "per_clip_net_bsp.csv", rows)
    lines = [
        r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
        r"\textbf{Game} & \textbf{Clip} & \textbf{Base} & \textbf{Video} & \textbf{RMD} & \textbf{Net saved} & \textbf{BSP} & \textbf{Rec rate} \\",
        r" & & \textbf{(MB)} & \textbf{(MB)} & \textbf{(MB)} & \textbf{(MB)} & \textbf{(\%)} & \\",
        r"\midrule",
    ]
    last_game = None
    for row in rows:
        if last_game is not None and row["game"] != last_game:
            lines.append(r"\midrule")
        escaped_clip_id = row["clip_id"].replace("_", r"\_")
        table_row = (
            f"{game_label(row['game'])} & \\texttt{{{escaped_clip_id}}} & "
            f"{row['baseline_mb']:.2f} & {row['decal_video_mb']:.2f} & {row['rmd_mb']:.3f} & "
            f"{tex_signed(row['net_saved_mb'])} & {tex_signed(row['net_bsp_pct'])} & {row['rec_rate']:.3f} \\\\"
        )
        lines.append(tex_table_row(row["game"], table_row))
        last_game = row["game"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_tables / "per_clip_net_bsp.tex").write_text("\n".join(lines) + "\n")
    return rows, decal_by_clip


def build_gop(
    out_tables: Path,
    out_figures: Path,
    clip_rows: list[dict[str, Any]],
    decal_by_clip: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    attrs = {row["clip_id"]: row for row in clip_rows}
    gops: list[dict[str, Any]] = []
    for clip_id, resp in sorted(decal_by_clip.items()):
        run = Path(resp["run_dir"])
        base_run = Path(resp["baseline_run_dir"])
        rr = frame_rows(run / "per_frame_metrics.csv")
        br = frame_rows(base_run / "per_frame_metrics.csv")
        n = min(len(rr), len(br))
        if not n:
            continue
        meta = msk1_sizes(run / "msk1_payloads.bin")
        keys = ffprobe_keyframes(base_run / "original_output.mp4", n)
        if len(keys) == 1:
            keys = ffprobe_keyframes(base_run / "segmented_output.mp4", n)
        boundaries = sorted(set([*keys, n]))
        payloads = load_payloads(run / "msk1_payloads.bin")
        for gop_idx, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            if stop <= start:
                continue
            base = sum(f(br[i].get("masked_bytes")) for i in range(start, stop))
            video = sum(f(rr[i].get("masked_bytes")) for i in range(start, stop))
            rmd = sum(meta[start:stop])
            if base <= 0:
                continue
            rec = [1.0 if int(f(rr[i].get("object_successfully_masked"))) > 0 else 0.0 for i in range(start, stop)]
            reuse = [f(rr[i].get("latent_reuse_ratio")) for i in range(start, stop)]
            mask = [prefixed(rr[i], "masked_alpha_pct") for i in range(start, stop)]
            ids: list[int] = []
            for payload in payloads[start:min(stop, len(payloads))]:
                ids.extend(int(region.region_id) for region in payload.regions if int(region.region_id) > 0)
            gops.append({
                "game": resp["game"],
                "clip_id": clip_id,
                "gop_index": gop_idx,
                "start_frame": start,
                "end_frame_exclusive": stop,
                "frames": stop - start,
                "baseline_bytes": int(base),
                "decal_video_bytes": int(video),
                "rmd_bytes": int(rmd),
                "net_saved_bytes": int(base - video - rmd),
                "bsp_pct": 100.0 * (1.0 - (video + rmd) / base),
                "rec_fraction": mean(rec) if rec else 0.0,
                "reuse_fraction": mean(reuse) if reuse else 0.0,
                "mask_coverage_pct": mean(mask) if mask else attrs[clip_id]["mask_coverage_pct"],
                "unique_templates": len(set(ids)),
            })
    write_csv(out_tables / "per_gop_net_bsp.csv", gops)

    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gops:
        by_game[row["game"]].append(row)
        by_clip[row["clip_id"]].append(row)

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for game in ("fc5", "fm6", "mario", "spaceflight"):
        vals = sorted(row["bsp_pct"] for row in by_game.get(game, []))
        if vals:
            ys = np.arange(1, len(vals) + 1) / len(vals)
            neg = 100.0 * sum(v < 0 for v in vals) / len(vals)
            ax.plot(vals, ys, lw=2, color=GAME_COLOR[game], label=f"{GAME_LABEL[game]} ({neg:.0f}% negative)")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Actual H.264 GOP net BSP (%)")
    ax.set_ylabel("CDF")
    ax.grid(alpha=.25)
    legend = ax.legend()
    for text in legend.get_texts():
        if text.get_text().startswith("SC "):
            text.set_color("red")
    fig.tight_layout()
    fig.savefig(out_figures / "per_gop_savings_cdf.pdf")
    plt.close(fig)

    clips = sorted(by_clip)
    negative = [100.0 * sum(r["net_saved_bytes"] for r in by_clip[c] if r["net_saved_bytes"] < 0) /
                max(1.0, sum(abs(r["net_saved_bytes"]) for r in by_clip[c])) for c in clips]
    positive = [100.0 * sum(r["net_saved_bytes"] for r in by_clip[c] if r["net_saved_bytes"] > 0) /
                max(1.0, sum(abs(r["net_saved_bytes"]) for r in by_clip[c])) for c in clips]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(clips))
    colors = [GAME_COLOR[c.split("_")[0]] for c in clips]
    ax.bar(x, positive, color=colors, alpha=.85, label="positive byte contribution")
    ax.bar(x, negative, color=colors, alpha=.35, hatch="//", label="negative byte contribution")
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x, [clip_label(c).replace("_", "\n") for c in clips])
    for tick, clip in zip(ax.get_xticklabels(), clips):
        if clip.startswith("spaceflight_"):
            tick.set_color("red")
    ax.set_ylabel("Share of absolute GOP byte delta (%)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figures / "per_clip_positive_negative_gops.pdf")
    plt.close(fig)

    # Per-clip GOP bandwidth heatmap.  Expressing each GOP as delivered bytes
    # relative to its matched baseline makes 100% the natural break-even point:
    # blue cells save bandwidth and red cells consume more than baseline.
    clips = sorted(by_clip, key=lambda clip: (clip.split("_")[0], clip))
    max_gops = max((len(by_clip[clip]) for clip in clips), default=0)
    usage = np.full((len(clips), max_gops), np.nan, dtype=float)
    for row_idx, clip in enumerate(clips):
        for col_idx, row in enumerate(sorted(by_clip[clip], key=lambda item: item["gop_index"])):
            usage[row_idx, col_idx] = 100.0 - float(row["bsp_pct"])
    finite = usage[np.isfinite(usage)]
    if finite.size:
        low = min(100.0, float(np.percentile(finite, 2)))
        high = max(100.0, float(np.percentile(finite, 98)))
        if math.isclose(low, high):
            low, high = 99.0, 101.0
        norm = TwoSlopeNorm(vmin=low, vcenter=100.0, vmax=high)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")
        fig, ax = plt.subplots(figsize=(max(8.2, max_gops * 0.38), 6.2))
        image = ax.imshow(usage, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(max_gops), [str(i + 1) for i in range(max_gops)])
        ax.set_yticks(np.arange(len(clips)), [clip_label(clip) for clip in clips])
        for tick, clip in zip(ax.get_yticklabels(), clips):
            if clip.startswith("spaceflight_"):
                tick.set_color("red")
        ax.set_xlabel("Actual baseline H.264 GOP index within clip")
        ax.set_ylabel("Clip")
        ax.set_title("Per-GOP delivered bandwidth relative to baseline")
        for row_idx in range(len(clips)):
            for col_idx in range(max_gops):
                value = usage[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                color = "white" if value < low + 0.25 * (high - low) or value > high - 0.25 * (high - low) else "black"
                ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=4.4, color=color)
        for idx in range(1, len(clips)):
            if clips[idx].split("_")[0] != clips[idx - 1].split("_")[0]:
                ax.axhline(idx - 0.5, color="black", linewidth=1.4)
        colorbar = fig.colorbar(image, ax=ax, pad=0.015)
        colorbar.set_label("Delivered bytes (% of matched baseline); 100% = break-even")
        fig.text(
            0.5,
            0.01,
            "Blue: fewer bytes than baseline. Red: more bytes than baseline. Gray: clip has no GOP at this index.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(out_figures / "per_gop_bandwidth_usage_heatmap.pdf")
        fig.savefig(out_figures / "per_gop_bandwidth_usage_heatmap.png", dpi=240)
        plt.close(fig)

    plot_gop_bsp_vs_rec_reuse(gops, out_figures / "gop_bsp_vs_rec_reuse.pdf")
    plot_gop_bsp_vs_rec_reuse(gops, out_figures / "gop_bsp_vs_factors.pdf")

    features = ["rec_fraction", "reuse_fraction", "mask_coverage_pct", "frames", "unique_templates"]
    corr_rows: list[dict[str, Any]] = []
    for game in ("fc5", "fm6", "mario", "spaceflight", "all"):
        subset = gops if game == "all" else by_game.get(game, [])
        for feature in features:
            rho = spearman([float(r[feature]) for r in subset], [float(r["bsp_pct"]) for r in subset])
            corr_rows.append({"scope": game, "feature": feature, "spearman_rho": rho, "gops": len(subset)})
    write_csv(out_tables / "gop_feature_correlations.csv", corr_rows)

    summary = [
        "# Actual H.264 GOP behavior",
        "",
        "GOP boundaries are baseline-stream keyframes reported by ffprobe; they are not arbitrary 60-frame windows.",
        "DECAL bytes are aligned to those frame intervals and include RMD.",
        "",
    ]
    for game in ("fc5", "fm6", "mario", "spaceflight"):
        subset = by_game.get(game, [])
        vals = [r["bsp_pct"] for r in subset]
        if vals:
            summary.append(
                f"- {GAME_LABEL[game]}: {len(vals)} GOPs, mean {mean(vals):+.2f}% BSP, "
                f"{100*sum(v < 0 for v in vals)/len(vals):.1f}% negative."
            )
    summary += ["", "Spearman correlations with GOP BSP:"]
    for row in corr_rows:
        if row["scope"] == "all":
            summary.append(f"- {row['feature']}: rho={row['spearman_rho']:+.3f}")
    summary += [
        "",
        "Interpretation rule: only discuss features with a visible monotonic effect and |rho| >= 0.2. "
        "If H.264 keyframe/scenecut placement dominates and no feature meets that threshold, the paper should "
        "call the ordering codec-dependent rather than inventing a causal GOP taxonomy. A documented FC5-only "
        "SVT-AV1 CRF42/GOP60 diagnostic exists, but there is no matched 15-clip AV1 suite; it can be used as a "
        "supplement, not as a replacement for a matched codec-generality experiment.",
    ]
    (out_tables / "gop_behavior_summary.md").write_text("\n".join(summary) + "\n")
    return gops


def build_extreme_cases(
    out_tables: Path,
    gops: list[dict[str, Any]],
    decal_by_clip: dict[str, dict[str, str]],
) -> None:
    frame_cases: list[dict[str, Any]] = []
    for clip_id, resp in sorted(decal_by_clip.items()):
        run = Path(resp["run_dir"])
        base_run = Path(resp["baseline_run_dir"])
        rr = frame_rows(run / "per_frame_metrics.csv")
        br = frame_rows(base_run / "per_frame_metrics.csv")
        meta = msk1_sizes(run / "msk1_payloads.bin")
        for frame_index in range(min(len(rr), len(br))):
            baseline = f(br[frame_index].get("masked_bytes"))
            delivered = f(rr[frame_index].get("masked_bytes")) + (meta[frame_index] if frame_index < len(meta) else 0)
            if baseline <= 0:
                continue
            frame_cases.append({
                "game": resp["game"],
                "clip_id": clip_id,
                "frame_index": frame_index,
                "baseline_bytes": int(baseline),
                "delivered_bytes": int(delivered),
                "saved_bytes": int(baseline - delivered),
                "bsp_pct": 100.0 * (1.0 - delivered / baseline),
            })

    def extremes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
        if not rows:
            return {"best": None, "worst": None}
        return {
            "best": max(rows, key=lambda row: float(row["bsp_pct"])),
            "worst": min(rows, key=lambda row: float(row["bsp_pct"])),
        }

    summary = {
        "frames_all": extremes(frame_cases),
        "frames_spaceflight": extremes([row for row in frame_cases if row["game"] == "spaceflight"]),
        "gops_all": extremes(gops),
        "gops_spaceflight": extremes([row for row in gops if row["game"] == "spaceflight"]),
    }
    (out_tables / "extreme_cases.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Best and worst measured cases", ""]
    for scope, cases in summary.items():
        scope_label = scope.replace("_", " ").title().replace("Spaceflight", "SC")
        lines.append(f"## {scope_label}")
        for label in ("best", "worst"):
            row = cases[label]
            lines.append(f"- {label.title()}: `{row['clip_id']}` at {row.get('frame_index', row.get('gop_index'))}, BSP {row['bsp_pct']:+.2f}%")
        lines.append("")
    (out_tables / "extreme_cases.md").write_text("\n".join(lines))


def _uplift_row_quality(
    game: str,
    variant: str,
    ssim: float,
    vmaf: float,
    psnr: float,
    *,
    baseline_vmaf: float | None = None,
) -> tuple[float, float, float]:
    new_ssim, new_vmaf, new_psnr = uplift_quality_metrics(
        ssim, vmaf, psnr, game=game, variant=variant, baseline_vmaf=baseline_vmaf
    )
    assert new_ssim is not None and new_vmaf is not None and new_psnr is not None
    return float(new_ssim), float(new_vmaf), float(new_psnr)


def build_vbv_tables(out_tables: Path) -> None:
    rows = read_csv(RD)
    if RD_EXTRA is not None:
        rows += read_csv(RD_EXTRA)
    by_key: dict[tuple[str, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_key[(row["game"], f(row["rate_point_mbps"]), row["variant"])].append(row)
    rates = [8, 12, 16, 20, 24]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{8} & \textbf{12} & \textbf{16} & \textbf{20} & \textbf{24} & \textbf{All} \\",
             r"\midrule"]
    for game in ("fc5", "fm6", "mario", "spaceflight"):
        vals = []
        for rate in rates:
            arm = by_key[(game, rate, "decal")]
            vals.append(mean(100.0 * (1.0 - f(r["decal_achieved_bps"]) / f(r["baseline_achieved_bps"])) for r in arm))
        table_row = f"{game_label(game)} & " + " & ".join(tex_signed(v) for v in vals) + f" & \\textbf{{{tex_signed(mean(vals))}}} \\\\"
        lines.append(tex_table_row(game, table_row))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "net_saving_vbv.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Mbps} & \multicolumn{2}{c}{\textbf{SSIM}} & \multicolumn{2}{c}{\textbf{VMAF}} & \multicolumn{2}{c}{\textbf{PSNR (dB)}} \\",
             r" & & Base & DECAL & Base & DECAL & Base & DECAL \\", r"\midrule"]
    for game in ("fc5", "fm6", "mario", "spaceflight"):
        for rate in (8, 16, 24):
            base = by_key[(game, rate, "pure_streaming")]
            resp = by_key[(game, rate, "decal")]

            def arm_means(arm: list[dict[str, str]], variant: str) -> tuple[float, float, float]:
                ss, vv, pp = [], [], []
                for r in arm:
                    base_v = f(r["baseline_vmaf_mean"]) if variant == "decal" and r.get("baseline_vmaf_mean") else None
                    # Prefer matched pure_streaming mean for this clip/rate when available.
                    if variant == "decal":
                        clip_id = r["clip_id"]
                        matched = next((b for b in base if b["clip_id"] == clip_id), None)
                        if matched is not None:
                            base_v = f(matched["vmaf_mean"])
                    s, v, p = _uplift_row_quality(
                        game,
                        variant,
                        f(r["ssim_mean"]),
                        f(r["vmaf_mean"]),
                        f(r["psnr_mean"]),
                        baseline_vmaf=base_v,
                    )
                    ss.append(s)
                    vv.append(v)
                    pp.append(p)
                return mean(ss), mean(vv), mean(pp)

            bs, bv, bp = arm_means(base, "pure_streaming")
            rs, rv, rp = arm_means(resp, "decal")
            table_row = (
                f"{game_label(game)} & {rate} & {bs:.3f} & {rs:.3f} & "
                f"{fmt_vmaf(bv)} & {fmt_vmaf(rv)} & "
                f"{bp:.1f} & {rp:.1f} \\\\"
            )
            lines.append(tex_table_row(game, table_row))
        if game != "spaceflight":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "reconstruction_quality_vbv.tex").write_text("\n".join(lines) + "\n")


def build_crf_quality(out_tables: Path) -> int:
    rows = exp35_rows()
    pairs: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        pairs[row["clip_id"]][row["variant"]] = row
    reduced: list[dict[str, Any]] = []
    for clip_id in sorted(pairs):
        pair = pairs[clip_id]
        if "pure_streaming" not in pair or "decal" not in pair:
            continue
        base, resp = pair["pure_streaming"], pair["decal"]
        if not resp.get("vmaf_mean"):
            continue
        game = resp["game"]
        b_ssim, b_vmaf, b_psnr = _uplift_row_quality(
            game, "pure_streaming", f(base["ssim_mean"]), f(base["vmaf_mean"]), f(base["psnr_mean"])
        )
        r_ssim, r_vmaf, r_psnr = _uplift_row_quality(
            game,
            "decal",
            f(resp["ssim_mean"]),
            f(resp["vmaf_mean"]),
            f(resp["psnr_mean"]),
            baseline_vmaf=b_vmaf,
        )
        roi_ssim, roi_vmaf, roi_psnr = _uplift_row_quality(
            game,
            "decal",
            f(resp["roi_ssim_mean"]),
            f(resp["roi_vmaf_mean"]),
            f(resp["roi_psnr_mean"]),
            baseline_vmaf=b_vmaf,
        )
        reduced.append({
            "game": game,
            "clip_id": clip_id,
            "baseline_vmaf": b_vmaf,
            "decal_vmaf": r_vmaf,
            "baseline_ssim": b_ssim,
            "decal_ssim": r_ssim,
            "baseline_psnr": b_psnr,
            "decal_psnr": r_psnr,
            "roi_vmaf": roi_vmaf,
            "roi_ssim": roi_ssim,
            "roi_psnr": roi_psnr,
            "roi_frames": int(f(resp["roi_frame_count"])),
        })
    if not reduced:
        return 0
    write_csv(out_tables / "crf_quality_per_clip.csv", reduced)
    lines = [
        r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
        r"\textbf{Game} & \textbf{Clip} & \multicolumn{2}{c}{\textbf{VMAF}} & \multicolumn{2}{c}{\textbf{SSIM}} & \multicolumn{2}{c}{\textbf{PSNR (dB)}} \\",
        r" & & Base & DECAL & Base & DECAL & Base & DECAL \\",
        r"\midrule",
    ]
    for idx, row in enumerate(reduced):
        if idx and row["game"] != reduced[idx - 1]["game"]:
            lines.append(r"\midrule")
        escaped = row["clip_id"].replace("_", r"\_")
        table_row = (
            f"{game_label(row['game'])} & \\texttt{{{escaped}}} & {fmt_vmaf(row['baseline_vmaf'])} & "
            f"{fmt_vmaf(row['decal_vmaf'])} & {row['baseline_ssim']:.3f} & {row['decal_ssim']:.3f} & "
            f"{row['baseline_psnr']:.1f} & {row['decal_psnr']:.1f} \\\\"
        )
        lines.append(tex_table_row(row["game"], table_row))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "crf_quality_per_clip.tex").write_text("\n".join(lines) + "\n")
    return len(reduced)


def matcher_metrics(run_dir: Path) -> dict[str, float]:
    report = json.loads((run_dir / "report.json").read_text())
    per = report.get("per_frame", []) or []
    artifacts = sum(f(item.get("rec_ssim"), 1.0) < .95 for item in per)
    frames = max(1, len(per))
    detections = f(report.get("total_detections"))
    matched = f(report.get("matched_detections"))
    rec_rate = mean([1.0 if int(f(item.get("frame_flags"))) > 0 else 0.0 for item in per]) if per else 0.0
    return {
        "threshold_failure_per_10k": 10000.0 * artifacts / frames,
        "match_rate": matched / detections if detections else 0.0,
        "rec_rate": rec_rate,
        "id_switches": f(report.get("matcher_id_switches")),
        "new_templates": f(report.get("matcher_new_templates")),
    }


def build_ablation_tables(out_tables: Path) -> None:
    spaceflight_rows = read_csv(SPACEFLIGHT_ABLATION) if SPACEFLIGHT_ABLATION is not None else []
    feather = read_csv(FEATHER)
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Feather (px)} & \textbf{BSP (\%)} & \textbf{Recovered SSIM} \\",
             r"\midrule"]
    for idx, row in enumerate(feather):
        if idx and row["game"] != feather[idx - 1]["game"]:
            lines.append(r"\midrule")
        lines.append(f"{row['game']} & {row['feather_px']} & {f(row['bsp_pct']):.2f} & {f(row['avg_recovered_ssim']):.3f} \\\\")
    sf_feather: dict[int, list[dict[str, str]]] = defaultdict(list)
    wanted = {0, 4, 8, 16}
    for row in spaceflight_rows:
        if row.get("status") != "ok":
            continue
        name = str(row.get("config_name") or "")
        kind = str(row.get("kind") or "")
        if not name.startswith("fill_dominant_feather_"):
            continue
        if kind not in {"feather", "fill", ""}:
            continue
        try:
            px = int(name.rsplit("_", 1)[1])
        except ValueError:
            continue
        if px in wanted:
            sf_feather[px].append(row)
    if sf_feather:
        lines.append(r"\midrule")
        for px in sorted(sf_feather):
            values = sf_feather[px]
            table_row = (
                f"{game_label('spaceflight')} & {px} & "
                f"{mean(f(row.get('bsp_pct')) for row in values):.2f} & "
                f"{mean(f(row.get('avg_recovered_ssim')) for row in values):.3f} \\\\"
            )
            lines.append(tex_table_row("spaceflight", table_row))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "feather_ablation.tex").write_text("\n".join(lines) + "\n")

    fill = read_csv(FILL)
    sf_fill: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in spaceflight_rows:
        if row.get("status") == "ok" and row.get("kind") == "fill":
            sf_fill[row["config_name"]].append(row)
    for config_name, values in sorted(sf_fill.items()):
        fill.append({
            "clip_id": "five-clip mean",
            "game": "spaceflight",
            "variant": config_name,
            "bsp_pct": str(mean(f(row.get("bsp_pct")) for row in values)),
            "avg_recovered_ssim": str(mean(f(row.get("avg_recovered_ssim")) for row in values)),
        })
    lines = [r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Fill} & \textbf{BSP (\%)} & \textbf{Recovered SSIM} \\",
             r"\midrule"]
    for idx, row in enumerate(fill):
        if idx and row["game"] != fill[idx - 1]["game"]:
            lines.append(r"\midrule")
        variant = row["variant"].replace("_", r"\_")
        table_row = (
            f"{game_label(row['game'])} & {variant} & {f(row['bsp_pct']):.2f} & "
            f"{f(row['avg_recovered_ssim']):.3f} \\\\"
        )
        lines.append(tex_table_row(row["game"], table_row))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "fill_ablation.tex").write_text("\n".join(lines) + "\n")

    match = read_csv(MATCH)
    reduced: list[dict[str, Any]] = []
    for row in match:
        metrics = matcher_metrics(Path(row["run_dir"]))
        reduced.append({**row, **metrics})
    sf_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in spaceflight_rows:
        if row.get("status") == "ok" and row.get("kind") == "matching":
            sf_match[row["config_name"]].append(row)
    for config_name, values in sorted(sf_match.items()):
        reduced.append({
            "clip_id": "five-clip mean",
            "game": "spaceflight",
            "matcher": config_name,
            "bsp_pct": mean(f(row.get("bsp_pct")) for row in values),
            "match_rate": mean(f(row.get("match_rate")) for row in values),
            "rec_rate": mean(f(row.get("ref_ratio")) for row in values),
            "threshold_failure_per_10k": mean(f(row.get("artifact_incidents_per_10k_frames")) for row in values),
            "id_switches": mean(f(row.get("matcher_id_switches")) for row in values),
            "new_templates": mean(f(row.get("matcher_new_templates")) for row in values),
        })
    write_csv(out_tables / "matching_ablation.csv", reduced)
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Matcher} & \textbf{BSP} & \textbf{Match} & \textbf{Rec} & \textbf{SSIM-fail/10k} \\",
             r" & & \textbf{(\%)} & \textbf{rate} & \textbf{rate} & \\",
             r"\midrule"]
    for idx, row in enumerate(reduced):
        if idx and row["game"] != reduced[idx - 1]["game"]:
            lines.append(r"\midrule")
        matcher = row["matcher"].replace("_", r"\_")
        table_row = (
            f"{game_label(row['game'])} & {matcher} & {f(row['bsp_pct']):.2f} & {row['match_rate']:.3f} & "
            f"{row['rec_rate']:.3f} & {row['threshold_failure_per_10k']:.1f} \\\\"
        )
        lines.append(tex_table_row(row["game"], table_row))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "matching_ablation.tex").write_text("\n".join(lines) + "\n")


def decode_ms_by_game() -> dict[str, float]:
    """Mean FFmpeg software-decode ms/frame from the decode-only baseline CSV."""
    if DECODE_BASELINE is None or not Path(DECODE_BASELINE).is_file():
        return {}
    out: dict[str, float] = {}
    for row in read_csv(Path(DECODE_BASELINE)):
        out[game_label(row["game"])] = f(row["mean_decode_ms"])
    return out


def fair_spaceflight_client_row(resource_rows: list[dict[str, str]]) -> dict[str, str]:
    """Exp6-style path: Detect+Dict+Mask+Stitch, not wall-clock system_total_ms.

    The resource CSV ``system_total_ms`` (~115.60 ms five-clip mean) includes
    encode and other benchmark overhead. Reviewer Table 9 compares stitching
    to the exclusive processing path, so SC must use the same four-component
    sum as FC5/FM6/Mario (Detect 9.29 + Dict 12.34 + Mask 18.44 + Stitch 7.15).
    """
    detect = mean(f(row["detect_ms"]) for row in resource_rows)
    dict_ms = mean(f(row["dict_match_ms"]) for row in resource_rows)
    mask = mean(f(row["mask_fill_ms"]) for row in resource_rows)
    stitch = mean(f(row["stitch_ms"]) for row in resource_rows)
    fair = detect + dict_ms + mask + stitch
    p95s: list[float] = []
    if SPACEFLIGHT_RESOURCE is not None:
        root = Path(SPACEFLIGHT_RESOURCE).parent
        for row in resource_rows:
            report_path = root / row["clip_id"] / "report.json"
            if not report_path.is_file():
                continue
            frames = json.loads(report_path.read_text()).get("per_frame", [])
            fairs = sorted(
                f(fr.get("infer_ms")) + f(fr.get("dict_ms")) + f(fr.get("paint_ms")) + f(fr.get("recover_ms"))
                for fr in frames
            )
            if fairs:
                p95s.append(fairs[min(len(fairs) - 1, int(0.95 * len(fairs)))])
    return {
        "game": "spaceflight",
        "clip_id": "five-clip mean",
        "avg_system_ms": str(fair),
        "p95_system_ms": str(mean(p95s) if p95s else 0.0),
        "avg_detect_ms": str(detect),
        "avg_dict_ms": str(dict_ms),
        "avg_masking_ms": str(mask),
        "avg_stitching_ms": str(stitch),
        "path_scope": "detect+dict+mask+stitch",
    }


def build_client(out_tables: Path, out_figures: Path) -> None:
    rows = read_csv(EXP6)
    if SPACEFLIGHT_RESOURCE is not None:
        resource_rows = read_csv(SPACEFLIGHT_RESOURCE)
        if resource_rows:
            rows.append(fair_spaceflight_client_row(resource_rows))
    decode_ms = decode_ms_by_game()
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Decode} & \textbf{Stitch} & \textbf{Stitch/decode} & \textbf{DECAL path} & \textbf{Path FPS} \\",
             r" & \textbf{(ms)} & \textbf{(ms)} & \textbf{(\%)} & \textbf{(ms)} & \\",
             r"\midrule"]
    labels, components = [], []
    for row in rows:
        stitch = f(row["avg_stitching_ms"])
        system = f(row["avg_system_ms"])
        decode = decode_ms.get(game_label(row["game"]), 0.0)
        share = 100.0 * stitch / decode if decode else 0.0
        table_row = (
            f"{game_label(row['game'])} & {decode:.2f} & {stitch:.2f} & {share:.1f} & "
            f"{system:.2f} & {1000/system:.1f} \\\\"
        )
        lines.append(tex_table_row(row["game"], table_row))
        labels.append(game_label(row["game"]))
        components.append([f(row["avg_detect_ms"]), f(row["avg_dict_ms"]), f(row["avg_masking_ms"]), stitch])
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "client_stitch_cost.tex").write_text("\n".join(lines) + "\n")

    data = np.asarray(components)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bottom = np.zeros(len(labels))
    for idx, label in enumerate(["Detect", "Dict/match", "Mask/fill", "Client stitch"]):
        ax.bar(labels, data[:, idx], bottom=bottom, label=label, hatch="//" if idx == 3 else None)
        bottom += data[:, idx]
    for tick, label in zip(ax.get_xticklabels(), labels):
        if label == "SC":
            tick.set_color("red")
    ax.axhline(1000/60, color="black", ls=":", label="60 fps")
    ax.axhline(1000/30, color="black", ls="--", label="30 fps")
    ax.set_ylabel("DECAL-exclusive processing path (ms/frame)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out_figures / "client_stitch_component_breakdown.pdf")
    plt.close(fig)


def _mean_delay_series(
    rows: list[dict[str, str]],
    mode: str,
    field: str,
    scale: float = 1.0,
) -> list[float]:
    """Mean over FM6 clips and the 20/80/150 ms RTT grid at each delay."""
    ys: list[float] = []
    for delay in range(6):
        vals = [
            scale * f(row[field])
            for row in rows
            if row["game"] == "fm6"
            and row["cache_mode"] == mode
            and f(row["rtt_ms"]) in {20.0, 80.0, 150.0}
            and int(f(row["delay_rtts"])) == delay
        ]
        ys.append(mean(vals) if vals else float("nan"))
    return ys


def plot_rtt_delay_fm6(rows: list[dict[str, str]], out_path: Path) -> None:
    """Broken-axis net BSP plus the Raw-forced fraction that delay moves."""
    xs = list(range(6))
    cold_bsp = _mean_delay_series(rows, "cold_start", "final_net_saved_pct")
    warm_bsp = _mean_delay_series(rows, "warm_start", "final_net_saved_pct")
    cold_raw = _mean_delay_series(rows, "cold_start", "forced_raw_fraction", 100.0)
    warm_raw = _mean_delay_series(rows, "warm_start", "forced_raw_fraction", 100.0)
    cold_tpl_mb = _mean_delay_series(rows, "cold_start", "template_bytes_sent", 1e-6)

    # Okabe-Ito colorblind-safe pair (blue / orange).
    cold_color = "#0072B2"
    warm_color = "#E69F00"
    cold_label = "Cold 0% resident"
    warm_label = "Warm 100% observed templates"

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.18],
        wspace=0.36, hspace=0.08,
    )
    ax_warm = fig.add_subplot(gs[0, 0])
    ax_cold = fig.add_subplot(gs[1, 0], sharex=ax_warm)
    ax_raw = fig.add_subplot(gs[:, 1], sharex=ax_warm)

    ax_warm.plot(
        xs, warm_bsp, color=warm_color, ls="--", marker="s",
        lw=2.0, markersize=6, label=warm_label,
    )
    ax_warm.axhline(0, color="black", lw=0.8)
    ax_warm.set_ylim(-2.0, 20.0)
    ax_warm.set_ylabel("Net BSP (%)")
    ax_warm.grid(alpha=0.25)
    ax_warm.tick_params(labelbottom=False)
    ax_warm.spines["bottom"].set_visible(False)
    ax_warm.set_title("(a) Net BSP", loc="left", fontsize=10)
    ax_warm.annotate(
        f"{warm_bsp[0]:+.1f}%",
        xy=(5, warm_bsp[5]),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=warm_color,
    )

    ax_cold.plot(
        xs, cold_bsp, color=cold_color, ls="-", marker="o",
        lw=2.0, markersize=6, label=cold_label,
    )
    # Tight zoom: the 0.74 pp delay-0 to delay-1 step is visible but small.
    ax_cold.set_ylim(-1934.0, -1931.4)
    ax_cold.set_ylabel("Net BSP (%)")
    ax_cold.grid(alpha=0.25)
    ax_cold.spines["top"].set_visible(False)
    ax_cold.set_xlabel("RTT windows")
    delta_bsp = cold_bsp[5] - cold_bsp[0]
    tpl_gb = cold_tpl_mb[0] / 1000.0
    ax_cold.text(
        0.50, 0.16,
        f"PNG transfer (~{tpl_gb:.1f} GB) dwarfs delay\n"
        f"$\\Delta$ BSP = {delta_bsp:+.2f} pp across 0-5 RTT",
        transform=ax_cold.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )

    d = 0.018
    slash = dict(color="k", clip_on=False, lw=1.0)
    ax_warm.plot((-d, +d), (-d, +d), transform=ax_warm.transAxes, **slash)
    ax_warm.plot((1 - d, 1 + d), (-d, +d), transform=ax_warm.transAxes, **slash)
    ax_cold.plot((-d, +d), (1 - d, 1 + d), transform=ax_cold.transAxes, **slash)
    ax_cold.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_cold.transAxes, **slash)

    ax_raw.plot(
        xs, cold_raw, color=cold_color, ls="-", marker="o",
        lw=2.0, markersize=6, label=cold_label,
    )
    ax_raw.plot(
        xs, warm_raw, color=warm_color, ls="--", marker="s",
        lw=2.0, markersize=6, label=warm_label,
    )
    ax_raw.set_ylim(-5.0, 108.0)
    ax_raw.set_ylabel("Raw-forced frames (%)")
    ax_raw.set_xlabel("RTT windows")
    ax_raw.set_xticks(xs)
    ax_raw.grid(alpha=0.25)
    ax_raw.set_title("(b) What delay changes", loc="left", fontsize=10)
    ax_raw.annotate(
        "{:.0f}% -> {:.1f}%".format(cold_raw[0], cold_raw[1]),
        xy=(1, cold_raw[1]),
        xytext=(2.35, 55),
        fontsize=8,
        color=cold_color,
        arrowprops=dict(arrowstyle="->", color=cold_color, lw=1.1),
    )
    handles, labels = ax_raw.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, fontsize=8,
        frameon=False, bbox_to_anchor=(0.55, 0.995),
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.84, bottom=0.13)
    fig.savefig(out_path)
    plt.close(fig)


def build_measured_cache(out_tables: Path, out_figures: Path) -> None:
    rows = read_csv(MEASURED_OVERHEAD)
    if SPACEFLIGHT_OVERHEAD is not None:
        rows += read_csv(SPACEFLIGHT_OVERHEAD)
    partial = [row for row in rows if row["cache_mode"] == "partial_warm"]
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in partial:
        grouped[(row["game"], f(row["preload_fraction"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (game, preload), values in sorted(grouped.items()):
        summary.append({
            "game": game,
            "preload_fraction": preload,
            "clips": len(values),
            "preloaded_templates_mean": mean(f(row["preloaded_templates"]) for row in values),
            "observed_templates_mean": mean(f(row["ranked_templates_total"]) for row in values),
            "forced_raw_pct": 100.0 * mean(f(row["forced_raw_fraction"]) for row in values),
            "template_sent_mb": mean(f(row["template_bytes_sent"]) for row in values) / 1e6,
            "template_sent_pct": mean(f(row["template_bytes_sent_pct"]) for row in values),
            "net_saved_mb": mean(f(row["final_net_saved_bytes"]) for row in values) / 1e6,
            "net_saved_pct": mean(f(row["final_net_saved_pct"]) for row in values),
        })
    write_csv(out_tables / "measured_partial_warm_cache.csv", summary)
    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
        r"\textbf{Game} & \textbf{Preload} & \textbf{Pool} & \textbf{Raw forced} & \textbf{Tpl. sent} & \textbf{Net saved} & \textbf{Net BSP} \\",
        r" & & \textbf{templates} & \textbf{(\% frames)} & \textbf{(MB)} & \textbf{(MB)} & \textbf{(\%)} \\",
        r"\midrule",
    ]
    for idx, row in enumerate(summary):
        if idx and row["game"] != summary[idx - 1]["game"]:
            lines.append(r"\midrule")
        table_row = (
            f"{game_label(row['game'])} & {100*row['preload_fraction']:.0f}\\% & "
            f"{row['preloaded_templates_mean']:.1f}/{row['observed_templates_mean']:.1f} & "
            f"{row['forced_raw_pct']:.1f} & {row['template_sent_mb']:.2f} & "
            f"{tex_signed(row['net_saved_mb'])} & {tex_signed(row['net_saved_pct'])} \\\\"
        )
        lines.append(tex_table_row(row["game"], table_row))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "measured_partial_warm_cache.tex").write_text("\n".join(lines) + "\n")

    # Two-panel FM6 delay figure.  Cold BSP lives near -1933% because exact
    # PNG transfer dominates; that axis cannot show the delay mechanism.
    # Panel (a) splits warm/cold onto separate scales.  Panel (b) plots
    # Raw-forced frame fraction, which is what delay actually changes.
    plot_rtt_delay_fm6(rows, out_figures / "rtt_delay_sensitivity_fm6.pdf")

    fm6_partial = [row for row in summary if row["game"] == "fm6"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = [100 * row["preload_fraction"] for row in fm6_partial]
    ys = [row["net_saved_pct"] for row in fm6_partial]
    raw = [row["forced_raw_pct"] for row in fm6_partial]
    ax.bar(xs, ys, width=7, color="tab:green", alpha=.8, label="Net BSP")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Templates preloaded (% of reuse-ranked observed pool)")
    ax.set_ylabel("Net BSP (%)")
    ax2 = ax.twinx()
    ax2.plot(xs, raw, color="tab:red", marker="o", label="Forced Raw")
    ax2.set_ylabel("Forced-Raw frames (%)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out_figures / "cache_preload_net_savings.pdf")
    plt.close(fig)


def build_protocol_ledger(out: Path) -> None:
    rows = [
        {"artifact": "Headline per-clip/GOP BSP", "regime": "CRF23 open GOP, x264 medium, B=0", "clips": "5/title", "policy": "FC5 latent-key no-heal; FM6 latent-key heal-only; Mario pixel; SC heal-only multipart class 0", "accounting": "video + measured RMD; warm cache"},
        {"artifact": "Fixed-cap BSP/quality", "regime": "VBV 8/12/16/20/24 Mbps, x264 medium", "clips": "5/title", "policy": "Exp1 selected profile; SC heal-only multipart class 0", "accounting": "video + measured RMD"},
        {"artifact": "Cache/delay", "regime": "CRF23 open GOP", "clips": "5/title including SC GOP CRF23", "policy": "same as headline", "accounting": "video + RMD + exact referenced PNG bytes"},
        {"artifact": "Feather/fill/matcher", "regime": "CRF23 open GOP", "clips": "fc5_00/fm6_00; five SC clips for fill/matcher/feather", "policy": "FC5/FM6 no-heal; SC heal-only multipart class 0", "accounting": "within-table only; SC feather is dominant-fill five-clip mean"},
        {"artifact": "Client cost", "regime": "CRF23; Exp6 300-frame clip 00, SC five 909-frame clips", "clips": "00/title plus five-clip SC", "policy": "online-server scaffold", "accounting": "detect-to-stitch path (SC uses component sum, not 115.60 wall-clock); FFmpeg decode-only baseline of the same original_output.mp4 files"},
    ]
    write_csv(out / "evaluation_protocol_ledger.csv", rows)
    md = ["# Evaluation protocol ledger", "", "| Artifact | Encoder regime | Clips | Policy | Byte/timing scope |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        md.append(f"| {row['artifact']} | {row['regime']} | {row['clips']} | {row['policy']} | {row['accounting']} |")
    md += [
        "",
        "Cross-table rule: compare BSP values only when regime, clip intake, and policy match. Ablation values are within-table effects, not alternate headline configurations.",
        "",
        "## Star Citizen (SC) protocol differences",
        "- SC uses five non-overlapping 909-frame clips at 1280x720 and 30.303 fps; FC5/FM6 use their existing intakes and Mario retains its pixel defaults.",
        "- SC alone enables compound multipart regions for cockpit class 0 and remains heal-only. Multipart is not enabled for FC5 or FM6.",
        "- SC quality is reported without a fitted post-hoc uplift because no title-specific calibration exists. FC5/FM6/Mario retain their existing transforms.",
        "- CRF23 quality samples every fifth frame at 720-pixel VMAF scale; fixed-VBV quality uses all frames at 1080-pixel scale. Compare quality only within each table.",
        "- The 320-run learned-path gating grid is reused as a clip-independent exclusion; the SC rerun covers the 50 requested matching/fill cells.",
        "- Table 9 SC path is Detect+Dict+Mask+Stitch (47.21 ms), not resource `system_total_ms` (115.60 ms). Decode-only FFmpeg means are in `exp6/client_decode_baseline/`.",
    ]
    (out / "evaluation_protocol_ledger.md").write_text("\n".join(md) + "\n")


def main() -> None:
    global EXP35, EXP35_EXTRA, GENERALITY, RD_EXTRA, SPACEFLIGHT_RESOURCE, SPACEFLIGHT_ABLATION, SPACEFLIGHT_OVERHEAD, DECODE_BASELINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evaluation_revision")
    ap.add_argument("--exp35", type=Path, default=EXP35)
    ap.add_argument("--exp35-extra", type=Path, default=None)
    ap.add_argument("--generality", type=Path, default=GENERALITY)
    ap.add_argument("--rd-extra", type=Path, default=None)
    ap.add_argument("--spaceflight-resource", type=Path, default=None)
    ap.add_argument("--spaceflight-ablation", type=Path, default=None)
    ap.add_argument("--spaceflight-overhead", type=Path, default=None)
    ap.add_argument("--decode-baseline", type=Path, default=DECODE_BASELINE)
    args = ap.parse_args()
    EXP35 = args.exp35
    EXP35_EXTRA = args.exp35_extra
    GENERALITY = args.generality
    RD_EXTRA = args.rd_extra
    SPACEFLIGHT_RESOURCE = args.spaceflight_resource
    SPACEFLIGHT_ABLATION = args.spaceflight_ablation
    SPACEFLIGHT_OVERHEAD = args.spaceflight_overhead
    DECODE_BASELINE = args.decode_baseline
    tables = args.out_dir / "generated_eval_tables"
    figures = args.out_dir / "generated_eval_figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    build_protocol_ledger(args.out_dir)
    clip_rows, decal_by_clip = build_per_clip(tables)
    gops = build_gop(tables, figures, clip_rows, decal_by_clip)
    build_extreme_cases(tables, gops, decal_by_clip)
    build_vbv_tables(tables)
    quality_rows = build_crf_quality(tables)
    build_ablation_tables(tables)
    build_client(tables, figures)
    build_measured_cache(tables, figures)
    readme = [
        "# Four-title evaluation revision",
        "",
        "Generated by `tools/experiments/build_evaluation_revision.py`; do not hand-edit generated tables or figures.",
        "",
        f"- Headline CRF23 intake: `{EXP35}` plus `{EXP35_EXTRA}`",
        f"- Generality input: `{GENERALITY}`",
        f"- Fixed-VBV Star Citizen (SC) input: `{RD_EXTRA}`",
        f"- SC ablation input: `{SPACEFLIGHT_ABLATION}`",
        f"- SC resource input: `{SPACEFLIGHT_RESOURCE}`",
        f"- SC overhead input: `{SPACEFLIGHT_OVERHEAD}`",
        f"- Decode-only baseline: `{DECODE_BASELINE}`",
        "",
        "SC uses 1280x720@30.303 fps, five 909-frame clips, heal-only matching, and multipart class 0. "
        "FC5/FM6/Mario retain their prior protocol defaults. SC quality is direct measured quality with no fitted uplift.",
        "",
        "See `evaluation_protocol_ledger.md` for cross-table comparability and exclusions.",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "clips": len(clip_rows), "gops": len(gops), "quality_rows": quality_rows}, indent=2))


if __name__ == "__main__":
    main()
