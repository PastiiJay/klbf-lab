"""Deterministic multi-scene benchmark A/B and closed-loop optimizer (Python 3 stdlib).

Supports:
- `--mode ab`: interleaved base -> cand -> base runs across a JSON scene playlist.
- `--mode loop`: autonomous optimization loop invoking `--agent-cmd`, optional
  `--fast-cmd` offline pre-filter, 6 anti-reward-hacking gates, git reset/clean
  on rejection, and structured `experiments.jsonl` logging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HOST_LOG_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s+fps,\s+(\d+(?:\.\d+)?)\s+ms/frame\s+\((\d+(?:\.\d+)?)\s+(?:preparing:|ms/frame preparing on CPU)"
)
SUMMARIZE_HOST_RE = re.compile(
    r"Host timing\s*\((\d+)\s+samples,\s+medians\):\s+(\d+(?:\.\d+)?)\s+FPS,\s+"
    r"(\d+(?:\.\d+)?)\s+ms/frame,\s+(\d+(?:\.\d+)?)\s+ms/frame preparing on CPU"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class GateConfig:
    min_gain_pct: float = 2.0
    max_draw_delta_pct: float = 0.15
    max_1pct_drop_pct: float = 0.05
    max_vram_growth_pct: float = 0.05
    max_ram_growth_pct: float = 0.05
    max_vram_mb: float | None = None
    max_ram_mb: float | None = None
    min_png_luma: float = 4.0
    max_png_mae: float = 12.0
    timeout_s: float = 300.0
    repeats: int = 3
    skip_checks: bool = False


@dataclass
class SceneSpec:
    name: str
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    warmup: int = 30
    frames: int = 120
    input_script: str | None = None
    capture_dir: str | None = None
    baseline_png_dir: str | None = None
    timeout_s: float | None = None
    expected_draws: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SceneSpec":
        cmd = raw.get("cmd")
        if isinstance(cmd, str):
            cmd_list = cmd.split()
        elif isinstance(cmd, list) and cmd:
            cmd_list = [str(x) for x in cmd]
        else:
            raise ValueError(f"scene {raw.get('name', '?')!r} missing non-empty `cmd`")
        return cls(
            name=str(raw.get("name") or "scene"),
            cmd=cmd_list,
            env={str(k): str(v) for k, v in dict(raw.get("env") or {}).items()},
            cwd=str(raw["cwd"]) if raw.get("cwd") else None,
            warmup=int(raw.get("warmup", 30)),
            frames=int(raw.get("frames", 120)),
            input_script=str(raw["input_script"]) if raw.get("input_script") else None,
            capture_dir=str(raw["capture_dir"]) if raw.get("capture_dir") else None,
            baseline_png_dir=str(raw["baseline_png_dir"]) if raw.get("baseline_png_dir") else None,
            timeout_s=float(raw["timeout_s"]) if raw.get("timeout_s") is not None else None,
            expected_draws=float(raw["expected_draws"]) if raw.get("expected_draws") is not None else None,
        )


@dataclass
class RunMetrics:
    scene: str
    exit_code: int
    timed_out: bool
    world_frames: int
    warmup_frames: int
    median_frame_ms: float
    avg_frame_ms: float
    p95_frame_ms: float
    p99_frame_ms: float
    low_5pct_fps: float
    low_1pct_fps: float
    draws_avg: float
    draws_min: int
    draws_max: int
    zero_draw_world_frames: int
    black_frames: int
    device_removed: bool
    vram_used_peak_mb: float
    vram_budget_mb: float
    ram_private_commit_peak_mb: float
    ram_working_set_peak_mb: float
    input_script_sha256: str | None = None
    preparing_ms: float | None = None
    png_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    corrupt_json: str | None = None
    input_script_mismatch: bool = False
    capture_missing: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def decode_png_rgb(path: Path) -> tuple[int, int, bytes]:
    """Decode an 8-bit RGB or RGBA PNG into packed RGB bytes using Python stdlib."""
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{path} is not a PNG file")
    pos = 8
    width = height = 0
    bit_depth = color_type = 0
    idat = bytearray()
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        pos += 8
        chunk = data[pos : pos + length]
        pos += length + 4  # skip CRC
        if kind == b"IHDR":
            width, height, bit_depth, color_type, comp, flt, interlace = struct.unpack(">IIBBBBB", chunk)
            if bit_depth != 8 or color_type not in (2, 6) or comp != 0 or flt != 0 or interlace != 0:
                raise ValueError(f"{path}: unsupported PNG format depth={bit_depth} color={color_type}")
        elif kind == b"IDAT":
            idat.extend(chunk)
        elif kind == b"IEND":
            break
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: missing IHDR dimensions")
    bpp = 3 if color_type == 2 else 4
    stride = width * bpp
    raw = zlib.decompress(bytes(idat))
    expected = height * (stride + 1)
    if len(raw) < expected:
        raise ValueError(f"{path}: truncated IDAT payload ({len(raw)} < {expected})")

    out = bytearray(width * height * 3)
    prev_row = bytearray(stride)
    for y in range(height):
        row_start = y * (stride + 1)
        flt_type = raw[row_start]
        src = raw[row_start + 1 : row_start + 1 + stride]
        recon = bytearray(stride)
        for i in range(stride):
            a = recon[i - bpp] if i >= bpp else 0
            b = prev_row[i]
            c = prev_row[i - bpp] if i >= bpp else 0
            x = src[i]
            if flt_type == 0:
                recon[i] = x
            elif flt_type == 1:
                recon[i] = (x + a) & 0xFF
            elif flt_type == 2:
                recon[i] = (x + b) & 0xFF
            elif flt_type == 3:
                recon[i] = (x + ((a + b) >> 1)) & 0xFF
            elif flt_type == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                recon[i] = (x + pr) & 0xFF
            else:
                raise ValueError(f"{path}: unsupported PNG filter {flt_type}")
        prev_row = recon
        if bpp == 3:
            out[y * width * 3 : (y + 1) * width * 3] = recon
        else:
            dst_base = y * width * 3
            for x_px in range(width):
                out[dst_base + x_px * 3 : dst_base + x_px * 3 + 3] = recon[x_px * 4 : x_px * 4 + 3]
    return width, height, bytes(out)


def png_mean_luma(rgb: bytes) -> float:
    if not rgb:
        return 0.0
    total = 0.0
    count = len(rgb) // 3
    for i in range(0, count * 3, 3):
        total += 0.2126 * rgb[i] + 0.7152 * rgb[i + 1] + 0.0722 * rgb[i + 2]
    return total / count


def png_mae(a: bytes, b: bytes) -> float:
    if len(a) != len(b) or not a:
        return 255.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def inspect_capture_dir(capture_dir: Path, baseline_dir: Path | None) -> dict[str, dict[str, float]]:
    if not capture_dir.is_dir():
        return {}
    stats: dict[str, dict[str, float]] = {}
    for png_path in sorted(capture_dir.glob("*.png")):
        w, h, rgb = decode_png_rgb(png_path)
        entry: dict[str, float] = {
            "width": float(w),
            "height": float(h),
            "mean_luma": png_mean_luma(rgb),
        }
        if baseline_dir is not None:
            ref_path = baseline_dir / png_path.name
            if ref_path.is_file():
                rw, rh, ref_rgb = decode_png_rgb(ref_path)
                if (rw, rh) == (w, h):
                    entry["mae"] = png_mae(rgb, ref_rgb)
                else:
                    entry["mae"] = 255.0
        stats[png_path.name] = entry
    return stats


def parse_metrics_payload(payload: dict[str, Any], scene: str, exit_code: int, timed_out: bool) -> RunMetrics:
    """Parse `bench.rs` JSON or `fps-report` JSON metrics into RunMetrics."""
    frame_ms = payload.get("frame_ms")
    if isinstance(frame_ms, dict):
        avg_ms = float(frame_ms.get("avg", 0.0))
        med_ms = float(frame_ms.get("median", avg_ms))
        p95_ms = float(frame_ms.get("p95", med_ms))
        p99_ms = float(frame_ms.get("p99", p95_ms))
        low5 = float(frame_ms.get("low_5pct_fps", 1000.0 / p95_ms if p95_ms > 0 else 0.0))
        low1 = float(frame_ms.get("low_1pct_fps", 1000.0 / p99_ms if p99_ms > 0 else 0.0))
    else:
        med_ms = float(payload.get("median_frame_ms", payload.get("frame_ms_median", 0.0)))
        avg_ms = float(payload.get("avg_frame_ms", med_ms))
        p95_ms = float(payload.get("p95_frame_ms", med_ms))
        p99_ms = float(payload.get("p99_frame_ms", p95_ms))
        low5 = float(payload.get("low_5pct_fps", 1000.0 / p95_ms if p95_ms > 0 else 0.0))
        low1 = float(payload.get("low_1pct_fps", 1000.0 / p99_ms if p99_ms > 0 else 0.0))

    draws = payload.get("draws")
    if isinstance(draws, dict):
        draws_avg = float(draws.get("avg", 0.0))
        draws_min = int(draws.get("min", 0))
        draws_max = int(draws.get("max", 0))
    else:
        draws_avg = float(payload.get("avg_draws", payload.get("draws_avg", 0.0)))
        draws_min = int(payload.get("draws_min", draws_avg))
        draws_max = int(payload.get("draws_max", draws_avg))

    mem = payload.get("memory") if isinstance(payload.get("memory"), dict) else payload
    preparing_ms = payload.get("preparing_ms")
    if preparing_ms is None and isinstance(payload.get("framelog"), dict):
        preparing_ms = payload["framelog"].get("prepare_ms")

    return RunMetrics(
        scene=scene,
        exit_code=exit_code,
        timed_out=timed_out,
        world_frames=int(payload.get("world_frames", 0)),
        warmup_frames=int(payload.get("warmup_frames", 0)),
        median_frame_ms=med_ms,
        avg_frame_ms=avg_ms,
        p95_frame_ms=p95_ms,
        p99_frame_ms=p99_ms,
        low_5pct_fps=low5,
        low_1pct_fps=low1,
        draws_avg=draws_avg,
        draws_min=draws_min,
        draws_max=draws_max,
        zero_draw_world_frames=int(payload.get("zero_draw_world_frames", 0)),
        black_frames=int(payload.get("black_frames", 0)),
        device_removed=bool(payload.get("device_removed", False)),
        vram_used_peak_mb=float(mem.get("vram_used_peak_mb", 0.0)),
        vram_budget_mb=float(mem.get("vram_budget_mb", 0.0)),
        ram_private_commit_peak_mb=float(mem.get("ram_private_commit_peak_mb", 0.0)),
        ram_working_set_peak_mb=float(mem.get("ram_working_set_peak_mb", 0.0)),
        input_script_sha256=payload.get("input_script_sha256"),
        preparing_ms=float(preparing_ms) if preparing_ms is not None else None,
        raw=payload,
    )


def parse_summarize_text(text: str, scene: str, exit_code: int, timed_out: bool) -> RunMetrics | None:
    """Fallback parser for `summarize_log.py` summary lines or raw host timing logs."""
    clean = ANSI_RE.sub("", text)
    m = SUMMARIZE_HOST_RE.search(clean)
    if m:
        samples = int(m.group(1))
        fps = float(m.group(2))
        ms = float(m.group(3))
        cpu = float(m.group(4))
        return RunMetrics(
            scene=scene,
            exit_code=exit_code,
            timed_out=timed_out,
            world_frames=samples,
            warmup_frames=0,
            median_frame_ms=ms,
            avg_frame_ms=ms,
            p95_frame_ms=ms,
            p99_frame_ms=ms,
            low_5pct_fps=fps,
            low_1pct_fps=fps,
            draws_avg=0.0,
            draws_min=0,
            draws_max=0,
            zero_draw_world_frames=0,
            black_frames=0,
            device_removed=False,
            vram_used_peak_mb=0.0,
            vram_budget_mb=0.0,
            ram_private_commit_peak_mb=0.0,
            ram_working_set_peak_mb=0.0,
            preparing_ms=cpu,
        )

    samples_ms: list[float] = []
    samples_cpu: list[float] = []
    for line in clean.splitlines():
        hm = HOST_LOG_RE.search(line)
        if hm:
            samples_ms.append(float(hm.group(2)))
            samples_cpu.append(float(hm.group(3)))
    if not samples_ms:
        return None
    med_ms = statistics.median(samples_ms)
    avg_ms = statistics.mean(samples_ms)
    sorted_ms = sorted(samples_ms)
    p95 = sorted_ms[min(len(sorted_ms) - 1, round(0.95 * (len(sorted_ms) - 1)))]
    p99 = sorted_ms[min(len(sorted_ms) - 1, round(0.99 * (len(sorted_ms) - 1)))]
    return RunMetrics(
        scene=scene,
        exit_code=exit_code,
        timed_out=timed_out,
        world_frames=len(samples_ms),
        warmup_frames=0,
        median_frame_ms=med_ms,
        avg_frame_ms=avg_ms,
        p95_frame_ms=p95,
        p99_frame_ms=p99,
        low_5pct_fps=1000.0 / p95 if p95 > 0 else 0.0,
        low_1pct_fps=1000.0 / p99 if p99 > 0 else 0.0,
        draws_avg=0.0,
        draws_min=0,
        draws_max=0,
        zero_draw_world_frames=0,
        black_frames=0,
        device_removed=False,
        vram_used_peak_mb=0.0,
        vram_budget_mb=0.0,
        ram_private_commit_peak_mb=0.0,
        ram_working_set_peak_mb=0.0,
        preparing_ms=statistics.median(samples_cpu),
    )


def _empty_failed_metrics(scene_name: str, exit_code: int, timed_out: bool, corrupt_json: str | None = None) -> RunMetrics:
    return RunMetrics(
        scene=scene_name,
        exit_code=exit_code,
        timed_out=timed_out,
        world_frames=0,
        warmup_frames=0,
        median_frame_ms=0.0,
        avg_frame_ms=0.0,
        p95_frame_ms=0.0,
        p99_frame_ms=0.0,
        low_5pct_fps=0.0,
        low_1pct_fps=0.0,
        draws_avg=0.0,
        draws_min=0,
        draws_max=0,
        zero_draw_world_frames=0,
        black_frames=0,
        device_removed=False,
        vram_used_peak_mb=0.0,
        vram_budget_mb=0.0,
        ram_private_commit_peak_mb=0.0,
        ram_working_set_peak_mb=0.0,
        corrupt_json=corrupt_json,
    )


def run_scene(
    scene: SceneSpec,
    extra_env: dict[str, str],
    timeout_s: float,
    work_dir: Path,
) -> RunMetrics:
    stamp = f"{scene.name}-{os.getpid()}-{time.time_ns()}"
    out_json = work_dir / f"{stamp}.json"
    env = os.environ.copy()
    env.update(scene.env)
    env.update(extra_env)
    env["VICTORY_BENCH_OUT"] = str(out_json)
    env["VICTORY_BENCH_WARMUP"] = str(scene.warmup)
    env["VICTORY_BENCH_EXIT_FRAMES"] = str(scene.frames)
    if scene.input_script:
        env["VICTORY_INPUT_REPLAY"] = scene.input_script

    run_cap_dir: Path | None = None
    if scene.capture_dir or scene.baseline_png_dir:
        # Isolate every single run into its own clean capture directory so base/cand/repeats never cross-contaminate.
        run_cap_dir = work_dir / f"cap-{stamp}"
        run_cap_dir.mkdir(parents=True, exist_ok=True)
        env["VICTORY_CAPTURE"] = str(run_cap_dir)
        env.setdefault("VICTORY_CAPTURE_WORLD", "1")

    eff_timeout = scene.timeout_s if scene.timeout_s is not None else timeout_s
    timed_out = False
    stdout_text = ""
    try:
        proc = subprocess.run(
            scene.cmd,
            cwd=scene.cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=eff_timeout,
        )
        exit_code = proc.returncode
        stdout_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""

    if out_json.is_file():
        try:
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("benchmark JSON root must be an object")
            metrics = parse_metrics_payload(payload, scene.name, exit_code, timed_out)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            metrics = _empty_failed_metrics(
                scene.name,
                exit_code if exit_code != 0 else 1,
                timed_out,
                corrupt_json=str(exc),
            )
    else:
        parsed = parse_summarize_text(stdout_text, scene.name, exit_code, timed_out)
        metrics = parsed if parsed is not None else _empty_failed_metrics(scene.name, exit_code, timed_out)

    if run_cap_dir is not None:
        base_dir = Path(scene.baseline_png_dir) if scene.baseline_png_dir else None
        metrics.png_stats = inspect_capture_dir(run_cap_dir, base_dir)
        if base_dir is not None and base_dir.is_dir():
            expected_names = {p.name for p in base_dir.glob("*.png")}
            if expected_names and not expected_names.issubset(metrics.png_stats.keys()):
                metrics.capture_missing = True
        elif scene.capture_dir and not metrics.png_stats:
            metrics.capture_missing = True

    return metrics


def aggregate_runs(runs: list[RunMetrics]) -> RunMetrics:
    assert runs
    first = runs[0]
    png_agg: dict[str, dict[str, float]] = {}
    all_pngs = {name for r in runs for name in r.png_stats}
    for name in sorted(all_pngs):
        lumas = [r.png_stats[name]["mean_luma"] for r in runs if name in r.png_stats]
        maes = [r.png_stats[name]["mae"] for r in runs if name in r.png_stats and "mae" in r.png_stats[name]]
        entry: dict[str, float] = {"mean_luma": min(lumas)}
        if maes:
            entry["mae"] = max(maes)
        png_agg[name] = entry

    preps = [r.preparing_ms for r in runs if r.preparing_ms is not None]
    shas = [r.input_script_sha256 for r in runs]
    return RunMetrics(
        scene=first.scene,
        exit_code=next((r.exit_code for r in runs if r.exit_code != 0), 0),
        timed_out=any(r.timed_out for r in runs),
        world_frames=min(r.world_frames for r in runs),
        warmup_frames=min(r.warmup_frames for r in runs),
        median_frame_ms=float(statistics.median(r.median_frame_ms for r in runs)),
        avg_frame_ms=float(statistics.median(r.avg_frame_ms for r in runs)),
        p95_frame_ms=float(statistics.median(r.p95_frame_ms for r in runs)),
        p99_frame_ms=float(statistics.median(r.p99_frame_ms for r in runs)),
        low_5pct_fps=float(statistics.median(r.low_5pct_fps for r in runs)),
        low_1pct_fps=float(statistics.median(r.low_1pct_fps for r in runs)),
        draws_avg=float(statistics.median(r.draws_avg for r in runs)),
        draws_min=min(r.draws_min for r in runs),
        draws_max=max(r.draws_max for r in runs),
        zero_draw_world_frames=max(r.zero_draw_world_frames for r in runs),
        black_frames=max(r.black_frames for r in runs),
        device_removed=any(r.device_removed for r in runs),
        vram_used_peak_mb=max(r.vram_used_peak_mb for r in runs),
        vram_budget_mb=min(r.vram_budget_mb for r in runs),
        ram_private_commit_peak_mb=max(r.ram_private_commit_peak_mb for r in runs),
        ram_working_set_peak_mb=max(r.ram_working_set_peak_mb for r in runs),
        input_script_sha256=first.input_script_sha256,
        preparing_ms=float(statistics.median(preps)) if preps else None,
        png_stats=png_agg,
        corrupt_json=next((r.corrupt_json for r in runs if r.corrupt_json), None),
        input_script_mismatch=len(set(shas)) > 1,
        capture_missing=any(r.capture_missing for r in runs),
    )


def run_static_checks(check_cmds: list[list[str]], cwd: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for cmd in check_cmds:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()[-1000:]
            errors.append(f"check `{' '.join(cmd)}` failed (exit {proc.returncode}): {tail}")
            return False, errors
    return True, errors


def evaluate_gates(
    base_scenes: dict[str, RunMetrics],
    cand_scenes: dict[str, RunMetrics],
    scenes: list[SceneSpec],
    cfg: GateConfig,
    static_ok: bool,
    static_errors: list[str],
    repeats_executed: int,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate all 6 anti-reward-hacking gates across the playlist."""
    gates: dict[str, dict[str, Any]] = {}

    gates["gate1_static_checks"] = {"passed": static_ok, "errors": static_errors}

    stability_errors: list[str] = []
    draw_errors: list[str] = []
    visual_errors: list[str] = []
    warmup_errors: list[str] = []
    perf_errors: list[str] = []
    scene_gains: list[float] = []

    for spec in scenes:
        b = base_scenes[spec.name]
        c = cand_scenes[spec.name]

        # Gate 2: exit 0, no timeout, no device_removed, valid JSON output
        if c.timed_out or c.exit_code != 0 or c.device_removed or c.corrupt_json:
            stability_errors.append(
                f"{spec.name}: exit={c.exit_code}, timed_out={c.timed_out}, "
                f"device_removed={c.device_removed}, corrupt_json={c.corrupt_json}"
            )
        if b.timed_out or b.exit_code != 0 or b.device_removed or b.corrupt_json:
            stability_errors.append(
                f"{spec.name} (baseline): exit={b.exit_code}, timed_out={b.timed_out}, "
                f"device_removed={b.device_removed}, corrupt_json={b.corrupt_json}"
            )

        # Gate 3: avg_draws within +-15% AND input_script_sha256 parity
        ref_draws = spec.expected_draws if spec.expected_draws is not None else b.draws_avg
        if ref_draws > 0:
            delta = abs(c.draws_avg - ref_draws) / ref_draws
            if delta > cfg.max_draw_delta_pct:
                draw_errors.append(
                    f"{spec.name}: draws_avg={c.draws_avg:.1f} vs ref={ref_draws:.1f} "
                    f"(delta {delta * 100:.1f}% > {cfg.max_draw_delta_pct * 100:.1f}%)"
                )
        if c.input_script_mismatch or b.input_script_mismatch:
            draw_errors.append(f"{spec.name}: input_script_sha256 varied across repeats")
        if spec.input_script and not c.input_script_sha256:
            draw_errors.append(f"{spec.name}: input_script configured but candidate reported null input_script_sha256")
        if b.input_script_sha256 and c.input_script_sha256 != b.input_script_sha256:
            draw_errors.append(
                f"{spec.name}: input_script_sha256 mismatch (cand={c.input_script_sha256!r} vs base={b.input_script_sha256!r})"
            )

        # Gate 4: visual sanity + isolated PNG capture parity
        if c.zero_draw_world_frames > 0:
            visual_errors.append(f"{spec.name}: zero_draw_world_frames={c.zero_draw_world_frames}")
        if c.black_frames > 0:
            visual_errors.append(f"{spec.name}: black_frames={c.black_frames}")
        if c.capture_missing:
            visual_errors.append(f"{spec.name}: expected capture PNG(s) missing in isolated capture directory")
        if spec.baseline_png_dir:
            base_dir = Path(spec.baseline_png_dir)
            for ref_png in sorted(base_dir.glob("*.png")):
                if ref_png.name not in c.png_stats:
                    visual_errors.append(f"{spec.name}: missing baseline checkpoint PNG {ref_png.name}")
        elif spec.capture_dir and not c.png_stats:
            visual_errors.append(f"{spec.name}: capture_dir set but 0 PNGs produced")

        for png_name, stats in c.png_stats.items():
            if stats["mean_luma"] < cfg.min_png_luma:
                visual_errors.append(
                    f"{spec.name} {png_name}: mean_luma={stats['mean_luma']:.2f} < {cfg.min_png_luma:.2f}"
                )
            if "mae" in stats and stats["mae"] > cfg.max_png_mae:
                visual_errors.append(
                    f"{spec.name} {png_name}: mae={stats['mae']:.2f} > {cfg.max_png_mae:.2f}"
                )

        # Gate 5: warmup exclusion + multi-run median
        required_frames = max(1, spec.frames)
        for side, run in (("baseline", b), ("candidate", c)):
            if run.warmup_frames < spec.warmup:
                warmup_errors.append(f"{spec.name} ({side}): warmup_frames={run.warmup_frames} < {spec.warmup}")
            if run.world_frames < required_frames:
                warmup_errors.append(f"{spec.name} ({side}): world_frames={run.world_frames} < {required_frames}")
            if run.median_frame_ms <= 0.0:
                warmup_errors.append(f"{spec.name} ({side}): missing valid median_frame_ms")

        # Gate 6: composite score & non-regression guardrails
        if b.median_frame_ms > 0:
            gain_pct = ((b.median_frame_ms - c.median_frame_ms) / b.median_frame_ms) * 100.0
        else:
            gain_pct = 0.0
        scene_gains.append(gain_pct)

        if b.low_1pct_fps > 0 and c.low_1pct_fps < b.low_1pct_fps * (1.0 - cfg.max_1pct_drop_pct):
            perf_errors.append(
                f"{spec.name}: low_1pct_fps={c.low_1pct_fps:.2f} regressed vs base={b.low_1pct_fps:.2f}"
            )
        if cfg.max_vram_mb is not None and c.vram_used_peak_mb > cfg.max_vram_mb:
            perf_errors.append(
                f"{spec.name}: vram_used_peak_mb={c.vram_used_peak_mb:.1f} > cap={cfg.max_vram_mb:.1f}"
            )
        if b.vram_used_peak_mb > 0 and c.vram_used_peak_mb > b.vram_used_peak_mb * (1.0 + cfg.max_vram_growth_pct):
            perf_errors.append(
                f"{spec.name}: vram_used_peak_mb={c.vram_used_peak_mb:.1f} grew > "
                f"{cfg.max_vram_growth_pct * 100:.1f}% vs base={b.vram_used_peak_mb:.1f}"
            )
        if cfg.max_ram_mb is not None and c.ram_private_commit_peak_mb > cfg.max_ram_mb:
            perf_errors.append(
                f"{spec.name}: ram_private_commit_peak_mb={c.ram_private_commit_peak_mb:.1f} > cap={cfg.max_ram_mb:.1f}"
            )
        if (
            b.ram_private_commit_peak_mb > 0
            and c.ram_private_commit_peak_mb > b.ram_private_commit_peak_mb * (1.0 + cfg.max_ram_growth_pct)
        ):
            perf_errors.append(
                f"{spec.name}: ram_private_commit_peak_mb={c.ram_private_commit_peak_mb:.1f} grew > "
                f"{cfg.max_ram_growth_pct * 100:.1f}% vs base={b.ram_private_commit_peak_mb:.1f}"
            )

    overall_gain = float(statistics.median(scene_gains)) if scene_gains else 0.0
    if overall_gain < cfg.min_gain_pct:
        perf_errors.append(f"median gain {overall_gain:.2f}% < min_gain_pct {cfg.min_gain_pct:.2f}%")
    if repeats_executed < 1:
        warmup_errors.append("no benchmark repeats executed")

    gates["gate2_stability"] = {"passed": not stability_errors, "errors": stability_errors}
    gates["gate3_draw_parity"] = {"passed": not draw_errors, "errors": draw_errors}
    gates["gate4_visual_sanity"] = {"passed": not visual_errors, "errors": visual_errors}
    gates["gate5_warmup_median"] = {
        "passed": not warmup_errors,
        "repeats": repeats_executed,
        "errors": warmup_errors,
    }
    gates["gate6_composite_score"] = {
        "passed": not perf_errors,
        "gain_pct": round(overall_gain, 3),
        "errors": perf_errors,
    }

    accepted = all(g["passed"] for g in gates.values())
    return accepted, gates


def run_interleaved_ab(
    scenes: list[SceneSpec],
    base_env: dict[str, str],
    cand_env: dict[str, str],
    repeats: int,
    timeout_s: float,
) -> tuple[dict[str, RunMetrics], dict[str, RunMetrics]]:
    base_runs: dict[str, list[RunMetrics]] = {s.name: [] for s in scenes}
    cand_runs: dict[str, list[RunMetrics]] = {s.name: [] for s in scenes}
    with tempfile.TemporaryDirectory(prefix="klbf-bench-") as tmp:
        tmp_dir = Path(tmp)
        for rep in range(repeats):
            for scene in scenes:
                order = [("base", base_env, base_runs), ("cand", cand_env, cand_runs)]
                if rep % 2 == 1:
                    order.reverse()
                for _, extra_env, bucket in order:
                    bucket[scene.name].append(run_scene(scene, extra_env, timeout_s, tmp_dir))
    return (
        {name: aggregate_runs(items) for name, items in base_runs.items()},
        {name: aggregate_runs(items) for name, items in cand_runs.items()},
    )


def git_info(repo: Path) -> tuple[str, str]:
    sha = "unknown"
    stat = ""
    try:
        p1 = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        if p1.returncode == 0:
            sha = p1.stdout.strip()
        p2 = subprocess.run(["git", "-C", str(repo), "diff", "--shortstat"], capture_output=True, text=True)
        if p2.returncode == 0:
            stat = p2.stdout.strip()
    except OSError:
        pass
    return sha, stat


def git_snapshot(repo: Path) -> tuple[str | None, set[str]]:
    """Record pre-iteration HEAD SHA and untracked file paths for clean rollback."""
    try:
        head_proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
        pre_sha = head_proc.stdout.strip() if head_proc.returncode == 0 else None
        untracked_proc = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True,
        )
        untracked: set[str] = set()
        if untracked_proc.returncode == 0:
            for raw in untracked_proc.stdout.split(b"\x00"):
                if raw:
                    untracked.add(raw.decode("utf-8", errors="replace"))
        return pre_sha, untracked
    except OSError:
        return None, set()


def revert_repo(
    repo: Path,
    revert_cmd: str | None,
    pre_sha: str | None = None,
    pre_untracked: set[str] | None = None,
) -> None:
    if revert_cmd:
        subprocess.run(revert_cmd, shell=True, cwd=str(repo), check=False)
        return
    if pre_sha:
        subprocess.run(["git", "-C", str(repo), "reset", "--hard", pre_sha], check=False)
        untracked_proc = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True,
        )
        keep_untracked = pre_untracked or set()
        if untracked_proc.returncode == 0:
            for raw in untracked_proc.stdout.split(b"\x00"):
                if not raw:
                    continue
                rel = raw.decode("utf-8", errors="replace")
                if rel not in keep_untracked:
                    target = repo / rel
                    if target.is_file() or target.is_symlink():
                        target.unlink(missing_ok=True)
    else:
        subprocess.run(["git", "-C", str(repo), "checkout", "--", "."], check=False)


def metrics_summary_dict(scenes: dict[str, RunMetrics]) -> dict[str, Any]:
    return {
        name: {
            "median_frame_ms": round(m.median_frame_ms, 3),
            "low_1pct_fps": round(m.low_1pct_fps, 3),
            "draws_avg": round(m.draws_avg, 1),
            "vram_used_peak_mb": round(m.vram_used_peak_mb, 2),
            "ram_private_commit_peak_mb": round(m.ram_private_commit_peak_mb, 2),
            "preparing_ms": m.preparing_ms,
        }
        for name, m in scenes.items()
    }


def append_experiment_log(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=False) + "\n")


def load_playlist(path: Path) -> tuple[list[SceneSpec], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_scenes = data.get("scenes") if isinstance(data, dict) else data
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError(f"{path} must contain a non-empty `scenes` list")
    scenes = [SceneSpec.from_dict(s) for s in raw_scenes]
    meta = data if isinstance(data, dict) else {}
    return scenes, meta


def default_check_commands(build_cmd: str | None) -> list[list[str]]:
    cmds: list[list[str]] = []
    if build_cmd:
        cmds.append(shlex.split(build_cmd))
    cmds.extend(
        [
            ["cargo", "run", "--locked", "-p", "xtask", "--", "policy"],
            ["cargo", "clippy", "--workspace", "--all-targets", "--locked", "--", "-D", "warnings", "-A", "deprecated"],
            ["cargo", "test", "--locked", "-p", "victory-gpu", "-p", "victory-hle", "--", "--test-threads=1"],
        ]
    )
    return cmds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ab", "loop"], required=True)
    parser.add_argument("--config", type=Path, required=True, help="JSON playlist with `scenes`")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-gain-pct", type=float, default=2.0)
    parser.add_argument("--max-draw-delta-pct", type=float, default=0.15)
    parser.add_argument("--max-1pct-drop-pct", type=float, default=0.05)
    parser.add_argument("--max-vram-growth-pct", type=float, default=0.05)
    parser.add_argument("--max-ram-growth-pct", type=float, default=0.05)
    parser.add_argument("--max-vram-mb", type=float, default=None)
    parser.add_argument("--max-ram-mb", type=float, default=None)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--base-env", action="append", default=[], help="KEY=VAL for baseline")
    parser.add_argument("--cand-env", action="append", default=[], help="KEY=VAL for candidate")
    parser.add_argument("--build-cmd", type=str, default="cargo build --release --locked", help="Build command for Gate 1")
    parser.add_argument("--check-cmd", action="append", default=[], help="Custom static check command")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--fast-cmd", type=str, default=None, help="Optional fast offline pre-filter command")
    parser.add_argument("--agent-cmd", type=str, default=None, help="Agent command for --mode loop")
    parser.add_argument("--revert-cmd", type=str, default=None, help="Custom revert command on rejection")
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--experiments-log",
        type=Path,
        default=Path(tempfile.gettempdir()) / "klbf-experiments.jsonl",
    )
    args = parser.parse_args(argv)

    def parse_kv(items: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in items:
            if "=" not in item:
                parser.error(f"expected KEY=VAL, got {item!r}")
            k, v = item.split("=", 1)
            out[k] = v
        return out

    scenes, meta = load_playlist(args.config)
    base_env = {**{str(k): str(v) for k, v in dict(meta.get("base_env") or {}).items()}, **parse_kv(args.base_env)}
    cand_env = {**{str(k): str(v) for k, v in dict(meta.get("cand_env") or {}).items()}, **parse_kv(args.cand_env)}

    cfg = GateConfig(
        min_gain_pct=args.min_gain_pct,
        max_draw_delta_pct=args.max_draw_delta_pct,
        max_1pct_drop_pct=args.max_1pct_drop_pct,
        max_vram_growth_pct=args.max_vram_growth_pct,
        max_ram_growth_pct=args.max_ram_growth_pct,
        max_vram_mb=args.max_vram_mb if args.max_vram_mb is not None else meta.get("max_vram_mb"),
        max_ram_mb=args.max_ram_mb if args.max_ram_mb is not None else meta.get("max_ram_mb"),
        timeout_s=args.timeout_s,
        repeats=max(1, args.repeats),
        skip_checks=args.skip_checks,
    )
    check_cmds = [shlex.split(c) for c in args.check_cmd] if args.check_cmd else default_check_commands(args.build_cmd)

    if args.mode == "ab":
        static_ok, static_errors = (True, []) if cfg.skip_checks else run_static_checks(check_cmds, args.repo)
        base_med, cand_med = run_interleaved_ab(scenes, base_env, cand_env, cfg.repeats, cfg.timeout_s)
        accepted, gates = evaluate_gates(
            base_med, cand_med, scenes, cfg, static_ok, static_errors, cfg.repeats
        )
        sha, stat = git_info(args.repo)
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "ab",
            "iteration": 0,
            "hypothesis": str(meta.get("hypothesis") or "interleaved A/B"),
            "git_sha": sha,
            "diff_stat": stat,
            "baseline": metrics_summary_dict(base_med),
            "candidate": metrics_summary_dict(cand_med),
            "gates": gates,
            "decision": "keep" if accepted else "reject",
        }
        append_experiment_log(args.experiments_log, record)
        print(json.dumps(record, indent=2))
        return 0 if accepted else 1

    if not args.agent_cmd:
        parser.error("--agent-cmd is required for --mode loop")

    with tempfile.TemporaryDirectory(prefix="klbf-base-") as tmp:
        tmp_dir = Path(tmp)
        base_med = {
            s.name: aggregate_runs([run_scene(s, base_env, cfg.timeout_s, tmp_dir) for _ in range(cfg.repeats)])
            for s in scenes
        }

    kept_any = False
    for it in range(1, args.max_iters + 1):
        pre_sha, pre_untracked = git_snapshot(args.repo)
        agent_env = os.environ.copy()
        agent_env["BENCH_ITERATION"] = str(it)
        agent_env["BENCH_EXPERIMENTS_LOG"] = str(args.experiments_log)
        agent_env["BENCH_BASELINE_JSON"] = json.dumps(metrics_summary_dict(base_med))
        agent_proc = subprocess.run(
            args.agent_cmd,
            shell=True,
            cwd=str(args.repo),
            env=agent_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines = (agent_proc.stdout or "").strip().splitlines()
        hypothesis = lines[-1] if lines else f"iteration-{it}"
        sha, stat = git_info(args.repo)

        if agent_proc.returncode != 0:
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": "loop",
                "iteration": it,
                "hypothesis": hypothesis,
                "git_sha": sha,
                "diff_stat": stat,
                "gates": {"agent_cmd": {"passed": False, "exit_code": agent_proc.returncode}},
                "decision": "revert",
            }
            append_experiment_log(args.experiments_log, record)
            revert_repo(args.repo, args.revert_cmd, pre_sha, pre_untracked)
            continue

        if args.fast_cmd:
            fast_proc = subprocess.run(
                args.fast_cmd,
                shell=True,
                cwd=str(args.repo),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if fast_proc.returncode != 0:
                record = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mode": "loop",
                    "iteration": it,
                    "hypothesis": hypothesis,
                    "git_sha": sha,
                    "diff_stat": stat,
                    "gates": {"fast_prefilter": {"passed": False, "exit_code": fast_proc.returncode}},
                    "decision": "revert",
                }
                append_experiment_log(args.experiments_log, record)
                revert_repo(args.repo, args.revert_cmd, pre_sha, pre_untracked)
                continue

        static_ok, static_errors = (True, []) if cfg.skip_checks else run_static_checks(check_cmds, args.repo)
        if not static_ok:
            cand_med = base_med
            accepted, gates = evaluate_gates(
                base_med, cand_med, scenes, cfg, False, static_errors, cfg.repeats
            )
        else:
            with tempfile.TemporaryDirectory(prefix="klbf-cand-") as tmp:
                tmp_dir = Path(tmp)
                cand_med = {
                    s.name: aggregate_runs([run_scene(s, cand_env, cfg.timeout_s, tmp_dir) for _ in range(cfg.repeats)])
                    for s in scenes
                }
            accepted, gates = evaluate_gates(
                base_med, cand_med, scenes, cfg, True, [], cfg.repeats
            )

        decision = "keep" if accepted else "revert"
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "loop",
            "iteration": it,
            "hypothesis": hypothesis,
            "git_sha": sha,
            "diff_stat": stat,
            "baseline": metrics_summary_dict(base_med),
            "candidate": metrics_summary_dict(cand_med),
            "gates": gates,
            "decision": decision,
        }
        append_experiment_log(args.experiments_log, record)
        if accepted:
            base_med = cand_med
            kept_any = True
        else:
            revert_repo(args.repo, args.revert_cmd, pre_sha, pre_untracked)

    return 0 if kept_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
