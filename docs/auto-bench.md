# Deterministic World-Frame Benchmarking & Closed-Loop Harness

This guide documents the opt-in deterministic input replay, per-world-frame JSON metrics, and automated A/B + optimization loop harness. See [results.md](klbf/results.md) and [summarize_log.py](klbf/summarize_log.py) for existing manual log measurements.

## Quick start: record one route and replay it twice

Use paths **outside** the repository (e.g. `D:\bench\...`). Replay waits at neutral input while `world_frame() == 0` (menus/loading) and starts at the first frame with `>= 500` draws.

**1. Record one route** (`30` warmup + `180` measured world frames, then clean exit):

```powershell
$env:VICTORY_INPUT_RECORD = 'D:\bench\route.jsonl'
$env:VICTORY_BENCH_WARMUP = '30'
$env:VICTORY_BENCH_EXIT_FRAMES = '180'
.\target\release\victory.exe run --console series-s D:\Title\game.exe --execute
Remove-Item Env:VICTORY_INPUT_RECORD
```

**2. Replay it twice** (baseline vs candidate, saving two JSON reports):

```powershell
$env:VICTORY_INPUT_REPLAY = 'D:\bench\route.jsonl'
$env:VICTORY_BENCH_WARMUP = '30'
$env:VICTORY_BENCH_EXIT_FRAMES = '180'

# Run 1 (baseline)
$env:VICTORY_BENCH_OUT = 'D:\bench\base.json'
D:\builds\base\victory.exe run --console series-s D:\Title\game.exe --execute

# Run 2 (candidate)
$env:VICTORY_BENCH_OUT = 'D:\bench\cand.json'
D:\builds\cand\victory.exe run --console series-s D:\Title\game.exe --execute
```

Or run interleaved A/B repeats automatically with `tools/bench_loop.py --mode ab` (see Section 3 below).

## 1. World-Frame Bench Telemetry (`victory-gpu::bench`)

All benchmark reporting and exit hooks are disabled unless configured:

- `VICTORY_BENCH_OUT=<abs_path.json>`: writes JSON metrics atomically (`<file>.tmp.<pid>.<seq>` + rename). Paths inside any `.victory-root` repository tree are refused.
- `VICTORY_BENCH_WARMUP=<W>`: world frames excluded from statistics (default `30`).
- `VICTORY_BENCH_EXIT_FRAMES=<N>`: cleanly exits (`0`) after flushing `warmup + N` world frames.

`victory_gpu::bench::world_frame()` stays `0` through menus and loading screens, latches to `1` on the first presented frame that executes at least `500` draws, and increments monotonically by `1` on every presented frame thereafter.

The JSON output includes:
- `world_frames`, `warmup_frames`
- `frame_ms`: `avg`, `median`, `p95`, `p99`, `low_5pct_fps`, `low_1pct_fps`
- `draws`: `avg`, `min`, `max`
- `zero_draw_world_frames`, `black_frames`, `device_removed`
- `vram_used_peak_mb`, `vram_budget_mb`, `ram_private_commit_peak_mb`, `ram_working_set_peak_mb`
- `input_script_sha256` (BLAKE3 hex digest of the active replay script)

## 2. Deterministic World-Frame Input Record & Replay (`gameinput_script`)

- `VICTORY_INPUT_RECORD=<abs_path.jsonl>`: logs `PadState` and `KeyState` changes tagged by `world_frame()` (`wframe >= 1`, skipping loading screens).
- `VICTORY_INPUT_REPLAY=<abs_path.jsonl>`: replays either recorded JSONL traces or declarative range lines:

```json
{"from": 0, "to": 120, "ly": 1.0, "rt": 0.8}
{"from": 50, "to": 60, "buttons": 4}
```

Replay waits neutrally while `world_frame() == 0` so variable loading times and FPS differences across builds never drift input timing.

### Delivery contract

- **Loading screens (`world_frame == 0`)**: replay holds neutral input (slot 0 connected, no keys) and neither advances playback state nor consumes pending pulses.
- **Recorded traces (`wframe`)**: step-function holds — the latest recorded entry with `entry.wframe <= world_frame` stays active until the next recorded change.
- **Multi-frame declarative spans (`to > from`)**: active strictly while `from <= world_frame <= to`; spans never latch past `to` if all frames in `[from, to]` are skipped between polls.
- **One-frame declarative pulses (`from == to`, `from >= 1`)**: on the live stateful playback path (`ScriptedSource` / `ReplayPlayback`), if no poll occurs during `world_frame == from`, an unpolled pulse in the zero-poll gap (`last_polled_wframe < from <= world_frame`) latches and delivers once on the first eligible polled world frame (`world_frame >= from`), staying consistent across repeated polls and pad/key reads within that single world frame before releasing permanently on the next world frame. Stateless point queries (`Script::pad_at` / `Script::keys_at`) evaluate exact `from <= world_frame <= to`.

## 3. Automated A/B & Optimization Loop (`tools/bench_loop.py`)

Create a JSON playlist (outside the repository when referencing local capture/game paths):

```json
{
  "hypothesis": "Compare baseline vs candidate",
  "scenes": [
    {
      "name": "traversal_drive",
      "cmd": ["D:/builds/victory.exe", "run", "--console", "series-s", "D:/Title/game.exe", "--execute"],
      "input_script": "D:/bench/drive.jsonl",
      "warmup": 30,
      "frames": 180,
      "timeout_s": 300
    }
  ]
}
```

Run interleaved A/B (`base -> cand -> base` alternating per repeat):

```powershell
python -X utf8 tools/bench_loop.py --mode ab --config D:\bench\playlist.json --repeats 3
```

Run the autonomous optimization loop:

```powershell
python -X utf8 tools/bench_loop.py --mode loop --config D:\bench\playlist.json --agent-cmd "<agent>" --max-iters 5 --experiments-log D:\bench\experiments.jsonl
```

### Anti-Reward-Hacking Gates

Every candidate must pass all 6 gates before it is kept:
1. **Static checks**: `cargo run --locked -p xtask -- policy`, workspace `clippy`, and `cargo test --locked`.
2. **Stability**: exit code `0`, no timeout, `device_removed == false`.
3. **Draw parity**: `avg_draws` within `±15%` of baseline (`--max-draw-delta-pct`).
4. **Visual sanity**: `zero_draw_world_frames == 0`, `black_frames == 0`, plus optional checkpoint PNG luma/MAE parity.
5. **Warmup & median**: warmup frames verified and per-scene medians taken over `--repeats` runs.
6. **Composite score**: `median frame_ms` improvement `>= --min-gain-pct` (default `2.0%`) with non-regression guardrails on `low_1pct_fps`, `vram_used_peak_mb`, and `ram_private_commit_peak_mb`.
