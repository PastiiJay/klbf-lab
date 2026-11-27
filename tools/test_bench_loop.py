"""Unit tests for `tools/bench_loop.py` covering all 6 anti-reward-hacking gates."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_loop


def write_test_png(path: Path, width: int, height: int, rgb_pixel: tuple[int, int, int]) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    row = bytes([0]) + bytes(rgb_pixel) * width
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.write_bytes(png)


MOCK_RUNNER = """
import json, os, pathlib, struct, sys, zlib
state_file = pathlib.Path(sys.argv[1])
data = json.loads(state_file.read_text(encoding="utf-8"))
role = "cand" if os.environ.get("MOCK_ROLE") == "cand" else "base"
payload = data[role]
out_path = pathlib.Path(os.environ["VICTORY_BENCH_OUT"])
if payload.get("_corrupt_json"):
    out_path.write_text("{truncated", encoding="utf-8")
else:
    out_path.write_text(json.dumps(payload), encoding="utf-8")
cap_dir = os.environ.get("VICTORY_CAPTURE")
png_spec = payload.get("_png")
if cap_dir and png_spec:
    p = pathlib.Path(cap_dir) / png_spec["name"]
    w, h = int(png_spec["w"]), int(png_spec["h"])
    rgb = tuple(png_spec["rgb"])
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    raw = (bytes([0]) + bytes(rgb) * w) * h
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    p.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
exit_codes = payload.get("_exit_codes")
if exit_codes:
    idx_file = state_file.parent / f"idx_{role}.txt"
    n = int(idx_file.read_text()) if idx_file.exists() else 0
    idx_file.write_text(str(n + 1))
    sys.exit(int(exit_codes[n % len(exit_codes)]))
sys.exit(int(payload.get("_exit_code", 0)))
"""

MOCK_AGENT = """
import json, os, pathlib, sys
state_file = pathlib.Path(sys.argv[1])
it = int(os.environ["BENCH_ITERATION"])
scenarios = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
cur = json.loads(state_file.read_text(encoding="utf-8"))
cur["cand"] = scenarios[it - 1]["payload"]
state_file.write_text(json.dumps(cur), encoding="utf-8")
print(scenarios[it - 1]["hypothesis"])
"""


def make_payload(
    median_ms: float = 20.0,
    low1_fps: float = 40.0,
    draws_avg: float = 4000.0,
    vram_mb: float = 7000.0,
    ram_mb: float = 12000.0,
    zero_draws: int = 0,
    black_frames: int = 0,
    device_removed: bool = False,
    warmup_frames: int = 30,
    input_sha: str | None = "deadbeef",
) -> dict:
    return {
        "world_frames": 120,
        "warmup_frames": warmup_frames,
        "frame_ms": {
            "avg": median_ms,
            "median": median_ms,
            "p95": median_ms * 1.2,
            "p99": median_ms * 1.3,
            "low_5pct_fps": 1000.0 / (median_ms * 1.15),
            "low_1pct_fps": low1_fps,
        },
        "draws": {"avg": draws_avg, "min": int(draws_avg * 0.9), "max": int(draws_avg * 1.1)},
        "zero_draw_world_frames": zero_draws,
        "black_frames": black_frames,
        "device_removed": device_removed,
        "vram_used_peak_mb": vram_mb,
        "vram_budget_mb": 12000.0,
        "ram_private_commit_peak_mb": ram_mb,
        "ram_working_set_peak_mb": ram_mb * 0.85,
        "input_script_sha256": input_sha,
    }


class BenchLoopTests(unittest.TestCase):
    def run_ab_case(
        self,
        tmp: Path,
        base_payload: dict,
        cand_payload: dict,
        extra_args: list[str] | None = None,
        scene_overrides: dict | None = None,
    ):
        runner = tmp / "runner.py"
        runner.write_text(MOCK_RUNNER, encoding="utf-8")
        state = tmp / "state.json"
        state.write_text(json.dumps({"base": base_payload, "cand": cand_payload}), encoding="utf-8")
        scene = {
            "name": "city_drive",
            "cmd": [sys.executable, str(runner), str(state)],
            "warmup": 30,
            "frames": 120,
        }
        if scene_overrides:
            scene.update(scene_overrides)
        playlist = tmp / "playlist.json"
        playlist.write_text(json.dumps({"hypothesis": "test case", "scenes": [scene]}), encoding="utf-8")
        exp_log = tmp / "experiments.jsonl"
        argv = [
            "--mode",
            "ab",
            "--config",
            str(playlist),
            "--repo",
            str(tmp),
            "--repeats",
            "3",
            "--skip-checks",
            "--base-env",
            "MOCK_ROLE=base",
            "--cand-env",
            "MOCK_ROLE=cand",
            "--experiments-log",
            str(exp_log),
        ]
        if extra_args:
            argv.extend(extra_args)
        rc = bench_loop.main(argv)
        entry = json.loads(exp_log.read_text(encoding="utf-8").strip().splitlines()[-1])
        return rc, entry

    def test_keep_on_real_win(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc, entry = self.run_ab_case(
                Path(d),
                make_payload(median_ms=20.0, low1_fps=40.0),
                make_payload(median_ms=17.0, low1_fps=46.0),
            )
            self.assertEqual(rc, 0)
            self.assertEqual(entry["decision"], "keep")
            self.assertTrue(all(g["passed"] for g in entry["gates"].values()))
            self.assertAlmostEqual(entry["gates"]["gate6_composite_score"]["gain_pct"], 15.0, places=2)

    def test_reject_when_a_repeat_ends_before_the_requested_frame_count(self) -> None:
        spec = bench_loop.SceneSpec(name="drive", cmd=[], warmup=30, frames=120)

        def report(ms: float, frames: int):
            payload = {**make_payload(median_ms=ms), "world_frames": frames}
            return bench_loop.parse_metrics_payload(payload, spec.name, 0, False)

        baseline = [report(20.0, 120) for _ in range(3)]
        candidate = [report(17.0, 120) for _ in range(3)]
        for side in ("baseline", "candidate"):
            with self.subTest(side=side):
                base_runs = baseline.copy()
                cand_runs = candidate.copy()
                if side == "baseline":
                    base_runs[1] = report(20.0, 1)
                else:
                    cand_runs[1] = report(17.0, 1)
                accepted, gates = bench_loop.evaluate_gates(
                    {spec.name: bench_loop.aggregate_runs(base_runs)},
                    {spec.name: bench_loop.aggregate_runs(cand_runs)},
                    [spec],
                    bench_loop.GateConfig(),
                    True,
                    [],
                    3,
                )
                self.assertFalse(accepted)
                gate = gates["gate5_warmup_median"]
                self.assertFalse(gate["passed"])
                self.assertIn(side, " ".join(gate["errors"]))

    def test_reject_on_negative_crash_exit_code_mixed_with_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cand = make_payload(median_ms=15.0)
            cand["_exit_codes"] = [-1073741819, 0, 0]
            rc, entry = self.run_ab_case(Path(d), make_payload(median_ms=20.0), cand)
            self.assertEqual(rc, 1)
            self.assertEqual(entry["decision"], "reject")
            self.assertFalse(entry["gates"]["gate2_stability"]["passed"])

    def test_reject_on_corrupt_json_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cand = make_payload(median_ms=15.0)
            cand["_corrupt_json"] = True
            rc, entry = self.run_ab_case(Path(d), make_payload(median_ms=20.0), cand)
            self.assertEqual(rc, 1)
            self.assertEqual(entry["decision"], "reject")
            self.assertFalse(entry["gates"]["gate2_stability"]["passed"])

    def test_reject_on_draw_skip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc, entry = self.run_ab_case(
                Path(d),
                make_payload(median_ms=20.0, draws_avg=4000.0),
                make_payload(median_ms=12.0, draws_avg=2500.0),
            )
            self.assertEqual(rc, 1)
            self.assertEqual(entry["decision"], "reject")
            self.assertFalse(entry["gates"]["gate3_draw_parity"]["passed"])

    def test_reject_if_only_candidate_reports_the_configured_replay(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc, entry = self.run_ab_case(
                Path(d),
                make_payload(median_ms=20.0, input_sha=None),
                make_payload(median_ms=17.0),
                scene_overrides={"input_script": "route.jsonl"},
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(entry["decision"], "reject")
            self.assertFalse(entry["gates"]["gate3_draw_parity"]["passed"])

            rc_both_null, entry_both_null = self.run_ab_case(
                Path(d),
                make_payload(median_ms=20.0, input_sha=None),
                make_payload(median_ms=17.0, input_sha=None),
                scene_overrides={"input_script": "route.jsonl"},
            )
            self.assertNotEqual(rc_both_null, 0)
            self.assertEqual(entry_both_null["decision"], "reject")
            self.assertFalse(entry_both_null["gates"]["gate3_draw_parity"]["passed"])

    def test_reject_on_input_script_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc, entry = self.run_ab_case(
                Path(d),
                make_payload(median_ms=20.0, input_sha="aaa111"),
                make_payload(median_ms=15.0, input_sha=None),
                scene_overrides={"input_script": "dummy.jsonl"},
            )
            self.assertEqual(rc, 1)
            self.assertEqual(entry["decision"], "reject")
            self.assertFalse(entry["gates"]["gate3_draw_parity"]["passed"])

    def test_reject_on_device_removed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc, entry = self.run_ab_case(
                Path(d),
                make_payload(median_ms=20.0),
                make_payload(median_ms=16.0, device_removed=True),
            )
            self.assertEqual(rc, 1)
            self.assertEqual(entry["decision"], "reject")
            self.assertFalse(entry["gates"]["gate2_stability"]["passed"])

    def test_reject_on_black_frame_and_end_to_end_png_gates(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rc, entry = self.run_ab_case(
                tmp,
                make_payload(median_ms=20.0),
                make_payload(median_ms=16.0, black_frames=2),
            )
            self.assertEqual(rc, 1)
            self.assertEqual(entry["decision"], "reject")
            self.assertFalse(entry["gates"]["gate4_visual_sanity"]["passed"])

            base_pngs = tmp / "base_pngs"
            base_pngs.mkdir()
            write_test_png(base_pngs / "frame-30.png", 8, 8, (120, 130, 140))

            # 1) Black PNG written to isolated capture dir fails Gate 4
            cand_black = make_payload(median_ms=16.0)
            cand_black["_png"] = {"name": "frame-30.png", "w": 8, "h": 8, "rgb": [1, 1, 1]}
            base_good = make_payload(median_ms=20.0)
            base_good["_png"] = {"name": "frame-30.png", "w": 8, "h": 8, "rgb": [120, 130, 140]}
            rc_black, entry_black = self.run_ab_case(
                tmp, base_good, cand_black, scene_overrides={"baseline_png_dir": str(base_pngs)}
            )
            self.assertEqual(rc_black, 1)
            self.assertFalse(entry_black["gates"]["gate4_visual_sanity"]["passed"])

            # 2) Missing PNG in candidate when baseline_png_dir expects it fails Gate 4 (no cross-run contamination)
            cand_missing = make_payload(median_ms=16.0)
            rc_missing, entry_missing = self.run_ab_case(
                tmp, base_good, cand_missing, scene_overrides={"baseline_png_dir": str(base_pngs)}
            )
            self.assertEqual(rc_missing, 1)
            self.assertFalse(entry_missing["gates"]["gate4_visual_sanity"]["passed"])

            # 3) Matching PNG passes Gate 4
            cand_good = make_payload(median_ms=16.0)
            cand_good["_png"] = {"name": "frame-30.png", "w": 8, "h": 8, "rgb": [122, 131, 139]}
            rc_ok, entry_ok = self.run_ab_case(
                tmp, base_good, cand_good, scene_overrides={"baseline_png_dir": str(base_pngs)}
            )
            self.assertEqual(rc_ok, 0)
            self.assertTrue(entry_ok["gates"]["gate4_visual_sanity"]["passed"])

    def test_reject_on_vram_cap(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc, entry = self.run_ab_case(
                Path(d),
                make_payload(median_ms=20.0, vram_mb=7000.0),
                make_payload(median_ms=16.0, vram_mb=9500.0),
                extra_args=["--max-vram-mb", "8192"],
            )
            self.assertEqual(rc, 1)
            self.assertEqual(entry["decision"], "reject")
            self.assertFalse(entry["gates"]["gate6_composite_score"]["passed"])

    def test_reject_on_noise_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc, entry = self.run_ab_case(
                Path(d),
                make_payload(median_ms=20.0),
                make_payload(median_ms=19.85),  # 0.75% gain < default 2.0%
            )
            self.assertEqual(rc, 1)
            self.assertEqual(entry["decision"], "reject")
            self.assertFalse(entry["gates"]["gate6_composite_score"]["passed"])

    def test_loop_mode_logs_experiments_and_reverts_bad_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            runner = tmp / "runner.py"
            runner.write_text(MOCK_RUNNER, encoding="utf-8")
            agent = tmp / "agent.py"
            agent.write_text(MOCK_AGENT, encoding="utf-8")
            revert_marker = tmp / "reverted.txt"
            state = tmp / "state.json"
            base_p = make_payload(median_ms=20.0, draws_avg=4000.0)
            state.write_text(json.dumps({"base": base_p, "cand": base_p}), encoding="utf-8")
            scenarios = tmp / "scenarios.json"
            scenarios.write_text(
                json.dumps(
                    [
                        {
                            "hypothesis": "skip half the draws",
                            "payload": make_payload(median_ms=11.0, draws_avg=1800.0),
                        },
                        {
                            "hypothesis": "real optimization",
                            "payload": make_payload(median_ms=17.5, low1_fps=45.0, draws_avg=4000.0),
                        },
                    ]
                ),
                encoding="utf-8",
            )
            playlist = tmp / "playlist.json"
            playlist.write_text(
                json.dumps(
                    {
                        "scenes": [
                            {
                                "name": "city_drive",
                                "cmd": [sys.executable, str(runner), str(state)],
                                "warmup": 30,
                                "frames": 120,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            exp_log = tmp / "experiments.jsonl"
            revert_cmd = f'"{sys.executable}" -c "import pathlib; pathlib.Path(r\'{revert_marker}\').write_text(\'ok\')"'
            agent_cmd = f'"{sys.executable}" "{agent}" "{state}" "{scenarios}"'
            rc = bench_loop.main(
                [
                    "--mode",
                    "loop",
                    "--config",
                    str(playlist),
                    "--repo",
                    str(tmp),
                    "--repeats",
                    "2",
                    "--max-iters",
                    "2",
                    "--skip-checks",
                    "--base-env",
                    "MOCK_ROLE=base",
                    "--cand-env",
                    "MOCK_ROLE=cand",
                    "--agent-cmd",
                    agent_cmd,
                    "--revert-cmd",
                    revert_cmd,
                    "--experiments-log",
                    str(exp_log),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(revert_marker.is_file())
            rows = [json.loads(line) for line in exp_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["decision"], "revert")
            self.assertEqual(rows[0]["hypothesis"], "skip half the draws")
            self.assertFalse(rows[0]["gates"]["gate3_draw_parity"]["passed"])
            self.assertEqual(rows[1]["decision"], "keep")
            self.assertEqual(rows[1]["hypothesis"], "real optimization")
            self.assertTrue(all(g["passed"] for g in rows[1]["gates"].values()))

    def test_git_revert_cleans_staged_committed_and_untracked_changes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            tracked = repo / "tracked.txt"
            tracked.write_text("v1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True)
            pre_existing = repo / "keep_me.txt"
            pre_existing.write_text("keep\n", encoding="utf-8")

            pre_sha, pre_untracked = bench_loop.git_snapshot(repo)
            self.assertIn("keep_me.txt", pre_untracked)

            # Simulate an agent committing changes and adding an untracked file.
            tracked.write_text("v2\n", encoding="utf-8")
            new_untracked = repo / "new_junk.rs"
            new_untracked.write_text("junk\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-m", "bad"], check=True, capture_output=True)

            bench_loop.revert_repo(repo, None, pre_sha, pre_untracked)
            self.assertEqual(tracked.read_text(encoding="utf-8"), "v1\n")
            self.assertFalse(new_untracked.exists())
            self.assertTrue(pre_existing.is_file())

    def test_summarize_log_fallback_parsing(self) -> None:
        text = "Host timing (42 samples, medians): 48.0 FPS, 20.8 ms/frame, 14.2 ms/frame preparing on CPU\n"
        parsed = bench_loop.parse_summarize_text(text, "scene1", 0, False)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.world_frames, 42)
        self.assertAlmostEqual(parsed.median_frame_ms, 20.8)
        self.assertAlmostEqual(parsed.preparing_ms or 0.0, 14.2)


if __name__ == "__main__":
    unittest.main()
