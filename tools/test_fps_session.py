"""Check the reported median and the shareable 12+ fps figure on synthetic logs."""
from datetime import datetime, timedelta, timezone
import codecs
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SessionReportTests(unittest.TestCase):
    def test_reported_statistics(self):
        cases = [
            ([10, 20], 15, 50),
            ([2.6, 2.7], 2.65, 0),
            ([12, 15, 18], 15, 100),
            ([11, 12, 15, 18, 11, 11], 11.5, 50),
            ([9], 9, 0),
        ]
        script = Path(__file__).with_name("fps-session.py")
        for rates, median, percent in cases:
            with self.subTest(rates=rates), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "session.log"
                report = Path(directory) / "report.json"
                start = datetime(2026, 11, 26, tzinfo=timezone.utc)
                lines = []
                for second, fps in enumerate(rates):
                    timestamp = (start + timedelta(seconds=second)).isoformat()
                    lines.append(f"{timestamp} INFO {fps} fps, {1000 / fps:.3f} ms/frame, 1000 draws/frame\n")
                log.write_text("".join(lines), encoding="utf8")
                output = subprocess.check_output(
                    [sys.executable, str(script), str(log), "--out", str(report)], text=True
                )
                self.assertEqual(json.loads(report.read_text(encoding="utf8"))["median"], median)
                self.assertIn(f"time at 12+ fps: {percent} %", output)

    def test_wrapped_records_keep_their_timestamp_and_original_file_hash(self):
        lines = [
            "\x1b[2m2026-11-26T10:00:00Z\x1b[0m INFO victory_gpu::host:",
            "12.0 fps, 83.3 ms/frame (1.0",
            "preparing), 1000 draws/frame, dropped 0",
            "2026-11-26T10:00:05Z INFO victory_gpu::host: 30.0 fps, 33.3 ms/frame",
            "2026-11-26T10:00:06Z INFO other: 2000 draws/frame",
            "2026-11-26T10:00:09Z INFO victory_gpu::host: 9.0 fps, 111.1 ms/frame, 999 draws/frame",
            "2026-11-26T10:01:00Z INFO victory_gpu::host: 15.0 fps, 66.7 ms/frame, 1100 draws/frame",
        ]
        text = "\r\n".join(lines)
        for encoding, bom in [("utf-8", b""), ("utf-8", codecs.BOM_UTF8),
                              ("utf-16-le", codecs.BOM_UTF16_LE), ("utf-16-be", codecs.BOM_UTF16_BE)]:
            with self.subTest(encoding=encoding, bom=bom), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "console.txt"
                report = Path(directory) / "report.json"
                raw = bom + text.encode(encoding)
                log.write_bytes(raw)
                subprocess.run([sys.executable, str(Path(__file__).with_name("fps-session.py")),
                                str(log), "--out", str(report)], check=True, capture_output=True)
                data = json.loads(report.read_text(encoding="utf8"))
                self.assertEqual((data["samples"], data["median"], data["minutes"]), (2, 13.5, 1.0))
                self.assertEqual(data["log_sha256_16"], hashlib.sha256(raw).hexdigest()[:16])

    def test_bench_window_uses_fractional_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "session.log"
            report = Path(directory) / "report.json"
            # The second sample is 299.2 seconds after the first. The third
            # falls exactly on the exclusive 300-second boundary.
            stamps = ["10:00:00.900000", "10:05:00.100000", "10:05:00.900000"]
            log.write_text("".join(
                f"2026-11-26T{stamp}Z INFO 12 fps, 83.333 ms/frame, 1000 draws/frame\n"
                for stamp in stamps
            ), encoding="utf8")
            subprocess.run([sys.executable, str(Path(__file__).with_name("fps-session.py")),
                            str(log), "--bench", "bm-v1", "--out", str(report)],
                           check=True, capture_output=True)
            data = json.loads(report.read_text(encoding="utf8"))
            self.assertEqual(data["samples"], 2)

    def test_bench_fallback_cuts_300_seconds_and_is_not_certified(self):
        start = datetime(2026, 11, 26, tzinfo=timezone.utc)
        for seconds, samples, complete in [(400, 300, True), (100, 100, False)]:
            with self.subTest(seconds=seconds), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "session.log"
                report = Path(directory) / "report.json"
                # A menu sample first (not world), then one world sample a second.
                lines = [f"{start.isoformat()} INFO 60 fps, 16.667 ms/frame, 10 draws/frame\n"]
                for second in range(seconds):
                    timestamp = (start + timedelta(seconds=10 + second)).isoformat()
                    lines.append(f"{timestamp} INFO 12 fps, 83.333 ms/frame, 1000 draws/frame\n")
                log.write_text("".join(lines), encoding="utf8")
                output = subprocess.check_output([sys.executable, str(Path(__file__).with_name("fps-session.py")),
                                                  str(log), "--out", str(report), "--bench", "bm-v1"], text=True)
                data = json.loads(report.read_text(encoding="utf8"))
                self.assertEqual((data["samples"], data["bench"]["complete"]), (samples, complete))
                self.assertIn("NOT CERTIFIED (no input lock, no card)", output.splitlines()[-3])
                self.assertEqual("PARTIAL" in output, not complete)


if __name__ == "__main__":
    unittest.main()
