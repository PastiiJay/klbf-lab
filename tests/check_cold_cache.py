"""Check Windows shader pre-build with empty and missing inputs on first use.

From the repository root after building victory:
    py -3 tests/check_cold_cache.py target/release/victory.exe
"""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("victory", type=Path, help="path to the Windows victory executable")
    args = parser.parse_args()
    victory = args.victory.resolve()
    if not victory.is_file():
        parser.error(f"executable does not exist: {victory}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = root / "empty-input"
        inputs.mkdir()
        cache = root / "cache"
        env = os.environ.copy()
        env.pop("VICTORY_CONFIG", None)
        env["VICTORY_CACHE_DIR"] = str(cache)
        env["VICTORY_DATA_DIR"] = str(root / "data")
        command = [str(victory), "translate", "--cache", "--pipelines"]
        subprocess.run(
            [*command, str(inputs)],
            cwd=root, env=env, check=True, timeout=60,
        )
        if not (cache / "pipelines").is_dir():
            raise SystemExit("FAIL: pre-build returned without creating cache/pipelines")
        print("PASS: pre-build created cache/pipelines on a cold start")

        missing = root / "missing shaders.rpf"
        unused_cache = root / "missing-input-cache"
        env["VICTORY_CACHE_DIR"] = str(unused_cache)
        result = subprocess.run(
            [*command, str(inputs), str(missing)], cwd=root, env=env, timeout=60,
            capture_output=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0 or str(missing) not in result.stderr:
            raise SystemExit(f"FAIL: expected an error naming {missing}; got {result.stderr!r}")
        if unused_cache.exists():
            raise SystemExit("FAIL: pre-build created the cache before reporting a missing input")
    print("PASS: pre-build named the missing input before creating its cache")


if __name__ == "__main__":
    main()
