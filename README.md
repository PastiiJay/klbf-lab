<p align="center"><img src="docs/klbf/logo.svg" width="120" alt="KLBF Lab"></p>
<h1 align="center">KLBF Lab 39 — test branch (`beta`)</h1>
<p align="center">
  <img src="https://img.shields.io/badge/test-Lab%2039-ff7a00" alt="test: Lab 39">
  <img src="https://img.shields.io/badge/tested%20on-RX%209070%20XT-555" alt="tested on RX 9070 XT">
  <img src="https://img.shields.io/badge/stage-experimental-ffb000" alt="experimental">
  </p>

<p align="center"><b>Test builds of KLBF Lab. The stable version is the <a href="https://github.com/PastiiJay/klbf-lab">main branch</a>.</b></p>

**New to this? Follow the [beginner's guide](docs/GUIDE.md)** (copy-paste commands, page file, troubleshooting).

To build it: `git clone -b beta https://github.com/PastiiJay/klbf-lab`, then the same steps as the main README (pipeline pre-build with `translate --cache --pipelines`, then `run`).

This branch is for **AMD Radeon (RDNA, 64-lane) cards**. It builds on the latest [fxceless/victory-fork](https://github.com/fxceless/victory-fork) (25 Nov 2026, including triippywhale's AMD view-heap clamp and the skin/hair texture re-check; all rendering work there is its author's) and adds fixes found on an RX 9070 XT, plus fixes from other contributors credited below. It is a test build: expect visual defects. The stable branch is [`main`](https://github.com/PastiiJay/klbf-lab/tree/main). Previous test builds are kept as the tags `klbf-lab-33`, `klbf-lab-31`, `klbf-lab-28` and `klbf-lab-26`.

No game files are included. You need your own legally obtained copy.

> [!TIP]
> **Playing on Linux / Proton?** Our Linux work is led by community member **[@zocram4cc](https://github.com/zocram4cc)**: see his Wine/Proton fixes ([#34](https://github.com/PastiiJay/klbf-lab/pull/34), [#35](https://github.com/PastiiJay/klbf-lab/pull/35)), the [Linux notes in the guide](https://github.com/PastiiJay/klbf-lab/blob/main/docs/GUIDE.md#linux--proton-community-notes) and the RX 7800 XT report by [@Adiker](https://github.com/Adiker) ([#19](https://github.com/PastiiJay/klbf-lab/issues/19)). Post Linux results in an issue and tag @zocram4cc.

## New in Lab 39 (25 Nov 2026)

| | What changed | Result on our RX 9070 XT |
|---|---|---|
| **Water physics** | the ocean height map the GPU writes each frame is copied back into the game's memory, where its physics reads it (ported from Geckomacabre/vx-layer); the XOR swizzle modes it is stored in are now addressed (ported from MiamiGame/XTP, tables from Mesa's addrlib) | in the second character's first mission the boat **floats and moves**: with only the throttle held it sailed over a minute (wake, objectives updating) before hitting the bank; before, it sat on the seabed |
| **Menus and HUD in Series S mode** | the title's second present plane (interface) is read at the right place (Lab 38) | start screen, menus, map, HUD, subtitles and button prompts show; 41 min of play without a crash |
| **AMD FSR frame generation** (optional) | `VICTORY_FRAMEGEN=1` presents through AMD's frame generation swap chain (FidelityFX SDK 2.4, `amd_fidelityfx_framegeneration_dx12.dll` in `VICTORY_FFX_FG_DIR` or `VICTORY_FFX_DIR`); off by default; the interface is not interpolated | ~8.5 real fps → ~17 fps shown in the world, no loss of real fps, 12 min without a crash. Generated frames, not real ones: input latency does not improve. Should work on any DX12 card; tested on AMD only |
| **Pre-build on a fresh cache** | the `pipelines` folder is created first, so a crashed pre-build resumes ([#6](https://github.com/PastiiJay/klbf-lab-beta/issues/6), found by clems206) | |

> **Stability policy:** we never publish a build that crashed in our tests. Every session length shown here is when **we** stopped, to get back to work on the project or because a test run reached its planned end, not a crash.

Speed in Lab 39 runs: median 6.6–8.4 fps in the world (720p upscaled to 1080p with AMD FSR). Stability record: 77 min of play without a crash on Lab 39 (stopped by choice); it has never crashed in our test runs.

If the GPU is lost during play, repeat the same scene with `VICTORY_LATE_WAIT=0` and keep the other settings fixed. On one RX 9070 XT, `1` lost the device after about 10 minutes. With `0`, the same tester ran for 32 minutes before a separate title-side crash. An RX 9070 also reported device loss with the example settings. [RX 9070 XT comparison](https://github.com/PastiiJay/klbf-lab/issues/31#issuecomment-7176192100) · [RX 9070 report](https://github.com/PastiiJay/klbf-lab-beta/issues/6)

## Built by KLBF Lab

What this project wrote itself, checked commit by commit (everything taken from other forks keeps its credit in [Credits](#credits)):

- **A pipeline pre-build that survives AMD's shader compiler.** On top of the upstream pre-build, KLBF Lab saves progress as it goes (every 200 pipelines and every minute) and resumes after a crash. Pipelines that crash the driver (`amdxc64.dll`) are recorded in `prebuild-crashed.txt`, and every later run skips them. This is what lets the game start on Radeon cards that crash elsewhere.
- **Console wave32 programs on 64-lane AMD hardware.** Each half of a 64-lane wave runs as one console wave, with wave reductions done per half and the console's float-denormal handling. This is the base of correct pixel and vertex shading on RDNA.
- **NaN guard on colour exports and typed stores**, against the purple/black areas one invalid value spreads on AMD. Proposed upstream too: fxceless/victory-fork#837.
- **The console's plane composition.** Each present plane is composed with its own resources, in submission order, scaled the way the console scales it.
- **AMD frame generation (`VICTORY_FRAMEGEN=1`)** through FidelityFX SDK 2.4, with the interface plane handed to the SDK so menus and HUD are not interpolated. Also a no-SDK fallback for the title's upscale (`VICTORY_FFX_SIMPLE=1`).
- **A per-frame budget for texture uploads** (`VICTORY_TEXTURE_QUEUE_MB/MS`), so streaming doesn't stall a frame.
- **Proven in play, on our own saves.** From a new game to the open world: the prologue and the first mission completed, including the water mission with boats floating. The water fix itself is credited to Geckomacabre/vx-layer and MiamiGame/XTP; we integrated and proved it. Record: 77 min of play without a crash.
- **A beginner's guide** ([docs/GUIDE.md](https://github.com/PastiiJay/klbf-lab/blob/main/docs/GUIDE.md)) with every command to copy and paste, Windows 10 and Linux notes, and a troubleshooting table built from players' reports.

## What it fixes on AMD

| Problem on AMD | Cause we found | In this branch |
|---|---|---|
| Memory grows ~240 MB/s until Windows runs out | compute shader `39f0e9aeb156a61c` never leaves the AMD driver's compiler | skipped |
| Purple/magenta blocks stuck on screen | console 32-lane pixel shaders run in 64-lane waves on AMD; NaN piled up in temporal buffers | 32-lane programs run per half wave; NaN/Inf filtered on colour outputs |
| Missing or late textures | texture upload queue fixed at 192 MB / 250 ms per frame | configurable, larger by default (`VICTORY_TEXTURE_QUEUE_MB`, `VICTORY_TEXTURE_QUEUE_MS`) |
| Characters missing or broken, faces absent | two meshlet-cull compute shaders hung on AMD and had to be skipped | upstream VIctory #334 (buffer-state tracking) ported; `fc5e66f8f873d3bb` no longer skipped: the first character's face, hair and outfit render |
| Compute results not seen by other workgroups | `glc` loads not globally coherent on RDNA | upstream VIctory #1271 ported |
| Lost or misplaced asset loads | DirectStorage page lists read after the title changed them | read at the title's submit (analysis by JappertjeV, [#3](https://github.com/PastiiJay/klbf-lab-beta/issues/3)); `VICTORY_PAGE_LIST=enqueue|submit|late` to compare |
| Cell walls, window frame and posters missing (sky where the wall should be) | pipelines skipped while the world loads, textures streamed in part taken for others | ported from [myokex/victory-amd](https://github.com/myokex/victory-amd): pipelines kept across runs and waited for while a world loads (`VICTORY_PIPELINE_GATE_S`), textures shown from their first loaded mip, lost placed views refreshed (`VICTORY_TEXTURE_SWEEP`), dispatches over 65,535 groups split |
| Walls, window frame and outfit wrong for a whole run, at random | during loading the list of remapped texture ranges overflowed and was thrown away, so which textures got re-checked depended on timing | the list is 32× larger and an overflow triggers a full re-check instead of a drop (found by comparing logs of runs with and without walls) |
| The AMD shader compiler crashes (`amdxc64.dll`) on a few pipelines | driver bug on two pipelines | `victory translate --cache --pipelines` pre-builds every pipeline offline, saves as it goes, finds the crashing ones and skips them in game too |

## What you can expect (RX 9070 XT, Adrenalin 32.0.32029.2010, Series X mode)

| | Result |
|---|---|
| First cell | walls, window frame, posters and photos drawn; the first character's face, hair and orange outfit render; inmates in the yard |
| Speed, Series S mode + AMD FSR (recommended) | **~7.3 fps (median), 74 min with no device removal**, walls drawn throughout |
| Speed, Series X mode | ~4.8–5.2 fps (≈200 ms/frame); 41 min with no device removal (Lab 33) |
| Skipped shaders | only one left (`39f0e9aeb156a61c`, which never leaves the AMD compiler); the other 47 we used to skip now run on this card |
| Cell walls | drawn in 5 of 6 runs (about half before Lab 35). When they are missing, the window frame turns purple and the outfit dark for the whole run; still under investigation |
| Still wrong | a purple tint on some papers and photos, stretched "web" triangles around some characters that change every frame, some textiles look wet, colour banding in skies |

These are a few runs on one machine: other cards and drivers may behave differently.

## Run

Build like VIctory (`cargo build --release`), then pre-build the pipelines once (about 10 minutes, rerun it until it ends without a crash):

```powershell
$env:VICTORY_SHADER_OPT = '0'
$env:VICTORY_SKIP_PIPELINES = (Get-Content docs\klbf\skip-amd.txt -Raw).Trim()
target\release\victory.exe translate --cache --pipelines --jobs 8 <game dir>\shaders.rpf
```

Then play with:

```powershell
$env:VICTORY_SHADER_OPT = '0'
$env:VICTORY_SKIP_PIPELINES = (Get-Content docs\klbf\skip-amd.txt -Raw).Trim()
$env:VICTORY_LATE_WAIT = '1'   # If the GPU is lost during play, retry this line with '0'.
$env:VICTORY_WIDE_BACK = '1'
$env:VICTORY_PIPELINE_WAIT_MS = '24'
$env:VICTORY_RELAY_COPY_QUEUE = '0'
$env:VICTORY_FFX_DIR = '<FidelityFX SDK>\Kits\FidelityFX\signedbin'   # AMD FSR, needed for --console series-s
target\release\victory.exe run --console series-s <game dir>\<the title's exe> --execute
```

`VICTORY_SHADER_OPT=0` turns off the fork's shader clean-up passes, which crashed the AMD compiler for us. `VICTORY_TEXTURE_BUDGET_MB=6144` fixed some dark textiles for another tester but hid the cell walls in both our runs with it, so we leave it off for now. **Faster: Series S mode with AMD FSR.** Add `--console series-s` after `run`, and point `VICTORY_FFX_DIR` at the `signedbin` folder of AMD's official [FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) (`$env:VICTORY_FFX_DIR = '<sdk>\Kits\FidelityFX\signedbin'`). The game renders at 1280×720 and FSR upscales it to 1920×1080: ~7.3 fps over 74 min here. Without FSR the Series S picture stays black. With FSR configured, Lab 38 and later show the start screen, menus and HUD in Series S mode. Plug the controller in before starting: plugging it in during play pauses the title.

### Black screen under Wine

If the log reports `X3506: unrecognized compiler target 'vs_5_1'`, the shader compiler failed to build the pass that composites the game image and HUD. The 2013 `d3dcompiler_47.dll` (version `6.3.9600.16384`) rejects Shader Model 5.1 and can leave the window black even when FSR initializes.

Place the x64 `d3dcompiler_47.dll` from `Redist\D3D\x64` in a recent Windows SDK beside `victory.exe`. Add `d3dcompiler_47=n,b` to Wine's DLL overrides. Version `10.0.28000.2705` restored the visible startup sequence on an RX 6800 with Wine and vkd3d-proton. [Windows SDK downloads](https://learn.microsoft.com/en-us/windows/apps/windows-sdk/downloads)

### Seconds per frame under Wine

If the world takes seconds a frame under Wine/Proton while the GPU is mostly idle and `wineserver` uses most of a core, the cause is in Wine, not in this branch: wineserver keeps each process's mapped views in a list and walks it on every map and unmap ([Wine bug 60868](https://bugs.winehq.org/show_bug.cgi?id=60868)). The title remaps its memory pages by the thousand each frame, so every call gets slower as views pile up. `perf` on `wineserver` shows almost all of its time in `req_unmap_view` and `is_valid_view_addr`, and the log's texture and copy times grow with the remaps.

A `wineserver` that keeps the views in a tree fixes it: [CachyOS/wine-cachyos#50](https://github.com/CachyOS/wine-cachyos/pull/50) (by ajsem1337, based on neonbyte1's version; also measured in [fxceless/victory-fork#184](https://github.com/fxceless/victory-fork/issues/184)). It changes only `server/`, so it can be built for the Wine of your Proton and dropped into a copy of that Proton as `files/bin/wineserver`; the server protocol version must match the Proton build (regenerate `server_protocol.h` with `tools/make_requests` before building). On an RX 6900 XT with Proton Experimental 11.0-20261112, same build and settings, the first minutes in the world went from 2,068 to 509 ms a frame (textures 482 → 73 ms, copies 417 → 99 ms) and `wineserver` from ~70 % to ~6 % of a core.

Please report results in an [issue](https://github.com/PastiiJay/klbf-lab/issues): GPU, driver, how long it ran, fps in the window title, and what looks wrong. For a scene-by-scene report, run `py -3 docs/klbf/summarize_log.py run.log --lines FIRST LAST` with the first and last line numbers for one scene. The [log summary tool](docs/klbf/summarize_log.py) reports median timing samples and separates real and presented FPS when frame generation is logged. Patch notes: [CHANGELOG.md](CHANGELOG.md). Full run log: [docs/klbf/results.md](docs/klbf/results.md).

Product names are only used to say what this works with; they belong to their owners. KLBF Lab is not affiliated with them. License: GPL-3.0.

## Credits

**Code and fixes in our builds come from these people and projects, thank you:**
- [@Marvisak](https://github.com/Marvisak) for [VIctory](https://github.com/Marvisak/VIctory), and [@fxceless](https://github.com/fxceless) (hyphennn) for [victory-fork](https://github.com/fxceless/victory-fork), the base of KLBF Lab.
- [@myokex](https://github.com/myokex) ([victory-amd](https://github.com/myokex/victory-amd)): missing walls on AMD.
- [@Geckomacabre](https://github.com/Geckomacabre) (vx-layer) and [@MiamiGame](https://github.com/MiamiGame) ([XTP](https://github.com/MiamiGame/XTP)): the water physics fix (test builds).
- [@BrokeAsh](https://github.com/BrokeAsh): the RDNA 4 shader compiler workaround; [@gmankab](https://github.com/gmankab) (with AugustoZFJ's original fix): Wine pipeline caching and docs; [@HandshakeHackin](https://github.com/HandshakeHackin): test tools and fixes.
- Upstream VIctory pull requests #334, #1271, #1599 and #1696, and the DirectStorage analysis by JappertjeV.

**Testers and reporters:** [@clems206](https://github.com/clems206), [@TG0000](https://github.com/TG0000), [@BlackBu7let](https://github.com/BlackBu7let), [@kzrts](https://github.com/kzrts), and [@HishamMuhammed](https://github.com/HishamMuhammed), whose questions shaped the [beginner's guide](docs/GUIDE.md).

