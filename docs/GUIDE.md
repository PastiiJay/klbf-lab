# KLBF Lab: beginner's guide (Windows)

Every step below is a command to copy and paste, in order. You only need your own copy of the game's Xbox Series files in a folder. KLBF Lab includes no game files.

In this guide the game files are in **`C:\Games\Title`** (the folder that holds `game_scarlett_master.exe` and `shaders.rpf`). If yours are elsewhere, replace that path everywhere.

Open **PowerShell** (Start menu → type `PowerShell` → Enter) and paste each block. Wait for one block to finish before the next.

---

## 1. Check your PC (1 minute)

- A recent graphics card with DirectX 12 (tested on AMD Radeon RX 9070 XT; other AMD and NVIDIA cards are being tested).
- **More than 8 GB of video memory is recommended.** Some 8 GB setups have [crashed during startup](https://github.com/PastiiJay/klbf-lab/issues/11). One RTX 2080 8 GB reached the 3D world and ran at about 2–3 FPS in gameplay during a 22-minute diagnostic session. [RTX 2080 report](https://github.com/PastiiJay/klbf-lab/issues/65)
- **32 GB of RAM** is recommended. On a 16 GB PC, check the page file in step 2.
- About 15 GB of free disk space for the tools and the shader cache.

## 2. Check the Windows commit limit

If a run reports a memory allocation failure, open **Task Manager → Performance → Memory** and compare the two **Committed** values. On one 32 GB PC running beta-2, the shipped pools reached the roughly 42 GB commit limit. Smaller pools brought the peak down to 36–38 GB, and the Radeon device removal during world load still occurred. Record both the peak and the error when reporting a similar run. [RX 6950 XT report and pool sizes](https://github.com/PastiiJay/klbf-lab/issues/82)

To inspect your page file settings on a 16 GB PC or when committed memory approaches the limit, run

```powershell
SystemPropertiesPerformance.exe
```

Then: **Advanced** tab → **Virtual memory: Change…** → tick **Automatically manage paging file size for all drives** (or untick it, pick your fastest drive, **Custom size**, Initial `32768`, Maximum `49152`) → **Set** → **OK** → restart the PC.

## 3. Install the tools (10–20 minutes, once)

Installs Git, Rust and the Microsoft C++ build tools, all from their official sources through `winget` (built into Windows 10/11):

```powershell
winget install --id Git.Git -e --source winget
winget install --id Rustlang.Rustup -e --source winget
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

**Close PowerShell and open a new one** so the new tools are found.

## 4. Download and build KLBF Lab (5–15 minutes)

```powershell
cd $HOME
git clone -b beta https://github.com/PastiiJay/klbf-lab
cd klbf-lab
cargo build --release
```

The commands above already select the beta branch. If you previously cloned the stable build, run these commands from that checkout to switch and build the latest beta:

```powershell
git fetch origin
git switch beta
git pull --ff-only
cargo build --release
```

## 5. Get AMD's FidelityFX SDK (for the faster Series S mode)

Download **FidelityFX SDK v2.4.0** from AMD's official page: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases. Unzip it to **`C:\FidelityFX-SDK`**. Check that this file exists: `C:\FidelityFX-SDK\Kits\FidelityFX\signedbin\amd_fidelityfx_loader_dx12.dll`.

## 6. Settings (paste at the start of every PowerShell session)

```powershell
cd $HOME\klbf-lab
$env:VICTORY_SHADER_OPT = '0'
$env:VICTORY_SKIP_PIPELINES = (Get-Content docs\klbf\skip-amd.txt -Raw).Trim()
$env:VICTORY_LATE_WAIT = '1'
$env:VICTORY_WIDE_BACK = '1'
$env:VICTORY_PIPELINE_WAIT_MS = '24'
$env:VICTORY_RELAY_COPY_QUEUE = '0'
$env:VICTORY_FFX_DIR = 'C:\FidelityFX-SDK\Kits\FidelityFX\signedbin'
```

If your graphics card stops responding about a minute after the game world starts loading, add `$env:VICTORY_LATE_WAIT = '0'`.

## 7. Pre-build the shaders (10–20 minutes, once)

This compiles all of the game's shaders for your graphics card ahead of time, so the game doesn't stutter or crash on them later:

```powershell
target\release\victory.exe translate --cache --pipelines --jobs 8 C:\Games\Title\shaders.rpf
```

If it crashes (the graphics driver sometimes does), **run the same line again**: it resumes where it stopped and remembers what crashed. It's done when it ends without a crash. Do it again after each KLBF Lab update.

## 8. Play

```powershell
target\release\victory.exe run --console series-s C:\Games\Title\game_scarlett_master.exe --execute
```

- **Plug in the controller before starting.**
- On the stable build the start screen and menus can stay **black**. Press **A** on the controller to advance. Enter also works on the start screen. The [account picker only responds to the controller](https://github.com/PastiiJay/klbf-lab/issues/69).
- The first minutes in a new area are slow while shaders finish loading.
- Saves work (autosave and manual).

## Updating later

```powershell
cd $HOME\klbf-lab
git pull
cargo build --release
```

Then step 6, step 7 again, and step 8.

## Something went wrong?

| What you see | What to do |
|---|---|
| `amd_fidelityfx_loader_dx12.dll not found`, then a crash | Step 5, and check the path in `VICTORY_FFX_DIR` |
| Crash about 3 seconds after start, black window | Check step 2 on a 16 GB system. This symptom has several possible causes. If it persists on beta, include the build and driver details below |
| `X3506` or `vs_5_1` in the log, black window | Old Windows shader compiler: copy the x64 `d3dcompiler_47.dll` from a recent Windows SDK (`Redist\D3D\x64`) next to `victory.exe` |
| `error: unexpected argument '--cache'` | You built an old copy: redo step 4 |
| Graphics card stops responding after ~1 min | `$env:VICTORY_LATE_WAIT = '0'` |
| Crash while loading: the log shows `dstorage: request failed: zlib … hr=0x8007000d`, then `0x80000003` at `victory-image…+0x19b5024` | Some game files are incomplete or damaged: copy the game folder again from your source (thanks @Emaxoso) |

Still stuck? From the same checkout where you built the executable, run:

```powershell
git branch --show-current
git rev-parse --short HEAD
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, BuildNumber
```

Include the output in your [issue](https://github.com/PastiiJay/klbf-lab/issues), along with the last lines PowerShell printed. If you downloaded a binary, give its release or build label instead of the Git details. For a black window at startup, mention whether any logo or menu appeared and whether the result changed after trying beta.

---

Thanks to [@HishamMuhammed](https://github.com/HishamMuhammed), whose questions shaped this guide.
