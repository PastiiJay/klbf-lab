# KLBF Lab: the launcher's setup window and shader preparation, dot-sourced by play.ps1.
# Settings: <home>\launcher\settings.json (the current choice) and presets\<name>.json (KLBF_LAUNCHER_DIR
# moves that folder, for tests). <home> = Saved Games\KLBF Lab (Get-KlbfHome): data, cache, launcher,
# save-backups, bench-data.

# (27/11) Hardware we have tested: on it the window marks the suggested options "(recommended for your
# hardware)" (without picking them). Add a line as players' reports come in: gen as Get-KlbfGpuKind names it.
$script:KlbfTested = @(
    @{ gen = 'RDNA 4'; minVramGB = 16 }
)

$script:KlbfModes = '1280x720', '1664x936', '1920x1080', '2560x1440', '3840x2160'
# AMD's presets: play.ps1's -FsrQuality values and their names (no Native AA: at 1.0 the Profile X memory
# plan overflows, victory-loader resolution.rs).
$script:KlbfPresets = [ordered]@{ quality = 'Quality'; balanced = 'Balanced'; performance = 'Performance'; ultraperformance = 'Ultra Performance' }

# (27/11) KLBF Lab's home: the Saved Games folder's "KLBF Lab" (KLBF_HOME overrides, for tests). Never
# AppData: a process started from a packaged (MSIX) app sees its own private AppData, and the player's saves
# split into two folders (27/11).
function Get-KlbfHome {
    if ($env:KLBF_HOME) { return $env:KLBF_HOME }
    $saved = $null
    try {
        if (-not ('KlbfKnown' -as [type])) {
            Add-Type -Namespace '' -Name KlbfKnown -MemberDefinition '[DllImport("shell32.dll", CharSet = CharSet.Unicode)] public static extern int SHGetKnownFolderPath(ref Guid id, int flags, IntPtr token, out string path);'
        }
        $id = [Guid]'4C5C32FF-BB9D-43b0-B5B4-2D72E54EAAA4' # FOLDERID_SavedGames
        $p = ''
        if ([KlbfKnown]::SHGetKnownFolderPath([ref]$id, 0, [IntPtr]::Zero, [ref]$p) -eq 0) { $saved = $p }
    } catch {}
    if (-not $saved) { $saved = Join-Path $env:USERPROFILE 'Saved Games' }
    Join-Path $saved 'KLBF Lab'
}

function Get-KlbfLauncherDir {
    if ($env:KLBF_LAUNCHER_DIR) { return $env:KLBF_LAUNCHER_DIR }
    Join-Path (Get-KlbfHome) 'launcher'
}

# The folders an older launcher used (AppData's klbf-lab, and the copy a packaged app saw): KLBF_OLD_HOMES
# (';'-separated) for tests.
function Get-KlbfOldHomes {
    if ($null -ne $env:KLBF_OLD_HOMES) { return @($env:KLBF_OLD_HOMES -split ';' | Where-Object { $_ }) }
    @(Join-Path $env:LOCALAPPDATA 'klbf-lab') + @(Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA 'Packages\*\LocalCache\Local\klbf-lab') -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
}

# The newest save file's time under a home's data folder, or $null without saves.
function Get-KlbfNewestSave([string]$Dir) {
    $r = Get-KlbfSavesRoot (Join-Path $Dir 'data')
    if (-not $r) { return $null }
    $s = @(Get-KlbfSaves $r)
    if (-not $s) { return $null }
    ($s | Sort-Object date -Descending | Select-Object -First 1).date
}

# (27/11) One-time move to the new home: COPIES (never moves or deletes) from the old folder with the newest
# save: data (saves and game settings), launcher settings and presets, save backups, and the shader cache
# unless -NoCache. Files already in the new home are never overwritten; nothing is copied into a data folder
# that already has saves. Returns what was done (one line), or '' when there was nothing to copy.
function Move-KlbfOldData([string]$To = (Get-KlbfHome), [string[]]$From = (Get-KlbfOldHomes), [switch]$NoCache) {
    $best = $null; $when = $null
    foreach ($h in $From) {
        if (-not (Test-Path -LiteralPath $h)) { continue }
        $n = Get-KlbfNewestSave $h
        if ($n -and (-not $when -or $n -gt $when)) { $best = $h; $when = $n }
    }
    if (-not $best) { return '' }
    $parts = @('launcher', 'save-backups') + @(if (-not $NoCache) { 'cache' })
    if (-not (Get-KlbfNewestSave $To)) { $parts = @('data') + $parts }
    $copied = @()
    foreach ($p in $parts) {
        $src = Join-Path $best $p
        if (-not (Test-Path -LiteralPath $src)) { continue }
        $n = 0
        foreach ($f in (Get-ChildItem -LiteralPath $src -Recurse -File -Force)) {
            $rel = $f.FullName.Substring($src.Length).TrimStart('\')
            if ($p -eq 'data' -and $rel -match '\\saves\\saves(\\|$)') { continue }
            $dst = Join-Path (Join-Path $To $p) $rel
            if (Test-Path -LiteralPath $dst) { continue }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
            Copy-Item -LiteralPath $f.FullName -Destination $dst
            (Get-Item -LiteralPath $dst).LastWriteTime = $f.LastWriteTime
            $n++
        }
        # The game's empty folders too (a save container without files yet)
        foreach ($d in (Get-ChildItem -LiteralPath $src -Recurse -Directory -Force)) {
            $rel = $d.FullName.Substring($src.Length).TrimStart('\')
            if ($p -eq 'data' -and $rel -match '\\saves\\saves(\\|$)') { continue }
            New-Item -ItemType Directory -Force -Path (Join-Path (Join-Path $To $p) $rel) | Out-Null
        }
        if ($n) { $copied += "$p ($n files)" }
    }
    if (-not $copied) { return '' }
    "Copied from the older KLBF Lab folder (saves of $($when.ToString('yyyy-MM-dd HH:mm'))): $($copied -join ', '). The older folder is left as it was."
}

# (27/11) Before a start: the data folder has no saves while an older folder has some. The message to show
# (the game is not started), or ''.
function Test-KlbfLostSaves([string]$Data, [string[]]$From = (Get-KlbfOldHomes)) {
    $r = Get-KlbfSavesRoot $Data
    if ($r -and @(Get-KlbfSaves $r)) { return '' }
    foreach ($h in $From) {
        $n = Get-KlbfNewestSave $h
        if ($n) {
            return "No saves in the KLBF Lab data folder, but an older KLBF Lab folder has saves (newest $($n.ToString('yyyy-MM-dd HH:mm'))). The game was not started, so it does not begin a new game over them. Copy them once with: . .\launcher-menu.ps1; Move-KlbfOldData  (then start again)."
        }
    }
    ''
}

# The owner's setting: Profile X, FSR 4 Performance, 3840x2160, frame generation, hooks and texture trim
# (= -Tout -FsrQuality performance -Output 3840x2160).
function Get-KlbfDefaults {
    [ordered]@{ profile = 'X'; upscaler = 'fsr'; preset = 'performance'; output = '3840x2160'; frameGen = $true; advanced = $true; game = ''; shotsDir = ''; display = -1; borderless = $false }
}

# Any object (a parsed JSON) over the defaults: unknown or invalid values keep the default.
function ConvertTo-KlbfChoice($o) {
    $c = Get-KlbfDefaults
    if (-not $o) { return $c }
    if ($o.profile -in 'X', 'S') { $c.profile = [string]$o.profile }
    if ($o.upscaler -in 'fsr', 'title') { $c.upscaler = [string]$o.upscaler }
    if ($o.preset -in 'quality', 'balanced', 'performance', 'ultraperformance') { $c.preset = [string]$o.preset }
    if ($o.output -in $script:KlbfModes) { $c.output = [string]$o.output }
    if ($o.frameGen -is [bool]) { $c.frameGen = $o.frameGen }
    if ($o.advanced -is [bool]) { $c.advanced = $o.advanced }
    if ($o.game -is [string]) { $c.game = $o.game }
    if ($o.shotsDir -is [string]) { $c.shotsDir = $o.shotsDir }
    if ($o.borderless -is [bool]) { $c.borderless = $o.borderless }
    if (($o.display -is [int] -or $o.display -is [long]) -and $o.display -ge -1 -and $o.display -lt 16) { $c.display = [int]$o.display }
    $c
}

# (27/11) Where F8's screenshots go (KLBF_SHOTS_DIR): the folder chosen in Settings, else Pictures\KLBF Lab
# (made by the game on the first screenshot).
function Get-KlbfShotsDir($Choice) {
    if ($Choice -and $Choice.shotsDir) { return [string]$Choice.shotsDir }
    Join-Path ([Environment]::GetFolderPath('MyPictures')) 'KLBF Lab'
}

# (27/11) What play.ps1 does when the game ends with $Code: 10 (F10, Settings) opens the settings window,
# whose Play starts it again; 11 (F11, Restart) starts it again with the same settings; anything else stops.
# After a benchmark (whatever the exit): 'quick', the settings window on the Quick tab with the new card.
function Get-KlbfNextStart([int]$Code, [bool]$Bench = $false) {
    if ($Bench) { return 'quick' }
    switch ($Code) { 10 { 'settings' } 11 { 'restart' } default { '' } }
}

# F11's start: play.ps1's own parameters again, without the settings window.
function Get-KlbfRestartArgs($Bound) {
    $a = @{}
    foreach ($k in $Bound.Keys) { if ($k -notin 'Setup', 'Quick') { $a[$k] = $Bound[$k] } }
    $a.NoMenu = [switch]$true
    $a
}

# (27/11) The desktop shortcut that starts this build with no console flashing: wscript.exe runs
# "KLBF Lab.vbs" (beside play.ps1, written when the shortcut is made), which starts PowerShell hidden on the
# folder's play.ps1 with -Setup (the settings window first; it shows itself, see Set-KlbfTheme).
function Get-KlbfShortcutSpec([string]$BuildDir = $PSScriptRoot, [string]$Victory = '') {
    $ico = Join-Path $BuildDir 'klbf-lab.ico'
    if (-not (Test-Path -LiteralPath $ico)) { $ico = Join-Path $BuildDir 'apps\victory\klbf-lab.ico' }
    @{
        target = Join-Path $env:SystemRoot 'System32\wscript.exe'
        arguments = '"{0}"' -f (Join-Path $BuildDir 'KLBF Lab.vbs')
        workdir = $BuildDir
        icon = $ico
        vbs = Join-Path $BuildDir 'KLBF Lab.vbs'
    }
}

# The starter script: the folder from its own place, PowerShell hidden (window style 0), not waited for.
function Get-KlbfVbsText {
    @(
        "' KLBF Lab: starts play.ps1 of this folder with no console window (the desktop shortcut runs this)."
        'Dim d: d = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)'
        'CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & d & "\play.ps1"" -Victory """ & d & "\victory.exe"" -Setup", 0, False'
    ) -join "`r`n"
}

# (27/11) The official components the release zip does not carry, fetched once into the build folder:
# 'agility' (Microsoft's DirectX 12 Agility SDK 1.620.2, NuGet, into D3D12\) and 'ffx' (AMD's signed
# FidelityFX DLLs, SDK v2.4.0 / FSR 4.1.2, AMD's GitHub at the tag's commit, into ffx\), each file checked
# against its SHA-256 before it is put in place, the license beside them. $true when all are there.
function Install-KlbfComponent([ValidateSet('agility', 'ffx')][string]$Name, [string]$BuildDir, [switch]$CheckOnly) {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    $c = @{
        agility = @{ dir = 'D3D12'; zip = 'https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.620.2'; license = @('LICENSE.txt', 'LICENSE-Microsoft-Agility-SDK.txt')
            files = [ordered]@{
                'build/native/bin/x64/D3D12Core.dll'      = '70a6d47d335b4a15180544c01a35b84474522a369dc582d85fe057313a3aeaa0'
                'build/native/bin/x64/d3d12SDKLayers.dll' = '3140822f550003d322b0a0681e0c40af7364c561296a1d7eeb96b3258c9cd32d'
            } }
        ffx = @{ dir = 'ffx'; base = 'https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/7539ae8b6568ad29ed78a753404351daf3b5cca7/Kits/FidelityFX'; license = @('docs/license.md', 'LICENSE-AMD-FidelityFX.md')
            files = [ordered]@{
                'signedbin/amd_fidelityfx_loader_dx12.dll'          = 'e2191d3c0828be45eb8affd4e3ec0378f2293adc345b772a7170edcf3b021ecb'
                'signedbin/amd_fidelityfx_upscaler_dx12.dll'        = 'aad4fccccb29084a655f9a4f9f613bd22c8c03fb4a0f8840148cc6543b4ccace'
                'signedbin/amd_fidelityfx_framegeneration_dx12.dll' = 'e6843252f58b0341a0d80f6a20bc0d840643ee95bc48c22cac08745344577bb6'
            } }
    }[$Name]
    $dst = Join-Path $BuildDir $c.dir
    # (27/11) Checked by hash, not only there: a newer launcher with other component versions fetches them again
    $have = { param($rel) $p = Join-Path $dst (Split-Path -Leaf $rel); (Test-Path -LiteralPath $p) -and (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash -eq $c.files[$rel] }
    if (@($c.files.Keys | Where-Object { -not (& $have $_) }).Count -eq 0) { return $true }
    if ($CheckOnly) { return $false }
    $tmp = Join-Path ([IO.Path]::GetTempPath()) "klbf-$Name-$([guid]::NewGuid().ToString('N'))"
    try {
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null
        $get = { param($rel) $out = Join-Path $tmp ($rel -replace '/', '_'); Invoke-WebRequest -UseBasicParsing "$($c.base)/$rel" -OutFile $out; $out }
        if ($c.zip) {
            Invoke-WebRequest -UseBasicParsing $c.zip -OutFile (Join-Path $tmp 'p.zip')
            Expand-Archive -LiteralPath (Join-Path $tmp 'p.zip') -DestinationPath (Join-Path $tmp 'p')
            $get = { param($rel) Join-Path (Join-Path $tmp 'p') $rel }
        }
        $got = [ordered]@{}
        foreach ($rel in $c.files.Keys) {
            $f = & $get $rel
            if ((Get-FileHash -Algorithm SHA256 -LiteralPath $f).Hash -ne $c.files[$rel]) { return $false }
            $got[$rel] = $f
        }
        $lic = & $get $c.license[0]
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        foreach ($rel in $got.Keys) { Copy-Item -LiteralPath $got[$rel] -Destination (Join-Path $dst (Split-Path -Leaf $rel)) -Force }
        Copy-Item -LiteralPath $lic -Destination (Join-Path $dst $c.license[1]) -Force
        $true
    } catch { $false } finally { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
}

# (27/11) Updates. This build's version: the VERSION file of the release zip ('' in a source checkout, which
# is never updated).
function Get-KlbfVersion([string]$BuildDir = $PSScriptRoot) {
    $f = Join-Path $BuildDir 'VERSION'
    if (Test-Path -LiteralPath $f -PathType Leaf) { (Get-Content -LiteralPath $f -TotalCount 1).Trim() } else { '' }
}

# The latest GitHub release when it is newer than $Current: @{ tag; page; zip; sha } (sha from the release
# notes, '' if absent), else $null (offline, rate limit, not newer, no version). $Url: a local JSON for tests.
function Get-KlbfUpdate([string]$Current, [string]$Url = $(if ($env:KLBF_UPDATE_URL) { $env:KLBF_UPDATE_URL } else { 'https://api.github.com/repos/PastiiJay/klbf-lab/releases/latest' })) {
    $ver = { param($t) if ($t -match '^v(\d+\.\d+\.\d+)$') { [version]$Matches[1] } }
    $have = & $ver $Current
    if (-not $have) { return $null }
    try {
        $r = if ($Url -notmatch '^https?://') { Get-Content -LiteralPath $Url -Raw | ConvertFrom-Json } else {
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            Invoke-RestMethod -UseBasicParsing -TimeoutSec 10 -Uri $Url }
    } catch { return $null }
    $new = & $ver ([string]$r.tag_name)
    $asset = @($r.assets | Where-Object { $_.name -eq 'KLBF-Lab-win64.zip' })[0]
    if (-not $new -or $new -le $have -or -not $asset) { return $null }
    @{ tag = [string]$r.tag_name; page = [string]$r.html_url; zip = [string]$asset.browser_download_url
        sha = $(if ([string]$r.body -match 'SHA-256 of the zip: ([0-9a-fA-F]{64})') { $Matches[1].ToLower() } else { '' }) }
}

# SHA256SUMS.txt of a folder: name -> hash (plain names only, no folders), empty without the file.
function Read-KlbfSums([string]$Dir) {
    $h = @{}
    $f = Join-Path $Dir 'SHA256SUMS.txt'
    if (Test-Path -LiteralPath $f) {
        foreach ($l in Get-Content -LiteralPath $f) { if ($l -match '^([0-9a-fA-F]{64}) \*?([^\\/:*?"<>|]+)$') { $h[$Matches[2]] = $Matches[1].ToLower() } }
    }
    $h
}

# '' when every file of $Dir is listed in its SHA256SUMS.txt with the right hash and every listed file is
# there; else what is wrong.
function Test-KlbfSums([string]$Dir) {
    $sums = Read-KlbfSums $Dir
    if (-not $sums.Count) { return 'SHA256SUMS.txt is missing or empty' }
    foreach ($i in Get-ChildItem -LiteralPath $Dir -Force | Where-Object Name -ne 'SHA256SUMS.txt') {
        if ($i.PSIsContainer -or -not $sums.ContainsKey($i.Name)) { return "$($i.Name) is not listed in SHA256SUMS.txt" }
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $i.FullName).Hash.ToLower() -ne $sums[$i.Name]) { return "$($i.Name) does not match SHA256SUMS.txt" }
    }
    foreach ($n in $sums.Keys) { if (-not (Test-Path -LiteralPath (Join-Path $Dir $n) -PathType Leaf)) { return "$n is missing" } }
    ''
}

# Whether a program of the build folder runs (the game is victory.exe).
function Test-KlbfRunning([string]$BuildDir) {
    $d = [IO.Path]::GetFullPath($BuildDir).TrimEnd('\') + '\'
    [bool](Get-Process -ErrorAction SilentlyContinue | Where-Object { try { $_.Path -and $_.Path.StartsWith($d, [StringComparison]::OrdinalIgnoreCase) } catch { $false } })
}

# The checked new files ($Stage) into the build folder: the files of the old and new SHA256SUMS.txt go to
# "<folder>.previous" (kept until the next update), the new ones move in; any failure puts everything back
# and throws. Other files stay (bench-results, D3D12, ffx, game-folder.txt).
function Switch-KlbfBuild([string]$Stage, [string]$BuildDir) {
    $prev = "$($BuildDir.TrimEnd('\')).previous"
    if (Test-Path -LiteralPath $prev) { Remove-Item -LiteralPath $prev -Recurse -Force }
    New-Item -ItemType Directory -Path $prev | Out-Null
    $new = @(@((Read-KlbfSums $Stage).Keys) + 'SHA256SUMS.txt')
    $old = @(@((Read-KlbfSums $BuildDir).Keys) + $new | Sort-Object -Unique | Where-Object { Test-Path -LiteralPath (Join-Path $BuildDir $_) -PathType Leaf })
    $out = New-Object Collections.ArrayList; $in = New-Object Collections.ArrayList
    try {
        foreach ($n in $old) { Move-Item -LiteralPath (Join-Path $BuildDir $n) -Destination (Join-Path $prev $n) -ErrorAction Stop; [void]$out.Add($n) }
        foreach ($n in $new) { Move-Item -LiteralPath (Join-Path $Stage $n) -Destination (Join-Path $BuildDir $n) -ErrorAction Stop; [void]$in.Add($n) }
    } catch {
        $e = $_
        foreach ($n in $in) { Move-Item -LiteralPath (Join-Path $BuildDir $n) -Destination (Join-Path $Stage $n) -Force -ErrorAction SilentlyContinue }
        foreach ($n in $out) { Move-Item -LiteralPath (Join-Path $prev $n) -Destination (Join-Path $BuildDir $n) -Force -ErrorAction SilentlyContinue }
        throw $e
    }
}

# The Update button: the release zip downloaded, checked against the release notes' SHA-256 and each file
# against its SHA256SUMS.txt and VERSION, then switched in (Switch-KlbfBuild). Never touches Saved Games.
# @{ ok; message }.
function Install-KlbfUpdate($Update, [string]$BuildDir) {
    $ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'
    $no = { param($m) @{ ok = $false; message = "Update not installed: $m" } }
    $tmp = Join-Path ([IO.Path]::GetTempPath()) "klbf-update-$([guid]::NewGuid().ToString('N'))"
    $stage = "$($BuildDir.TrimEnd('\')).new"
    try {
        if ($Update.sha -notmatch '^[0-9a-f]{64}$') { return & $no 'the release notes give no SHA-256 for the zip.' }
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null
        $zip = Join-Path $tmp 'update.zip'
        try {
            if ($Update.zip -notmatch '^https?://') { Copy-Item -LiteralPath $Update.zip -Destination $zip }
            else {
                [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 300 -Uri $Update.zip -OutFile $zip
            }
        } catch { return & $no 'the download failed (offline?). Nothing was changed.' }
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLower() -ne $Update.sha) { return & $no 'the downloaded zip does not match the SHA-256 in the release notes. Nothing was changed.' }
        if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
        Expand-Archive -LiteralPath $zip -DestinationPath $stage
        $bad = Test-KlbfSums $stage
        if ($bad) { return & $no "$bad. Nothing was changed." }
        if ((Get-KlbfVersion $stage) -ne $Update.tag) { return & $no "the zip's VERSION is not $($Update.tag). Nothing was changed." }
        if (Test-KlbfRunning $BuildDir) { return & $no 'the game is running. Close it, then click Update again.' }
        try { Switch-KlbfBuild $stage $BuildDir } catch { return & $no "a file could not be replaced ($($_.Exception.Message)). The previous version was put back." }
        @{ ok = $true; message = "Updated to $($Update.tag) $([char]0x2014) shaders will be prepared again" }
    } catch { & $no "$($_.Exception.Message). Nothing was changed." }
    finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# The update line beside the version: "Update available: v26.11.2 - Update   What's new" (two links).
function Set-KlbfUpdateLine($c, $Update) {
    $c.updateInfo = $Update
    $t = "Update available: $($Update.tag) $([char]0x2014) "
    $c.update.Text = "${t}Update   What's new"
    $c.update.Links.Clear()
    [void]$c.update.Links.Add($t.Length, 6, 'update')
    [void]$c.update.Links.Add($t.Length + 9, 10, 'notes')
    $c.update.Visible = $true
}

# Desktop\KLBF Lab.lnk (the Desktop known folder) from a spec; $Path for tests.
function New-KlbfShortcut($Spec, [string]$Path = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'KLBF Lab.lnk')) {
    if ($Spec.vbs) { Set-Content -LiteralPath $Spec.vbs -Value (Get-KlbfVbsText) -Encoding ascii }
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($Path)
    $s.TargetPath = $Spec.target; $s.Arguments = $Spec.arguments; $s.WorkingDirectory = $Spec.workdir
    if (Test-Path -LiteralPath $Spec.icon) { $s.IconLocation = "$($Spec.icon),0" }
    $s.Save()
    $Path
}

# The button and the first-run offer: replaces an existing shortcut only after $Ask says yes. Its path, or ''.
function Set-KlbfShortcut($Spec, $Ask, [string]$Path = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'KLBF Lab.lnk')) {
    if ((Test-Path -LiteralPath $Path) -and -not (& $Ask 'Replace the KLBF Lab shortcut on the desktop with one for this build?')) { return '' }
    New-KlbfShortcut $Spec $Path
}

# Once, at the end of the first setup: "Create a desktop shortcut?" (the answer remembered beside the settings).
function Invoke-KlbfShortcutOffer($Spec, $Ask, [string]$Path = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'KLBF Lab.lnk')) {
    $mark = Join-Path (Get-KlbfLauncherDir) 'shortcut-asked.txt'
    if (Test-Path -LiteralPath $mark) { return '' }
    # (27/11) a shortcut already made (by hand or the Quick tab): no question
    if (Test-Path -LiteralPath $Path) { return '' }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $mark) | Out-Null
    $yes = & $Ask 'Create a desktop shortcut for KLBF Lab?'
    Set-Content -LiteralPath $mark -Value $(if ($yes) { 'yes' } else { 'no' }) -Encoding utf8
    if (-not $yes) { return '' }
    Set-KlbfShortcut $Spec $Ask $Path
}

# (27/11) This process groups with the game under one KLBF Lab taskbar button (the game sets the same id),
# shown with the windows' flask instead of PowerShell's icon.
function Set-KlbfAppId {
    try {
        if (-not ('KlbfAppId' -as [type])) { Add-Type -Namespace '' -Name KlbfAppId -MemberDefinition '[DllImport("shell32.dll", CharSet = CharSet.Unicode)] public static extern int SetCurrentProcessExplicitAppUserModelID(string id);' }
        [KlbfAppId]::SetCurrentProcessExplicitAppUserModelID('PastiiJay.KLBFLab') -eq 0
    } catch { $false }
}

# Every key of KLBF Lab, as the Quick tab lists them.
function Get-KlbfShortcuts {
    @(
        'F8    Screenshot (in the game): the game image, with a small KLBF Lab mark'
        'F10  Settings: closes the game and opens this window; Play starts it again'
        'F11  Restart: closes the game and starts it again with the same settings'
        'F10  during a benchmark: stops it (F8 and F11 do nothing during a run)'
        'Alt+Enter  borderless fullscreen on and off (in the game)'
        'Shift  held while starting: opens these settings first'
    )
}

# Whether $Path is a game folder play.ps1 can start: it has game_scarlett_master.exe.
function Test-KlbfGameFolder([string]$Path) {
    [bool]($Path -and (Test-Path -LiteralPath (Join-Path $Path 'game_scarlett_master.exe') -PathType Leaf))
}

# Room on the drive of the cache. 10 GB measured 27/11: a full shader and pipeline cache 7.2 GB, saves and
# logs 11 MB, 10 save backups under 0.2 GB, plus margin for the cache to grow.
$script:KlbfSpaceGB = 10
# (27/11) The shader cache: about 7.2 GB once full (measured 27/11). Only ray tracing and -NoProbeRays
# (play.ps1 switches) change the translated shaders, so the cache grows again after one of them changes;
# none of the window's choices do (profile, upscaler, preset, output, frame generation, hooks).
$script:KlbfCacheFullGB = 7.2

function Get-KlbfCacheDir([string]$Cache) { if ($Cache) { $Cache } else { Join-Path (Get-KlbfHome) 'cache' } }

# Bytes in the cache folder (0 without one).
function Get-KlbfCacheBytes([string]$Cache) {
    $d = Get-KlbfCacheDir $Cache
    if (-not (Test-Path -LiteralPath $d)) { return 0 }
    [int64](Get-ChildItem -LiteralPath $d -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
}

# Before Prepare shaders: what the cache still needs (the full size less what it holds) against the free
# room on its drive. $FreeGB (tests) stands for the drive's.
function Test-KlbfPrepareSpace([string]$Cache, $FreeGB = $null) {
    $have = (Get-KlbfCacheBytes $Cache) / 1GB
    $need = [math]::Round([math]::Max(0.0, $script:KlbfCacheFullGB - $have), 1)
    if ($null -eq $FreeGB) { $FreeGB = (Get-KlbfSpace $Cache).freeGB }
    @{ ok = $FreeGB -ge $need; needGB = $need; freeGB = $FreeGB }
}

# Clear shader cache: the cache folder's contents go, but the list of effects the game created
# (pipelines\known-pipelines.bin, what is made ahead next time) is kept. Refuses anything but a folder
# named "cache" (never a drive or a profile folder).
function Clear-KlbfShaderCache([string]$Cache) {
    $d = Get-KlbfCacheDir $Cache
    if (-not (Test-Path -LiteralPath $d -PathType Container) -or (Split-Path -Leaf $d) -ne 'cache') { throw 'Not a KLBF Lab cache folder: nothing deleted.' }
    $keep = Join-Path $d 'pipelines\known-pipelines.bin'
    foreach ($i in (Get-ChildItem -LiteralPath $d -Force)) {
        if ($i.Name -eq 'pipelines' -and (Test-Path -LiteralPath $keep)) {
            Get-ChildItem -LiteralPath $i.FullName -Force | Where-Object Name -ne 'known-pipelines.bin' | Remove-Item -Recurse -Force
        } else { Remove-Item -LiteralPath $i.FullName -Recurse -Force }
    }
}

function Get-KlbfSpace([string]$Cache) {
    if (-not $Cache) { $Cache = Get-KlbfCacheDir '' }
    $root = [IO.Path]::GetPathRoot($Cache)
    $free = try { [math]::Floor((New-Object IO.DriveInfo($root)).AvailableFreeSpace / 1GB) } catch { -1 }
    @{ drive = $root.TrimEnd('\'); needGB = $script:KlbfSpaceGB; freeGB = $free; ok = $free -ge $script:KlbfSpaceGB }
}

# The saved choice, or $null without a file (first run). A damaged file gives the defaults.
function Read-KlbfSettings([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { ConvertTo-KlbfChoice (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json) } catch { Get-KlbfDefaults }
}

function Save-KlbfSettings($Choice, [string]$Path) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    ([pscustomobject]$Choice | ConvertTo-Json) | Set-Content -LiteralPath $Path -Encoding utf8
}

function Get-KlbfPresetPath([string]$Name) {
    $safe = ($Name -replace '[^\w \-\.]', '').Trim().Trim('.')
    if (-not $safe) { return $null }
    Join-Path (Join-Path (Get-KlbfLauncherDir) 'presets') "$safe.json"
}

function Get-KlbfPresetNames {
    $dir = Join-Path (Get-KlbfLauncherDir) 'presets'
    if (-not (Test-Path -LiteralPath $dir)) { return @() }
    @(Get-ChildItem -LiteralPath $dir -Filter *.json | Sort-Object Name | ForEach-Object { $_.BaseName })
}

# The play.ps1 parameters a choice stands for. Profile S is the title's FidelityFX path sent to the PC FSR
# (-Fsr): its render size and output are the title's. Frame generation is fed by the FSR upscale.
function Get-KlbfSwitches($Choice) {
    $on = [switch]$true
    $s = [ordered]@{}
    $fsr = $Choice.profile -eq 'S' -or $Choice.upscaler -eq 'fsr'
    if ($Choice.profile -eq 'S') {
        $s.Fsr = $on
    } else {
        if ($fsr) { $s.FsrX = $on; $s.FsrQuality = $Choice.preset }
        $s.Output = $Choice.output
    }
    if ($fsr -and $Choice.frameGen) { $s.FrameGen = $on }
    if ($Choice.advanced) { $s.Hooks = $on; $s.TexTrim = $on }
    $s
}

# (27/11) The monitors in the order Windows lists them (KLBF_WINDOW_MONITOR's numbers): @{ primary; device;
# mode (the real mode, DPI-independent); bounds; work = x, y, width, height of the work area }.
function Get-KlbfScreens {
        try {
            Add-Type -AssemblyName System.Windows.Forms
            if (-not ('KlbfDisplay' -as [type])) {
                Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class KlbfDisplay {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct DEVMODE {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
        public short dmSpecVersion, dmDriverVersion, dmSize, dmDriverExtra; public int dmFields;
        public int dmPositionX, dmPositionY, dmDisplayOrientation, dmDisplayFixedOutput;
        public short dmColor, dmDuplex, dmYResolution, dmTTOption, dmCollate;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
        public short dmLogPixels; public int dmBitsPerPel, dmPelsWidth, dmPelsHeight, dmDisplayFlags, dmDisplayFrequency;
        public int dmICMMethod, dmICMIntent, dmMediaType, dmDitherType, dmReserved1, dmReserved2, dmPanningWidth, dmPanningHeight;
    }
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern bool EnumDisplaySettings(string device, int mode, ref DEVMODE dm);
    public static string Mode(string device) {
        var dm = new DEVMODE(); dm.dmSize = (short)Marshal.SizeOf(typeof(DEVMODE));
        return EnumDisplaySettings(device, -1, ref dm) ? dm.dmPelsWidth + "x" + dm.dmPelsHeight : "";
    }
}
'@
            }
            @([System.Windows.Forms.Screen]::AllScreens | ForEach-Object {
                $w = $_.WorkingArea
                @{ primary = $_.Primary; device = $_.DeviceName; mode = [KlbfDisplay]::Mode($_.DeviceName); bounds = '{0}x{1}' -f $_.Bounds.Width, $_.Bounds.Height; work = @($w.X, $w.Y, $w.Width, $w.Height) }
            })
        } catch { @() }
}

# The PRIMARY monitor's real mode, where the game window opens by default; the first video controller's
# mode was another monitor's on a two-screen PC. $Screens (tests): as Get-KlbfScreens gives them.
function Get-KlbfScreenSize($Screens = $null) {
    if ($null -eq $Screens) { $Screens = Get-KlbfScreens }
    $p = @($Screens | Where-Object { $_.primary }) + @($Screens) | Select-Object -First 1
    if (-not $p) { return '1920x1080' }
    if ($p.mode) { $p.mode } else { $p.bounds }
}

# (27/11) Display: "Screen 1 (main) 5120x1440", "Screen 2 2560x1440" (no device names).
function Get-KlbfDisplayNames($Screens) {
    $i = 0
    foreach ($m in @($Screens)) { $i++; 'Screen {0}{1} {2}' -f $i, $(if ($m.primary) { ' (main)' } else { '' }), $(if ($m.mode) { $m.mode } else { $m.bounds }) }
}

# The monitor a choice's display names (-1 or one that is gone: the main one), as an index in $Screens.
function Get-KlbfDisplayIndex($Choice, $Screens) {
    $list = @($Screens)
    $d = if ($Choice) { [int]$Choice.display } else { -1 }
    if ($d -ge 0 -and $d -lt $list.Count) { return $d }
    for ($i = 0; $i -lt $list.Count; $i++) { if ($list[$i].primary) { return $i } }
    0
}

# The top-left corner that centres a $Size (width, height) window on a work area (x, y, width, height),
# kept inside it when larger. Negative coordinates (a monitor left of the main one) are fine.
function Get-KlbfCentre($Work, $Size) {
    $x = $Work[0] + [math]::Max(0, [math]::Floor(($Work[2] - $Size[0]) / 2))
    $y = $Work[1] + [math]::Max(0, [math]::Floor(($Work[3] - $Size[1]) / 2))
    @([int]$x, [int]$y)
}

function Test-KlbfShift {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        ([System.Windows.Forms.Control]::ModifierKeys -band [System.Windows.Forms.Keys]::Shift) -ne 0
    } catch { $false }
}

# What the translated shaders depend on that the launcher knows of: the build and the settings that change
# the translation. Of the window's choices none does (profile, upscaler, preset, output, frame generation,
# hooks and texture trim are not read by the translator); ray tracing and -NoProbeRays do.
function Get-KlbfShaderStamp([string]$Victory, [string]$Build, $Probe, $Rays) {
    $built = Get-Item -LiteralPath $Victory
    '{0}|{1}|{2}|probe={3}|rays={4}' -f $built.Length, $built.LastWriteTimeUtc.Ticks, $Build, $Probe, $Rays
}

# Why shaders must be prepared again, from the stamp they were prepared for ($Have, '' if never).
function Get-KlbfPrepareReason([string]$Have, [string]$Want) {
    $build = { param($s) ($s -split '\|')[0..2] -join '|' }
    if ($Have -and (& $build $Have) -eq (& $build $Want)) {
        'You changed how the game is rendered: shaders must be prepared again. This takes a few minutes.'
    } else {
        'This build of KLBF Lab has not prepared its shaders yet. This takes a few minutes.'
    }
}

# This PC: @{ gpu; vramGB; ramGB; screen }. The card's memory from the driver's registry entry
# (HardwareInformation.qwMemorySize; Win32_VideoController's AdapterRAM stops at 4 GB), the largest card.
function Get-KlbfHardware {
    $gpu = ''; $vram = 0
    $class = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
    foreach ($k in (Get-ChildItem -LiteralPath $class -ErrorAction SilentlyContinue | Where-Object PSChildName -match '^\d{4}$')) {
        $p = Get-ItemProperty -LiteralPath $k.PSPath -ErrorAction SilentlyContinue
        $bytes = $p.'HardwareInformation.qwMemorySize'
        if ($bytes -is [byte[]]) { $bytes = [BitConverter]::ToUInt64($bytes, 0) }
        if ($p.DriverDesc -and $bytes -and [double]$bytes -gt $vram) { $gpu = [string]$p.DriverDesc; $vram = [double]$bytes }
    }
    if (-not $gpu) {
        $v = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Sort-Object AdapterRAM -Descending | Select-Object -First 1
        if ($v) { $gpu = [string]$v.Name; $vram = [double]$v.AdapterRAM }
    }
    # Installed memory (Windows' total leaves out what is reserved: 31.2 GB on a 32 GB PC)
    $ram = 0
    try {
        if (-not ('KlbfMem' -as [type])) { Add-Type -Namespace '' -Name KlbfMem -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool GetPhysicallyInstalledSystemMemory(out ulong kb);' }
        $kb = [uint64]0
        if ([KlbfMem]::GetPhysicallyInstalledSystemMemory([ref]$kb)) { $ram = [double]$kb * 1KB }
    } catch {}
    if (-not $ram) { $ram = [double](Get-CimInstance Win32_PhysicalMemory -ErrorAction SilentlyContinue | Measure-Object Capacity -Sum).Sum }
    if (-not $ram) { $ram = [double](Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue).TotalPhysicalMemory }
    @{ gpu = $gpu; vramGB = [math]::Round($vram / 1GB); ramGB = [math]::Round($ram / 1GB); screen = Get-KlbfScreenSize }
}

# The card's generation from its name (the title needs mesh shaders: RDNA 2 and later on AMD).
function Get-KlbfGpuKind([string]$Name) {
    if ($Name -match 'Radeon.*RX\s*(\d)\d{3}') {
        switch ($Matches[1]) {
            '9' { return @{ gen = 'RDNA 4'; level = 'ok'; text = 'supported' } }
            '7' { return @{ gen = 'RDNA 3'; level = 'ok'; text = 'supported, less tested' } }
            '6' { return @{ gen = 'RDNA 2'; level = 'warn'; text = 'experimental (a fix for the card stopping is in progress)' } }
            '5' { return @{ gen = 'RDNA 1'; level = 'bad'; text = 'not supported (no mesh shaders, which the title needs)' } }
        }
    }
    # (27/11, nuit/etude-nvidia.md) Other cards by what they can run: the title draws with mesh shaders
    $nomesh = 'not supported (no mesh shaders, which the title needs)'
    if ($Name -match 'RTX\s*(20|30|40|50)\d\d') { return @{ gen = 'NVIDIA RTX'; level = 'warn'; text = 'supported, Profile S recommended' } }
    if ($Name -match 'GTX\s*16\d\d') { return @{ gen = 'NVIDIA GTX 16'; level = 'warn'; text = 'not tested' } }
    if ($Name -match 'GTX|GeForce\s*\d{3}') { return @{ gen = ''; level = 'bad'; text = $nomesh } }
    if ($Name -match '\bArc\b') { return @{ gen = 'Intel Arc'; level = 'warn'; text = 'not tested' } }
    if ($Name -match 'Intel.*(UHD|Iris|HD Graphics)') { return @{ gen = ''; level = 'bad'; text = $nomesh } }
    if ($Name -match 'Radeon.*(RX\s*[45]\d0\b|Vega|R[579]\s)') { return @{ gen = ''; level = 'bad'; text = $nomesh } }
    @{ gen = ''; level = 'warn'; text = 'not known' }
}

# One verdict per line: @{ what; level = ok|warn|bad; text }.
function Get-KlbfVerdicts($Hw) {
    $k = Get-KlbfGpuKind $Hw.gpu
    $name = if ($Hw.gpu) { $Hw.gpu + $(if ($k.gen) { " ($($k.gen))" } else { '' }) } else { 'not found' }
    @(
        @{ what = 'Graphics card'; level = $k.level; text = "$name - $($k.text)" }
        $(if ($Hw.vramGB -ge 16) { @{ what = 'Video memory'; level = 'ok'; text = "$($Hw.vramGB) GB" } }
          elseif ($Hw.vramGB -ge 12) { @{ what = 'Video memory'; level = 'warn'; text = "$($Hw.vramGB) GB - 16 GB recommended" } }
          else { @{ what = 'Video memory'; level = 'bad'; text = "$($Hw.vramGB) GB - under 12 GB: Profile S recommended" } })
        $(if ($Hw.ramGB -ge 32) { @{ what = 'Memory'; level = 'ok'; text = "$($Hw.ramGB) GB" } }
          elseif ($Hw.ramGB -ge 24) { @{ what = 'Memory'; level = 'warn'; text = "$($Hw.ramGB) GB - 32 GB recommended" } }
          else { @{ what = 'Memory'; level = 'bad'; text = "$($Hw.ramGB) GB - may run out of memory (the game uses about 21 GB)" } })
        @{ what = 'Screen'; level = 'ok'; text = $Hw.screen }
    )
}

# The upscaler's name on this card: what ran last time if the game said (klbf-upscaler.txt in the cache,
# written from the DLL's provider version), else AMD FSR 4 on RDNA 4 and AMD FSR 3.1 elsewhere (the DLL
# falls back to FSR 3.1 without RDNA 4).
function Get-KlbfUpscalerName($Hw, [string]$Cache) {
    if (-not $Cache) { $Cache = Join-Path $env:LOCALAPPDATA 'klbf-lab\cache' }
    $ran = Join-Path $Cache 'klbf-upscaler.txt'
    if (Test-Path -LiteralPath $ran) { $t = (Get-Content -LiteralPath $ran -TotalCount 1).Trim(); if ($t -match '^AMD FSR [\d.]+$') { return $t } }
    if ((Get-KlbfGpuKind $Hw.gpu).gen -eq 'RDNA 4') { 'AMD FSR 4' } else { 'AMD FSR 3.1' }
}

# (27/11, the owner) The first start's setting, the same for every PC until players' reports say more:
# Profile X, the built-in (console) upscaler, the largest title mode not above the screen, no frame
# generation, hooks and texture trim. The Your PC panel's button applies the hardware table instead.
function Get-KlbfUniversal($Hw) {
    # (27/11, owner) Every PC: Profile X, AMD FSR (FSR 4 on RDNA 4, FSR 3.1 elsewhere) Ultra Performance, no
    # frame generation, Advanced on. Where FSR cannot start, the game falls back to its own upscaler by itself.
    $c = Get-KlbfDefaults
    $c.upscaler = 'fsr'; $c.preset = 'ultraperformance'; $c.frameGen = $false; $c.advanced = $true
    # (27/11) Profile X: output 3840x2160 on every screen (Windows scales the window down). Below it the
    # title's HUD and menus get too big and their texts overlap (known issue); the cost is the upscaler's.
    $c.output = '3840x2160'
    $c
}

# Whether this PC's card is in $KlbfTested.
function Test-KlbfTested($Hw) {
    $gen = (Get-KlbfGpuKind $Hw.gpu).gen
    [bool]@($script:KlbfTested | Where-Object { $_.gen -eq $gen -and $Hw.vramGB -ge $_.minVramGB }).Count
}

# The settings for this PC (27/11): Profile X only for RDNA 3 and 4 with 16 GB or more (RDNA 4: the owner's
# default; RDNA 3: output = the screen if it is one of the title's modes, else 2560x1440). Everything else,
# RDNA 2 and any card under 16 GB included, Profile S (a 12 GB RX 6750 XT ran 25 min stable on it).
function Get-KlbfRecommended($Hw) {
    $c = Get-KlbfDefaults
    $gen = (Get-KlbfGpuKind $Hw.gpu).gen
    # Frame generation only on RDNA 4 (elsewhere FSR 3.1 frame generation on a base far under AMD's 60 fps)
    if ($gen -notin 'RDNA 3', 'RDNA 4' -or $Hw.vramGB -lt 16) { $c.profile = 'S'; $c.frameGen = $gen -eq 'RDNA 4'; return $c }
    if ($gen -eq 'RDNA 4') { return $c }
    $c.frameGen = $false
    $c
}

function New-KlbfFont([float]$Size = 9, [string]$Style = 'Regular') {
    New-Object System.Drawing.Font('Segoe UI', $Size, [System.Drawing.FontStyle]::$Style)
}

# KLBF Lab's colours, as on the game window's frame (victory-gpu shell.rs set_colors): violet border,
# near-black caption, white text.
# Violet only for accents (the Play button, the header line); the rest in neutral greys.
$script:KlbfTheme = @{ back = 0x15101F; field = 0x2A2A30; line = 0x55555E; violet = 0x7C3AED; text = 0xFFFFFF; soft = 0xB8B8C0 }

function Get-KlbfColor([int]$Rgb) {
    [System.Drawing.Color]::FromArgb(($Rgb -shr 16) -band 255, ($Rgb -shr 8) -band 255, $Rgb -band 255)
}

# Our icon: beside this script (a version folder) or in the repository (apps\victory\klbf-lab.ico).
function Get-KlbfIconPath {
    foreach ($p in (Join-Path $PSScriptRoot 'klbf-lab.ico'), (Join-Path $PSScriptRoot 'apps\victory\klbf-lab.ico')) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
}

# The icon's largest picture: its PNG frame when it has one (Icon.ToBitmap garbles those), else 48 x 48.
function Get-KlbfLogo([string]$Ico) {
    $b = [IO.File]::ReadAllBytes($Ico)
    $best = $null; $size = 0
    for ($i = 0; $i -lt [BitConverter]::ToUInt16($b, 4); $i++) {
        $e = 6 + 16 * $i
        $w = if ($b[$e] -eq 0) { 256 } else { $b[$e] }
        $len = [BitConverter]::ToInt32($b, $e + 8); $at = [BitConverter]::ToInt32($b, $e + 12)
        if ($w -gt $size -and $b[$at] -eq 0x89 -and $b[$at + 1] -eq 0x50) { $size = $w; $best = @($at, $len) }
    }
    if ($best) { return [System.Drawing.Image]::FromStream((New-Object IO.MemoryStream($b, $best[0], $best[1]))) }
    (New-Object System.Drawing.Icon($Ico, 48, 48)).ToBitmap()
}

# A form in KLBF Lab's colours and icon; on Windows 11 its frame too (violet border, dark caption).
function Set-KlbfTheme($Form) {
    $Form.BackColor = Get-KlbfColor $script:KlbfTheme.back
    $Form.ForeColor = Get-KlbfColor $script:KlbfTheme.text
    $Form.Font = New-KlbfFont
    $ico = Get-KlbfIconPath
    if ($ico) { $Form.Icon = New-Object System.Drawing.Icon($ico) }
    # (27/11) Started hidden (Start-Process -WindowStyle Hidden, a scheduled task), the process's first
    # ShowWindow takes the hidden state from its STARTUPINFO: the form never appeared and the launcher waited
    # forever. Shown again here (a second ShowWindow is not overridden) and brought to the front.
    $Form.add_Shown({
        param($sender)
        try {
            # Every dialog (bench guide, Prepare now/Later, the shaders' progress...): centred on the chosen
            # display; the settings window places itself (Manual).
            if ($sender.StartPosition -eq 'CenterScreen') {
                try {
                    $scr = @(Get-KlbfScreens)
                    if ($scr.Count) {
                        $i = Get-KlbfDisplayIndex (Read-KlbfSettings (Join-Path (Get-KlbfLauncherDir) 'settings.json')) $scr
                        $sender.Location = New-Object System.Drawing.Point(@(Get-KlbfCentre $scr[$i].work @($sender.Width, $sender.Height)))
                    }
                } catch {}
            }
            if (-not ('KlbfShow2' -as [type])) {
                Add-Type -Namespace '' -Name KlbfShow2 -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
[DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint f);
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
'@
            }
            $h = $sender.Handle
            [void][KlbfShow2]::ShowWindow($h, 1) # SW_SHOWNORMAL, a second time: the first one may have been hidden
            # Topmost once then back (brought in front of everything, not kept there), then focused
            [void][KlbfShow2]::SetWindowPos($h, [IntPtr](-1), 0, 0, 0, 0, 0x43) # HWND_TOPMOST, NOMOVE|NOSIZE|SHOWWINDOW
            [void][KlbfShow2]::SetWindowPos($h, [IntPtr](-2), 0, 0, 0, 0, 0x43) # HWND_NOTOPMOST
            [void][KlbfShow2]::SetForegroundWindow($h)
            $sender.Activate()
        } catch {}
    })
    $Form.add_HandleCreated({
        param($sender)
        try {
            if (-not ('KlbfDwm' -as [type])) {
                Add-Type -Namespace '' -Name KlbfDwm -MemberDefinition '[DllImport("dwmapi.dll")] public static extern int DwmSetWindowAttribute(IntPtr h, int a, ref int v, int n);'
            }
            # The game window's exact values (victory-gpu shell.rs set_colors): DWMWA_BORDER_COLOR 34 = 0x7C3AED,
            # CAPTION_COLOR 35 = 0x15101F, TEXT_COLOR 36 = 0xFFFFFF, as COLORREF (0x00BBGGRR)
            foreach ($a in @(@(34, 0xED3A7C), @(35, 0x1F1015), @(36, 0xFFFFFF))) { $v = $a[1]; [void][KlbfDwm]::DwmSetWindowAttribute($sender.Handle, $a[0], [ref]$v, 4) }
        } catch {}
    })
}

function Set-KlbfButton($Button, [bool]$Main = $false) {
    $Button.FlatStyle = 'Flat'
    $Button.FlatAppearance.BorderColor = Get-KlbfColor $(if ($Main) { $script:KlbfTheme.violet } else { $script:KlbfTheme.line })
    $Button.BackColor = Get-KlbfColor $(if ($Main) { $script:KlbfTheme.violet } else { $script:KlbfTheme.field })
    $Button.ForeColor = Get-KlbfColor $script:KlbfTheme.text
    $Button.UseVisualStyleBackColor = $false
}

$script:KlbfRepo = 'https://github.com/PastiiJay/klbf-lab'

# ---- Saves: the title's containers in <data>\<title id>\saves\<provider>\<user>\<name>~<hash>\v<N>\ ----
# (victory-hle xgamesave.rs). A version holds `manifest` (u32-prefixed strings: name, display name, blob
# names) and one file per blob: 0 the save, 1 its header, whose title (UTF-16 at 0x24) the menu shows.

function Get-KlbfSavesRoot([string]$Data) {
    if (-not $Data) { $Data = Join-Path (Get-KlbfHome) 'data' }
    Get-ChildItem -LiteralPath $Data -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName 'saves' } | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

function Get-KlbfBackupDir { if ($env:KLBF_LAUNCHER_DIR) { Join-Path (Split-Path -Parent $env:KLBF_LAUNCHER_DIR) 'save-backups' } else { Join-Path (Get-KlbfHome) 'save-backups' } }

function Read-KlbfManifest([string]$Path) {
    $b = [IO.File]::ReadAllBytes($Path); $at = 0; $out = @()
    while ($at + 4 -le $b.Length) {
        $n = [BitConverter]::ToUInt32($b, $at); $at += 4
        if ($at + $n -gt $b.Length) { return $null }
        $out += [Text.Encoding]::UTF8.GetString($b, $at, $n); $at += $n
    }
    , $out
}

function Write-KlbfManifest([string]$Path, [string[]]$Names) {
    $ms = New-Object IO.MemoryStream
    foreach ($s in $Names) { $u = [Text.Encoding]::UTF8.GetBytes($s); $ms.Write([BitConverter]::GetBytes([uint32]$u.Length), 0, 4); $ms.Write($u, 0, $u.Length) }
    [IO.File]::WriteAllBytes($Path, $ms.ToArray())
}

# The game's saves (not the profile's settings containers), as its Load menu lists them: the autosave first,
# then the newest first. Seen in the game (25/11), not read from the title: the menu sorts by date.
function Get-KlbfSaves([string]$SavesRoot) {
    if (-not $SavesRoot -or -not (Test-Path -LiteralPath $SavesRoot)) { return @() }
    $list = foreach ($dir in (Get-ChildItem -Path (Join-Path $SavesRoot '*\*\*') -Directory -ErrorAction SilentlyContinue | Where-Object Name -match '^SARC.*~[0-9a-f]{16}$')) {
        $v = Get-ChildItem -LiteralPath $dir.FullName -Directory | Where-Object { $_.Name -match '^v(\d+)$' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'manifest')) } |
            Sort-Object { [int]($_.Name.Substring(1)) } -Descending | Select-Object -First 1
        if (-not $v) { continue }
        $names = Read-KlbfManifest (Join-Path $v.FullName 'manifest')
        $title = ''
        $head = Join-Path $v.FullName '1'
        if (Test-Path -LiteralPath $head) {
            $h = [IO.File]::ReadAllBytes($head)
            if ($h.Length -gt 0x26) { $title = ([Text.Encoding]::Unicode.GetString($h, 0x24, [math]::Min(128, $h.Length - 0x24)) -split [char]0)[0] }
        }
        $data = Get-Item -LiteralPath (Join-Path $v.FullName '0') -ErrorAction SilentlyContinue
        [pscustomobject]@{
            name = $names[0]; display = $names[1]; blobs = @($names | Select-Object -Skip 2); title = $title
            auto = $title -like '(Auto)*'; date = $data.LastWriteTime; size = $data.Length; dir = $dir.FullName; version = $v.FullName; slot = ''
        }
    }
    $sorted = @(@($list | Where-Object auto | Sort-Object date -Descending) + @($list | Where-Object { -not $_.auto } | Sort-Object date -Descending))
    $n = 0
    foreach ($s in $sorted) { $s.slot = if ($s.auto) { 'Autosave' } else { $n++; "Slot $n" } }
    $sorted
}

# A zip of the saves folder in save-backups\<date>.zip; the last $Keep are kept. Its path, or $null.
# (27/11) A "saves" folder inside it (left by an old copy into an existing folder) is not the game's: left out.
function Backup-KlbfSaves([string]$SavesRoot, [string]$Dir = (Get-KlbfBackupDir), [int]$Keep = 10) {
    if (-not $SavesRoot -or -not (Test-Path -LiteralPath $SavesRoot)) {
        # (27/11) Said, never silent: an empty folder here hid the saves being in another folder
        Write-Warning "No saves to back up$(if ($SavesRoot) { " in $SavesRoot" } else { ' (no saves folder yet)' })."
        return $null
    }
    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    $zip = Join-Path $Dir ((Get-Date -Format 'yyyy-MM-dd_HH-mm-ss-fff') + '.zip')
    $root = (Get-Item -LiteralPath $SavesRoot).FullName.TrimEnd('\') + '\'
    $a = [IO.Compression.ZipFile]::Open($zip, 'Create')
    try {
        foreach ($f in (Get-ChildItem -LiteralPath $root -Recurse -Force)) {
            $rel = $f.FullName.Substring($root.Length)
            if ($rel -eq 'saves' -or $rel -like 'saves\*') { continue }
            if (-not $f.PSIsContainer) { [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($a, $f.FullName, $rel.Replace('\', '/')) }
            elseif (-not (Get-ChildItem -LiteralPath $f.FullName -Force)) { [void]$a.CreateEntry($rel.Replace('\', '/') + '/') }
        }
    } finally { $a.Dispose() }
    Get-ChildItem -LiteralPath $Dir -Filter *.zip | Sort-Object Name -Descending | Select-Object -Skip $Keep | Remove-Item
    $zip
}

# Puts a backup back: the current saves are backed up first, then replaced by the zip's.
function Restore-KlbfSaves([string]$Zip, [string]$SavesRoot, [string]$Dir = (Get-KlbfBackupDir)) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $before = Backup-KlbfSaves $SavesRoot $Dir
    if (Test-Path -LiteralPath $SavesRoot) {
        if (-not $before) { throw 'No backup of the current saves: nothing restored.' }
        Remove-Item -LiteralPath $SavesRoot -Recurse -Force
    }
    [IO.Compression.ZipFile]::ExtractToDirectory($Zip, $SavesRoot)
    $before
}

# Slot $From's save into slot $To (both existing saves): every save backed up first, then a new version of
# $To with $From's files and $To's own names, committed as the game commits one (v<N>.new renamed to v<N>).
function Copy-KlbfSave($From, $To, [string]$SavesRoot, [string]$Dir = (Get-KlbfBackupDir)) {
    if (-not $From -or -not $To -or $From.dir -eq $To.dir) { throw 'Pick two different slots.' }
    $backup = Backup-KlbfSaves $SavesRoot $Dir
    if (-not $backup) { throw 'No backup: nothing copied.' }
    $n = 1 + (Get-ChildItem -LiteralPath $To.dir -Directory | Where-Object Name -match '^v(\d+)$' | ForEach-Object { [int]$_.Name.Substring(1) } | Measure-Object -Maximum).Maximum
    $new = Join-Path $To.dir "v$n.new"
    New-Item -ItemType Directory -Force -Path $new | Out-Null
    Get-ChildItem -LiteralPath $From.version -File | Where-Object Name -ne 'manifest' | Copy-Item -Destination $new
    Write-KlbfManifest (Join-Path $new 'manifest') (@($To.name, $To.display) + $From.blobs)
    Rename-Item -LiteralPath $new -NewName "v$n"
    $backup
}

# ---- Report a bug and the bench ----

# The newest log of the title (the crash log beside its data), or ''.
function Get-KlbfLatestLog([string]$Data) {
    if (-not $Data) { $Data = Join-Path (Get-KlbfHome) 'data' }
    $l = Get-ChildItem -Path (Join-Path $Data '*\crashes.log') -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($l) { $l.FullName } else { '' }
}

# The block "Report a bug" copies: build, settings, this PC, Windows, the log to attach. Nothing goes in a URL.
function Get-KlbfBugReport($Choice, $Hw, [string]$Victory, [string]$Log, [string]$Windows) {
    $k = Get-KlbfGpuKind $Hw.gpu
    $build = if ($Victory -and (Test-Path -LiteralPath $Victory)) {
        $e = Get-Item -LiteralPath $Victory
        '{0} (victory.exe of {1:yyyy-MM-dd HH:mm})' -f $e.Directory.Name, $e.LastWriteTime
    } else { 'unknown' }
    $git = Join-Path $PSScriptRoot '.git'
    if (Test-Path -LiteralPath $git) { try { $build += ', commit ' + (git -C $PSScriptRoot rev-parse --short HEAD 2>$null) } catch {} }
    $up = if ($Choice.profile -eq 'S') { "the title's FSR (Profile S)" } elseif ($Choice.upscaler -eq 'fsr') { (Get-KlbfUpscalerName $Hw) + ' ' + $script:KlbfPresets[$Choice.preset] } else { "the title's own" }
    @(
        '### System (KLBF Lab Settings)'
        "- KLBF Lab: $build"
        "- Profile: $($Choice.profile)"
        "- Upscaler: $up"
        "- Output: $(if ($Choice.profile -eq 'S') { "the title's" } else { $Choice.output })"
        "- Frame generation: $(if ($Choice.frameGen -and ($Choice.profile -eq 'S' -or $Choice.upscaler -eq 'fsr')) { 'on (FG x2)' } else { 'off' })"
        "- Memory hooks and texture trim: $(if ($Choice.advanced) { 'on' } else { 'off' })"
        "- Graphics card: $($Hw.gpu)$(if ($k.gen) { " ($($k.gen))" }), $($Hw.vramGB) GB"
        "- Memory: $($Hw.ramGB) GB"
        "- Screen: $($Hw.screen)"
        "- Windows: $Windows"
        "- Log to attach: $(if ($Log) { $Log } else { 'none found: copy the text of the console window' })"
    ) -join "`r`n" | Hide-KlbfPaths -Game $Choice.game
}

# (27/11, privacy) A text with no path of this PC: the game folder as <game folder>, the user's profile as
# %USERPROFILE%, the computer's and the user's names out.
function Hide-KlbfPaths([Parameter(ValueFromPipeline)][string]$Text, [string]$Game) {
    if ($Game) { $Text = $Text.Replace($Game.TrimEnd('\'), '<game folder>') }
    if ($env:USERPROFILE) { $Text = $Text.Replace($env:USERPROFILE, '%USERPROFILE%') }
    foreach ($n in $env:COMPUTERNAME, $env:USERNAME) { if ($n -and $n.Length -gt 2) { $Text = [regex]::Replace($Text, '\b' + [regex]::Escape($n) + '\b', '<hidden>') } }
    $Text
}

# What the bench window says (KLBF Lab Bench BM-v1, docs/klbf/BENCH-BM-V1.md). No save is shipped: the
# player makes his own, at the point below.
function Get-KlbfBenchGuide {
    @(
        'KLBF Lab Bench BM-v1: 5 minutes, always the same scene, measured by the build itself.'
        ''
        'Once, before the first bench:'
        '1. Play about 10 minutes with this build, so its shaders and effects are prepared (the bench does not start before).'
        "2. Make the bench save: sleep until the morning in the main character's bed at his house (about 3 % into the story), go down to the laundry room, pick up the fishing book on the workbench, keep it in hand, and save."
        '   It must be your most recent save: the bench presses Continue, which loads the most recent one.'
        ''
        'Each bench:'
        '3. Close other heavy programs. Press Start bench, then touch nothing: after 5 seconds in the world the controller and keyboard are ignored until 5:00.'
        '   F10 stops the run early (the card then says PARTIAL).'
        '4. The result card covers the screen at the end; a button closes it and the game.'
        ''
        'Results: the bench-results folder beside victory.exe (card PNG, JSON, text).'
        "Post them: $script:KlbfRepo/issues/new?template=bench-bm-v1.yml"
        'All your saves are backed up before the run (Saves tab).'
    ) -join "`r`n"
}

# The newest bench result beside victory.exe (bench-results\*.png and its .json): @{ png; json } or $null.
function Get-KlbfLastBench([string]$Victory) {
    if (-not $Victory) { return $null }
    $dir = Join-Path (Split-Path -Parent $Victory) 'bench-results'
    $png = Get-ChildItem -LiteralPath $dir -Filter 'KLBF-Bench-*.png' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $png) { return $null }
    $json = [IO.Path]::ChangeExtension($png.FullName, '.json')
    @{ png = $png.FullName; json = $(if (Test-Path -LiteralPath $json) { $json } else { '' }) }
}

# (27/11) A short text to paste anywhere about a bench result (its JSON): figures, settings, card generation
# and memory, the verdict, the build's commit and the repository. Nothing that names the player or the PC:
# no player, paths, serial, share line, computer or user names.
function Get-KlbfBenchShareText($Result) {
    $fps = $Result.fps.avg; $set = $Result.settings
    $en = [Globalization.CultureInfo]::InvariantCulture
    $fg = [bool]$set.frame_generation
    $head = if ($fg) { [string]::Format($en, '{0:0} FPS (base {1:0})', $fps * 2, $fps) } else { [string]::Format($en, '{0:0.0} FPS', $fps) }
    $up = switch ($set.upscaler_kind) {
        'fsr' {
            $v = [string]$set.upscaler_version
            $name = if ($v -match '^(4|5)\.') { "AMD FSR $($Matches[1])" } elseif ($v -match '^(\d+)\.(\d+)') { "AMD FSR $($Matches[1]).$($Matches[2])" } else { 'AMD FSR' }
            ($name, $set.upscaler_preset | Where-Object { $_ }) -join ' '
        }
        'title' { 'Built-in (console) upscaler' }
        default { 'Native' }
    }
    $rows = { param($wh) if (-not $wh) { return '' }; $h = [int](($wh -split 'x')[1]); if ($h -eq 2160) { '4K' } else { "${h}p" } }
    $res = if ($set.render_size -and $set.render_size -ne $set.output_size) { "$(& $rows $set.render_size) -> $(& $rows $set.output_size)" } else { & $rows $set.output_size }
    $k = Get-KlbfGpuKind ([string]$Result.gpu)
    $card = if ($k.gen) { $k.gen } else { 'GPU' }
    @(
        "KLBF Lab Bench BM-v1: $head, 1% low $([string]::Format($en, '{0:0.0}', $Result.fps.low_1pct))"
        (@("Profile $($set.profile)", $up, $res, "frame generation $(if ($fg) { 'on' } else { 'off' })") | Where-Object { $_ }) -join ' - '
        "$card, $([math]::Round([double]$Result.gpu_vram_gb)) GB VRAM, $([math]::Round([double]$Result.ram_gb)) GB RAM"
        "$($Result.stamp) - KLBF Lab $($Result.sha)"
        $script:KlbfRepo
    ) -join "`r`n"
}

# The play.ps1 switches of a bench run: the player's settings, then -Bench (and -BenchPlayer).
function Get-KlbfBenchSwitches($Choice, [string]$Player) {
    $s = Get-KlbfSwitches $Choice
    $s.Bench = [switch]$true
    $s.BenchPlayer = if ($Player.Trim()) { $Player.Trim() } else { 'anonymous' }
    $s
}

# The data folder of a bench run (27/11): a copy of $Data in which the chosen save is the only save, so
# Continue can only load that one; the player's own saves are never opened by the run. Rebuilt each run.
function New-KlbfBenchData([string]$Data, $Save, [string]$Dest = (Join-Path (Split-Path -Parent (Get-KlbfLauncherDir)) 'bench-data')) {
    if (Test-Path -LiteralPath $Dest) { Remove-Item -LiteralPath $Dest -Recurse -Force }
    # The folder's CONTENTS into a new folder (Copy-Item of a folder into an existing one nests it)
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    foreach ($i in (Get-ChildItem -LiteralPath $Data -Force)) { Copy-Item -LiteralPath $i.FullName -Destination (Join-Path $Dest $i.Name) -Recurse }
    $root = Get-KlbfSavesRoot $Dest
    if ($root -and (Test-Path -LiteralPath (Join-Path $root 'saves'))) { Remove-Item -LiteralPath (Join-Path $root 'saves') -Recurse -Force }
    foreach ($s in (Get-KlbfSaves $root)) {
        if ((Split-Path -Leaf $s.dir) -ne (Split-Path -Leaf $Save.dir)) { Remove-Item -LiteralPath $s.dir -Recurse -Force }
    }
    $Dest
}

# How the card names a save: "#2 <in-game name>" (the Saves tab's number), "Autosave <name>".
# (27/11, public rule) A save by its number only ("#2", "Autosave"): in-game names are the title's mission
# names, never on a card, a screenshot or a post. KLBF_BENCH_SAVE carries this.
function Get-KlbfSaveLabel($Save) {
    if ($Save.auto) { 'Autosave' } else { '#' + ($Save.slot -replace '^Slot ', '') }
}

# "Slot 2" / "Autosave", with the in-game name after it only when $Show (the Saves tab's Show names).
function Get-KlbfSaveName($Save, [bool]$Show = $false) {
    $n = if ($Save.auto) { 'Autosave' } else { $Save.slot }
    if ($Show -and $Save.title) { "$n ($($Save.title))" } else { $n }
}

# The bench window: the guide, the save to load (the most recent by default), a GitHub login box,
# Start bench / Cancel. @{ player; save } or $null.
function Show-KlbfBenchGuide($Saves) {
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    $f = New-Object System.Windows.Forms.Form
    $f.Text = 'KLBF Lab Bench'; Set-KlbfTheme $f; $f.FormBorderStyle = 'FixedDialog'
    $f.MaximizeBox = $false; $f.MinimizeBox = $false; $f.StartPosition = 'CenterParent'
    $f.ClientSize = New-Object System.Drawing.Size(520, 456)
    $t = New-Object System.Windows.Forms.Label; $t.Text = Get-KlbfBenchGuide; $t.SetBounds(16, 12, 488, 300)
    $sl = New-Object System.Windows.Forms.Label; $sl.Text = 'Save the bench loads:'; $sl.SetBounds(16, 358, 200, 20)
    $pick = New-Object System.Windows.Forms.ComboBox; $pick.DropDownStyle = 'DropDownList'; $pick.ForeColor = [System.Drawing.Color]::Black; $pick.SetBounds(216, 355, 288, 24)
    foreach ($s in $Saves) { [void]$pick.Items.Add("$(Get-KlbfSaveName $s) ($($s.date.ToString('yyyy-MM-dd HH:mm')))") }
    if ($Saves) { $pick.SelectedIndex = [array]::IndexOf(@($Saves), (@($Saves) | Sort-Object date -Descending | Select-Object -First 1)) }
    $l = New-Object System.Windows.Forms.Label; $l.Text = 'Your GitHub login (on the card; empty: anonymous):'; $l.SetBounds(16, 322, 300, 20)
    $box = New-Object System.Windows.Forms.TextBox; $box.SetBounds(316, 319, 188, 24)
    $box.BackColor = Get-KlbfColor $script:KlbfTheme.field; $box.ForeColor = Get-KlbfColor $script:KlbfTheme.text
    $go = New-Object System.Windows.Forms.Button; $go.Text = 'Start bench'; $go.DialogResult = 'OK'; $go.SetBounds(296, 404, 100, 34)
    $no = New-Object System.Windows.Forms.Button; $no.Text = 'Cancel'; $no.DialogResult = 'Cancel'; $no.SetBounds(404, 404, 100, 34)
    $go.Enabled = [bool]$Saves
    Set-KlbfButton $go $true; Set-KlbfButton $no
    $f.Controls.AddRange(@($t, $l, $box, $sl, $pick, $go, $no)); $f.AcceptButton = $go; $f.CancelButton = $no
    if ($f.ShowDialog() -eq 'OK') { @{ player = $box.Text; save = @($Saves)[$pick.SelectedIndex] } } else { $null }
}

# The setup window, not shown: @{ form; c (its controls) }. $Hardware: Get-KlbfHardware's; $Info: @{ victory;
# data; windows } (all injected by tests). $ShaderStatus: one line under the tabs.
function New-KlbfMenuForm($Choice, $Hardware, [string]$ShaderStatus = '', $Info = @{}) {
    $Screen = $Hardware.screen
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()
    $f = New-Object System.Windows.Forms.Form
    $f.Text = 'KLBF Lab Settings'
    Set-KlbfTheme $f
    $f.FormBorderStyle = 'FixedDialog'; $f.MaximizeBox = $false; $f.MinimizeBox = $false
    # (27/11) Centred on the chosen display (moved when Display changes), not on the main one
    $f.StartPosition = 'Manual'
    $f.ClientSize = New-Object System.Drawing.Size(520, 906)
    $soft = Get-KlbfColor $script:KlbfTheme.soft
    $back = Get-KlbfColor $script:KlbfTheme.back
    $c = @{ recommended = (Get-KlbfRecommended $Hardware); info = $Info; hw = $Hardware; openUrl = { param($u) Start-Process $u }
        clipboard = { param($t) [System.Windows.Forms.Clipboard]::SetText($t) }; clipboardImage = { param($i) [System.Windows.Forms.Clipboard]::SetImage($i) } }
    $add = { param($parent, $ctl, $x, $y, $w, $h) $ctl.SetBounds($x, $y, $w, $h); $parent.Controls.Add($ctl); $ctl }
    $label = { param($parent, $text, $x, $y, $w, $h = 20) $l = New-Object System.Windows.Forms.Label; $l.Text = $text; & $add $parent $l $x $y $w $h }
    $radio = { param($parent, $text, $x, $y, $w) $r = New-Object System.Windows.Forms.RadioButton; $r.Text = $text; & $add $parent $r $x $y $w 22 }
    $group = { param($parent, $text, $y, $h) $g = New-Object System.Windows.Forms.GroupBox; $g.Text = $text; $g.ForeColor = $soft; & $add $parent $g 6 $y 480 $h }
    # Controls inside a group: white text (the group's caption is soft grey)
    $white = { param($ctl) $ctl.ForeColor = Get-KlbfColor $script:KlbfTheme.text; $ctl }
    $combo = { param($parent, $x, $y, $w)
        # Windows draws a drop-down list in its own light colours: black text on it
        $cb = New-Object System.Windows.Forms.ComboBox; $cb.DropDownStyle = 'DropDownList'; $cb.ForeColor = [System.Drawing.Color]::Black
        & $add $parent $cb $x $y $w 24 }
    $btn = { param($parent, $text, $x, $y, $w, $h = 28) $b = New-Object System.Windows.Forms.Button; $b.Text = $text; Set-KlbfButton $b; & $add $parent $b $x $y $w $h }

    # Header: our logo, the name, the question, the GitHub link
    $ico = Get-KlbfIconPath
    if ($ico) {
        $logo = New-Object System.Windows.Forms.PictureBox
        $logo.Image = Get-KlbfLogo $ico
        $logo.SizeMode = 'Zoom'
        [void](& $add $f $logo 16 12 48 48)
    }
    $c.version = Get-KlbfVersion
    $title = & $label $f 'KLBF Lab' 74 10 200 30
    $title.Font = New-KlbfFont 16 'Bold'
    $sub = & $label $f 'How do you want to play?' 76 40 220 20
    $sub.ForeColor = $soft
    $c.link = New-Object System.Windows.Forms.LinkLabel
    $c.link.Text = 'github.com/PastiiJay/klbf-lab'; $c.link.TextAlign = 'MiddleRight'
    $c.link.LinkColor = $soft; $c.link.ActiveLinkColor = Get-KlbfColor $script:KlbfTheme.text
    [void](& $add $f $c.link 280 40 224 20)
    $rule = New-Object System.Windows.Forms.Panel; $rule.BackColor = Get-KlbfColor $script:KlbfTheme.violet
    [void](& $add $f $rule 16 70 488 2)

    $tabs = New-Object System.Windows.Forms.TabControl
    [void](& $add $f $tabs 12 80 496 694)
    $c.tabs = $tabs
    $page = New-Object System.Windows.Forms.TabPage; $page.Text = 'Play'; $page.BackColor = $back; $page.ForeColor = Get-KlbfColor $script:KlbfTheme.text
    $savesPage = New-Object System.Windows.Forms.TabPage; $savesPage.Text = 'Saves'; $savesPage.BackColor = $back; $savesPage.ForeColor = Get-KlbfColor $script:KlbfTheme.text
    $quickPage = New-Object System.Windows.Forms.TabPage; $quickPage.Text = 'Quick'; $quickPage.BackColor = $back; $quickPage.ForeColor = Get-KlbfColor $script:KlbfTheme.text
    $tabs.TabPages.AddRange(@($page, $savesPage, $quickPage))

    # Your PC: what was found, a verdict per line (green: fine, amber: works with less, red: not supported)
    # Game folder: where the player's copy is; checked, and the room the cache needs on its drive
    $g = & $group $page 'Game folder' 4 96
    # (27/11, privacy) The path is hidden until Show, every time the window opens (screenshots get posted)
    $c.game = New-Object System.Windows.Forms.TextBox; $c.game.ReadOnly = $true
    $c.game.BackColor = Get-KlbfColor $script:KlbfTheme.field; $c.game.ForeColor = Get-KlbfColor $script:KlbfTheme.text
    [void](& $add $g $c.game 12 22 266 24)
    $c.showPath = $false
    $c.show = & $btn $g 'Show' 284 20 72
    $c.browse = & $btn $g "Browse$([char]0x2026)" 362 20 106
    $c.gameDot = & $label $g ([string][char]0x25CF) 12 50 18
    $c.gameText = & $white (& $label $g '' 30 50 444)
    $c.spaceDot = & $label $g ([string][char]0x25CF) 12 70 18
    $c.space = Get-KlbfSpace $Info.cache
    $c.spaceText = & $white (& $label $g ('Needs about {0} GB (cache, logs, backups); {1} GB free on that drive' -f $c.space.needGB, $c.space.freeGB) 30 70 450)
    $c.spaceDot.ForeColor = Get-KlbfColor $(if ($c.space.ok) { 0x22C55E } else { 0xF59E0B })

    $g = & $group $page 'Your PC' 104 140
    $c.pc = @{}
    $y = 20
    foreach ($v in (Get-KlbfVerdicts $Hardware)) {
        $dot = & $label $g ([string][char]0x25CF) 12 $y 18
        $dot.ForeColor = Get-KlbfColor @{ ok = 0x22C55E; warn = 0xF59E0B; bad = 0xEF4444 }[$v.level]
        $c.pc[$v.what] = & $white (& $label $g ('{0}: {1}' -f $v.what, $v.text) 30 $y 444)
        $y += 20
    }
    $c.recommend = & $btn $g 'Try suggested settings for my PC' 12 ($y + 4) 260

    $g = & $group $page 'Profile' 248 52
    # On tested hardware, the suggested options marked (not picked)
    $rec = $c.recommended
    $tested = Test-KlbfTested $Hardware
    $mark = { param($text, $on) if ($tested -and $on) { "$text (recommended for your hardware)" } else { $text } }
    $c.profX = & $white (& $radio $g (& $mark 'Profile X' ($rec.profile -eq 'X')) 12 20 330)
    $c.profS = & $white (& $radio $g (& $mark 'Profile S' ($rec.profile -eq 'S')) 346 20 130)

    $g = & $group $page 'Upscaler' 304 118
    $c.upFsr = & $white (& $radio $g (& $mark (Get-KlbfUpscalerName $Hardware $Info.cache) ($rec.upscaler -eq 'fsr')) 12 20 460)
    $c.upTitle = & $white (& $radio $g 'Built-in (console) upscaler' 12 42 460)
    $c.tips = New-Object System.Windows.Forms.ToolTip
    $c.tips.SetToolTip($c.upTitle, "The game's own upscaler, translated: works on every card; softer image, no frame generation.")
    $c.tips.SetToolTip($c.upFsr, 'AMD FSR (FSR 4 on RX 9000, FSR 3.1 elsewhere): sharper, presets, frame generation.')
    $c.presetText = if ($tested) { "Preset (recommended for your hardware: $($script:KlbfPresets[$rec.preset]))" } else { 'Preset' }
    $c.presetLabel = & $white (& $label $g "$($c.presetText):" 12 68 460)
    $c.upscalerGroup = $g
    $p = New-Object System.Windows.Forms.Panel
    [void](& $add $g $p 8 86 466 26)
    $c.presets = @{}
    $at = @{ quality = 4, 92; balanced = 100, 96; performance = 200, 118; ultraperformance = 322, 140 }
    foreach ($k in $script:KlbfPresets.Keys) {
        $c.presets[$k] = & $white (& $radio $p $script:KlbfPresets[$k] $at[$k][0] 2 $at[$k][1])
    }

    $g = & $group $page 'Output resolution, display, mode' 426 90
    $c.output = & $combo $g 12 22 112
    foreach ($m in $script:KlbfModes) { [void]$c.output.Items.Add($m) }
    $c.screens = @(if ($null -ne $Info.screens) { $Info.screens } else { Get-KlbfScreens })
    $c.display = & $combo $g 130 22 192
    # (27/11) Window or borderless fullscreen on the chosen display (Alt+Enter switches in the game)
    $c.mode = & $combo $g 328 22 146
    [void]$c.mode.Items.Add('Window'); [void]$c.mode.Items.Add('Borderless fullscreen')
    $c.mode.SelectedIndex = [int][bool]$Choice.borderless
    $c.tips.SetToolTip($c.mode, 'Borderless fullscreen covers the chosen display, the picture kept in shape. Alt+Enter switches in the game.')
    foreach ($n in (Get-KlbfDisplayNames $c.screens)) { [void]$c.display.Items.Add($n) }
    $c.primaryIndex = Get-KlbfDisplayIndex $null $c.screens
    if ($c.display.Items.Count) { $c.display.SelectedIndex = Get-KlbfDisplayIndex $Choice $c.screens }
    $c.tips.SetToolTip($c.display, 'The monitor the game window (and this window) opens on, centred.')
    $c.outputRec = if ($tested -and $rec.profile -eq 'X') { "Recommended for your hardware: $($rec.output)" } else { '' }
    $c.outputNote = & $white (& $label $g '' 12 50 460 36)

    $g = & $group $page 'Options' 520 64
    $c.fg = New-Object System.Windows.Forms.CheckBox
    $c.fgText = & $mark 'Frame generation (FG x2)' $rec.frameGen
    $c.fg.Text = $c.fgText
    [void](& $white (& $add $g $c.fg 12 16 460 22))
    $c.optionsGroup = $g
    $c.adv = New-Object System.Windows.Forms.CheckBox
    $c.adv.Text = 'Advanced: memory hooks and texture trim (faster, uses less RAM)'
    [void](& $white (& $add $g $c.adv 12 38 462 22))

    $g = & $group $page 'Saved profiles' 586 60
    # (27/11) Save overwrites the selected profile, Save as... makes a new one; the loaded one shows
    # "name *" once the settings differ from it.
    $c.saved = & $combo $g 12 24 150
    foreach ($n in (Get-KlbfPresetNames)) { [void]$c.saved.Items.Add($n) }
    if ($c.saved.Items.Count) { $c.saved.SelectedIndex = 0 }
    $c.load = & $btn $g 'Load' 168 22 70
    $c.save = & $btn $g 'Save' 244 22 70
    $c.saveAs = & $btn $g "Save as$([char]0x2026)" 320 22 80
    $c.delete = & $btn $g 'Delete' 406 22 62
    $c.save.Enabled = $null -ne $c.saved.SelectedItem

    # Saves tab
    $c.savesRoot = Get-KlbfSavesRoot $Info.data
    $c.savesList = New-Object System.Windows.Forms.ListView
    $c.savesList.View = 'Details'; $c.savesList.FullRowSelect = $true; $c.savesList.HeaderStyle = 'Nonclickable'
    $c.savesList.BackColor = Get-KlbfColor $script:KlbfTheme.field; $c.savesList.ForeColor = Get-KlbfColor $script:KlbfTheme.text
    # (27/11) In-game names (the title's mission names) hidden until Show names, every time the window opens
    foreach ($h in @(@('Slot', 90), @('Name in the game', 190), @('Date', 120), @('Size', 70))) { [void]$c.savesList.Columns.Add($h[0], $h[1]) }
    $c.showNames = $false
    [void](& $add $savesPage $c.savesList 6 8 480 250)
    $note = & $label $savesPage ("Numbered as the game's Load menu lists them: the autosave first, then the newest first. " +
        'All saves are backed up before every launch and every bench (the last 10 are kept).') 6 264 480 54
    $note.ForeColor = $soft
    $c.backup = & $btn $savesPage 'Back up all' 6 322 150
    $c.restore = & $btn $savesPage "Restore a backup$([char]0x2026)" 166 322 170
    $c.showNamesBtn = & $btn $savesPage "Show names" 346 322 140
    $g = & $group $savesPage 'Copy a save over another (all saves are backed up first)' 362 64
    $c.copyFrom = & $combo $g 12 26 170
    [void](& $white (& $label $g 'over' 190 30 36))
    $c.copyTo = & $combo $g 230 26 140
    $c.copy = & $btn $g 'Copy' 380 24 88
    $c.savesNote = & $white (& $label $savesPage '' 6 436 480 60)

    # Quick tab: the last bench card, to copy as an image or as a short text
    # (27/11) Heights from the content, top to bottom (nothing overlaps whatever the number of shortcut lines)
    $g = & $group $quickPage 'Last benchmark' 4 226
    $c.benchPic = New-Object System.Windows.Forms.PictureBox; $c.benchPic.SizeMode = 'Zoom'
    [void](& $add $g $c.benchPic 12 22 456 140)
    $c.benchInfo = & $white (& $label $g '' 12 166 456 20)
    $c.copyImage = & $btn $g 'Copy image' 12 188 140
    $c.copyText = & $btn $g 'Copy text' 158 188 140
    $c.quickNote = & $white (& $label $g '' 304 186 164 34)
    # (27/11) Shortcuts: every key, and the in-game actions here (the game is not running: Restart is Play
    # with the settings this window opened with; Settings is this window)
    $lines = @(Get-KlbfShortcuts)
    $g = & $group $quickPage 'Shortcuts' 236 (20 + 18 * $lines.Count + 4 + 28 + 10)
    $y = 20
    foreach ($line in $lines) { [void](& $white (& $label $g $line 12 $y 460 18)); $y += 18 }
    $c.quickRestart = & $btn $g 'Restart (F11)' 12 ($y + 4) 130
    $c.quickSettings = & $btn $g 'Settings (F10)' 147 ($y + 4) 130
    $c.shortcut = & $btn $g 'Create desktop shortcut' 282 ($y + 4) 186
    $c.tips.SetToolTip($c.quickRestart, 'Play with the settings this window opened with.')
    $g = & $group $quickPage 'Screenshots (F8)' (236 + $g.Height + 6) 84
    $c.shots = New-Object System.Windows.Forms.TextBox; $c.shots.ReadOnly = $true
    $c.shots.BackColor = Get-KlbfColor $script:KlbfTheme.field; $c.shots.ForeColor = Get-KlbfColor $script:KlbfTheme.text
    [void](& $white (& $label $g 'Folder:' 12 25 50))
    [void](& $add $g $c.shots 64 22 214 24)
    $c.showShots = $false
    $c.shotsShow = & $btn $g 'Show' 284 20 72
    $c.shotsChange = & $btn $g "Change$([char]0x2026)" 362 20 106
    $c.shotsOpen = & $btn $g 'Open folder' 12 50 150
    $c.shotsNote = & $white (& $label $g '' 172 54 300)
    Set-KlbfShotsPath $c $Choice.shotsDir
    $c.lastBench = Get-KlbfLastBench $Info.victory
    if ($c.lastBench) {
        $c.benchPic.Image = [System.Drawing.Image]::FromStream((New-Object IO.MemoryStream(, [IO.File]::ReadAllBytes($c.lastBench.png))))
        $c.benchInfo.Text = 'The newest card from the bench-results folder.'
    } else {
        $c.benchInfo.Text = 'No benchmark yet: Run benchmark makes one.'
        $c.copyImage.Enabled = $false; $c.copyText.Enabled = $false
    }

    # Under the tabs: the shaders' line, then a 3-column grid (156 px columns, 10 px apart): grey buttons on
    # the first row (room for more), Play alone in the right column of the second, violet.
    $c.status = & $label $f $ShaderStatus 16 780 276 22
    $c.status.ForeColor = $soft
    # (27/11) Clear shader cache, with its size (no path)
    $c.cacheBytes = Get-KlbfCacheBytes $Info.cache; $c.cacheGB = if ($c.cacheBytes -gt 0) { [math]::Max(0.1, [math]::Round($c.cacheBytes / 1GB, 1)) } else { 0 }
    $c.clearCache = & $btn $f "Clear shader cache ($($c.cacheGB) GB)" 298 776 206 26
    $c.clearCache.Enabled = $c.cacheGB -gt 0
    $col = { param($i) 16 + 166 * $i }
    $c.prepare = & $btn $f 'Prepare shaders' (& $col 0) 806 156 36
    $c.bench = & $btn $f 'Run benchmark' (& $col 1) 806 156 36
    $c.bug = & $btn $f 'Report a bug' (& $col 2) 806 156 36
    $c.bugNote = & $label $f '' (& $col 0) 848 322 35
    $c.bugNote.ForeColor = $soft
    # (27/11) Bottom left, opposite Play: the version (small, grey), and the update line when Show-KlbfMenu's
    # check finds a newer release (Set-KlbfUpdateLine)
    $c.versionLabel = & $label $f $(if ($c.version) { $c.version } else { 'dev' }) 16 884 56 18
    $c.versionLabel.ForeColor = $soft; $c.versionLabel.Font = New-KlbfFont 8
    $c.update = New-Object System.Windows.Forms.LinkLabel; $c.update.Visible = $false; $c.update.Font = New-KlbfFont 8
    $c.update.ForeColor = Get-KlbfColor $script:KlbfTheme.text; $c.update.LinkColor = Get-KlbfColor 0xA78BFA
    $c.update.ActiveLinkColor = Get-KlbfColor $script:KlbfTheme.text
    [void](& $add $f $c.update 72 884 274 18)
    $c.play = & $btn $f 'Play' (& $col 2) 850 156 44
    Set-KlbfButton $c.play $true
    $c.play.Font = New-KlbfFont 12 'Bold'

    $f.AcceptButton = $c.play

    $c.form = $f
    Set-KlbfGamePath $c $(if ($Choice.game) { $Choice.game } else { [string]$Info.game })
    Set-KlbfFormChoice $c $Choice
    Update-KlbfSavesTab $c
    # A closure sees variables, not this file's functions: they go in as variables.
    $fUpdate = ${function:Update-KlbfForm}; $fSet = ${function:Set-KlbfFormChoice}; $fGet = ${function:Get-KlbfFormChoice}
    $fPath = ${function:Get-KlbfPresetPath}; $fRead = ${function:Read-KlbfSettings}; $fSave = ${function:Save-KlbfSettings}
    $fSaves = ${function:Update-KlbfSavesTab}; $fBackup = ${function:Backup-KlbfSaves}; $fRestore = ${function:Restore-KlbfSaves}
    $fCopy = ${function:Copy-KlbfSave}; $fSaveName = ${function:Get-KlbfSaveName}; $fReport = ${function:Get-KlbfBugReport}; $fLog = ${function:Get-KlbfLatestLog}
    $fGamePath = ${function:Set-KlbfGamePath}; $fGuide = ${function:Show-KlbfBenchGuide}; $repo = $script:KlbfRepo; $backupDir = Get-KlbfBackupDir
    $update = { & $fUpdate $c }.GetNewClosure()
    foreach ($r in @($c.profX, $c.profS, $c.upFsr, $c.upTitle)) { $r.add_CheckedChanged($update) }
    $fMark = ${function:Update-KlbfProfileMark}; $fKey = ${function:Get-KlbfProfileKey}
    $markIt = { & $fMark $c }.GetNewClosure()
    foreach ($r in @($c.profX, $c.profS, $c.upFsr, $c.upTitle, $c.fg, $c.adv) + @($c.presets.Values)) { $r.add_CheckedChanged($markIt) }
    $c.output.add_SelectedIndexChanged($markIt)
    $c.output.add_SelectedIndexChanged($update)
    $fCentre = ${function:Get-KlbfCentre}
    $c.place = { if ($c.display.SelectedIndex -ge 0) { $c.form.Location = New-Object System.Drawing.Point(@(& $fCentre $c.screens[$c.display.SelectedIndex].work @($c.form.Width, $c.form.Height))) } }.GetNewClosure()
    $c.display.add_SelectedIndexChanged({ & $c.place; & $markIt; & $fUpdate $c }.GetNewClosure())
    $c.saved.add_SelectedIndexChanged({ $c.save.Enabled = $null -ne $c.saved.SelectedItem }.GetNewClosure())
    $c.load.add_Click({
        if (-not $c.saved.SelectedItem) { return }
        $name = $c.saved.SelectedItem -replace ' \*$', ''
        $c.loaded = $null
        & $fSet $c (& $fRead (& $fPath $name))
        $c.loaded = $name; $c.loadedKey = & $fKey (& $fGet $c)
        & $fMark $c
    }.GetNewClosure())
    $c.save.add_Click({
        if (-not $c.saved.SelectedItem) { return }
        $name = $c.saved.SelectedItem -replace ' \*$', ''
        $ask = if ($c.confirm) { $c.confirm } else { { param($t) [System.Windows.Forms.MessageBox]::Show($t, 'KLBF Lab', 'YesNo', 'Question') -eq 'Yes' } }
        if (-not (& $ask "Replace '$name' with the current settings?")) { return }
        & $fSave (& $fGet $c) (& $fPath $name)
        $c.loaded = $name; $c.loadedKey = & $fKey (& $fGet $c)
        & $fMark $c
    }.GetNewClosure())
    $c.saveAs.add_Click({
        Add-Type -AssemblyName Microsoft.VisualBasic
        $name = [Microsoft.VisualBasic.Interaction]::InputBox('Name of this profile:', 'KLBF Lab', '')
        $path = & $fPath $name
        if (-not $path) { return }
        & $fSave (& $fGet $c) $path
        $base = [IO.Path]::GetFileNameWithoutExtension($path)
        if ($c.loaded) { $c.loaded = $null; & $fMark $c }
        if (-not $c.saved.Items.Contains($base)) { [void]$c.saved.Items.Add($base) }
        $c.saved.SelectedItem = $base
        $c.loaded = $base; $c.loadedKey = & $fKey (& $fGet $c)
    }.GetNewClosure())
    $c.delete.add_Click({
        if (-not $c.saved.SelectedItem) { return }
        $name = $c.saved.SelectedItem -replace ' \*$', ''
        if ($c.loaded -eq $name) { $c.loaded = $null }
        $path = & $fPath $name
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
        $c.saved.Items.Remove($c.saved.SelectedItem)
        if ($c.saved.Items.Count) { $c.saved.SelectedIndex = 0 }
        $c.save.Enabled = $null -ne $c.saved.SelectedItem
    }.GetNewClosure())
    $c.recommend.add_Click({ & $fSet $c $c.recommended }.GetNewClosure())
    $c.link.add_LinkClicked({ & $c.openUrl $repo }.GetNewClosure())
    $c.bug.add_Click({
        $text = & $fReport (& $fGet $c) $c.hw $c.info.victory (& $fLog $c.info.data) $c.info.windows
        $c.lastReport = $text
        try { & $c.clipboard $text } catch {}
        $c.bugNote.Text = 'System info copied: paste it in the issue and attach the log.'
        & $c.openUrl "$repo/issues/new"
    }.GetNewClosure())
    $c.bench.add_Click({
        $r = if ($c.benchGuide) { & $c.benchGuide $c.saves } else { & $fGuide $c.saves }
        if ($null -eq $r) { return }
        $c.benchPlayer = $r.player; $c.benchSave = $r.save; $c.form.Tag = 'bench'; $c.form.Close()
    }.GetNewClosure())
    $c.browse.add_Click({
        $d = New-Object System.Windows.Forms.FolderBrowserDialog
        $d.Description = 'The folder of your copy of the game (the one with game_scarlett_master.exe)'
        if ($c.gamePath) { $d.SelectedPath = $c.gamePath }
        $picked = if ($c.pickFolder) { & $c.pickFolder } elseif ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }
        if ($picked) { & $fGamePath $c $picked; & $fUpdate $c }
    }.GetNewClosure())
    $c.show.add_Click({ $c.showPath = -not $c.showPath; & $fGamePath $c $c.gamePath }.GetNewClosure())
    $c.backup.add_Click({
        $z = & $fBackup $c.savesRoot
        $c.savesNote.Text = $(if ($z) { 'Backed up (KLBF-Lab save-backups folder).' } else { 'No saves yet.' })
    }.GetNewClosure())
    $c.restore.add_Click({
        $d = New-Object System.Windows.Forms.OpenFileDialog
        $d.InitialDirectory = $backupDir
        $d.Filter = 'Save backups (*.zip)|*.zip'
        if ($d.ShowDialog() -ne 'OK') { return }
        $ok = [System.Windows.Forms.MessageBox]::Show("Replace all your saves with this backup?`r`nYour current saves are backed up first.", 'KLBF Lab', 'YesNo', 'Question')
        if ($ok -ne 'Yes') { return }
        $z = & $fRestore $d.FileName $c.savesRoot
        & $fSaves $c
        $c.savesNote.Text = 'Restored. The saves before it are backed up too.' 
    }.GetNewClosure())
    $c.copy.add_Click({
        $from = if ($c.copyFrom.SelectedIndex -ge 0) { $c.saves[$c.copyFrom.SelectedIndex] }
        $to = if ($c.copyTo.SelectedIndex -ge 0) { $c.saves[$c.copyTo.SelectedIndex] }
        if (-not $from -or -not $to -or $from.dir -eq $to.dir) { $c.savesNote.Text = 'Pick two different saves.'; return }
        $ask = if ($c.confirm) { $c.confirm } else { { param($t) [System.Windows.Forms.MessageBox]::Show($t, 'KLBF Lab', 'YesNo', 'Warning') -eq 'Yes' } }
        if (-not (& $ask "Copy $(& $fSaveName $from $c.showNames) over $(& $fSaveName $to $c.showNames)?`r`nAll saves are backed up first.")) { return }
        $z = & $fCopy $from $to $c.savesRoot
        & $fSaves $c
        $c.savesNote.Text = 'Copied. The saves before it are backed up.' 
    }.GetNewClosure())
    $fShare = ${function:Get-KlbfBenchShareText}
    $c.showNamesBtn.add_Click({ $c.showNames = -not $c.showNames; & $fSaves $c }.GetNewClosure())
    $c.copyImage.add_Click({
        try { & $c.clipboardImage $c.benchPic.Image; $c.quickNote.Text = 'Card copied: paste it anywhere.' } catch { $c.quickNote.Text = 'Could not copy the image.' }
    }.GetNewClosure())
    $c.copyText.add_Click({
        if (-not $c.lastBench.json) { $c.quickNote.Text = 'No result file beside this card.'; return }
        $text = & $fShare (Get-Content -LiteralPath $c.lastBench.json -Raw | ConvertFrom-Json)
        $c.lastShare = $text
        try { & $c.clipboard $text; $c.quickNote.Text = 'Text copied: paste it anywhere.' } catch { $c.quickNote.Text = 'Could not copy the text.' }
    }.GetNewClosure())
    $c.play.add_Click({ $c.form.Tag = 'play'; $c.form.Close() }.GetNewClosure())
    $opened = $Choice; $fShots = ${function:Set-KlbfShotsPath}; $fShotsDir = ${function:Get-KlbfShotsDir}
    $c.quickRestart.add_Click({ & $fSet $c $opened; $c.form.Tag = 'play'; $c.form.Close() }.GetNewClosure())
    $c.quickSettings.add_Click({ $c.tabs.SelectedIndex = 0 }.GetNewClosure())
    $fShortcut = ${function:Set-KlbfShortcut}; $fSpec = ${function:Get-KlbfShortcutSpec}; $here = $PSScriptRoot
    $c.shortcut.add_Click({
        $ask = if ($c.confirm) { $c.confirm } else { { param($t) [System.Windows.Forms.MessageBox]::Show($t, 'KLBF Lab', 'YesNo', 'Question') -eq 'Yes' } }
        $spec = & $fSpec $here $c.info.victory
        $done = if ($c.shortcutPath) { & $fShortcut $spec $ask $c.shortcutPath } else { & $fShortcut $spec $ask }
        if ($done) { $c.shotsNote.Text = 'Desktop shortcut made.' }
    }.GetNewClosure())
    $c.shotsShow.add_Click({ $c.showShots = -not $c.showShots; & $fShots $c $c.shotsDir }.GetNewClosure())
    $c.shotsChange.add_Click({
        $d = New-Object System.Windows.Forms.FolderBrowserDialog
        $d.Description = 'The folder for screenshots (F8 in the game)'
        $d.SelectedPath = & $fShotsDir @{ shotsDir = $c.shotsDir }
        $picked = if ($c.pickShotsFolder) { & $c.pickShotsFolder } elseif ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }
        if ($picked) { & $fShots $c $picked; $c.shotsNote.Text = 'Saved when you press Play.' }
    }.GetNewClosure())
    $c.shotsOpen.add_Click({
        $dir = & $fShotsDir @{ shotsDir = $c.shotsDir }
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        & $c.openUrl $dir
    }.GetNewClosure())
    $fPrepSpace = ${function:Test-KlbfPrepareSpace}; $fClear = ${function:Clear-KlbfShaderCache}
    $c.tips.SetToolTip($c.prepare, 'About 7.2 GB once done. It grows again only after a change of ray tracing or probe rays; none of the choices here change the shaders.')
    $c.prepare.add_Click({
        $sp = & $fPrepSpace $c.info.cache $c.freeGB
        $ask = if ($c.confirm) { $c.confirm } else { { param($t) [System.Windows.Forms.MessageBox]::Show($t, 'KLBF Lab', 'YesNo', 'Warning') -eq 'Yes' } }
        if (-not $sp.ok -and -not (& $ask "Only $($sp.freeGB) GB free on the cache's drive; the shaders need about $($sp.needGB) GB more.`r`nPrepare anyway?")) { return }
        $c.form.Tag = 'prepare'; $c.form.Close()
    }.GetNewClosure())
    $c.clearCache.add_Click({
        $ask = if ($c.confirm) { $c.confirm } else { { param($t) [System.Windows.Forms.MessageBox]::Show($t, 'KLBF Lab', 'YesNo', 'Warning') -eq 'Yes' } }
        if (-not (& $ask "Delete the shader cache ($($c.cacheGB) GB)?`r`nThe shaders are prepared again before the next start (a few minutes).")) { return }
        try { & $fClear $c.info.cache; $c.cacheGB = 0; $c.clearCache.Text = 'Clear shader cache (0 GB)'; $c.clearCache.Enabled = $false; $c.status.Text = 'Shaders: not prepared yet (Play offers it).' }
        catch { $c.status.Text = "$_" }
    }.GetNewClosure())
    if ($c.screens.Count) { & $c.place } else { $f.StartPosition = 'CenterScreen' }
    @{ form = $f; c = $c }
}

# The Saves tab from the saves folder.
function Update-KlbfSavesTab($c) {
    $c.saves = @(Get-KlbfSaves $c.savesRoot)
    $c.savesList.Items.Clear(); $c.copyFrom.Items.Clear(); $c.copyTo.Items.Clear()
    foreach ($s in $c.saves) {
        $it = New-Object System.Windows.Forms.ListViewItem((Get-KlbfSaveName $s))
        [void]$it.SubItems.Add($(if ($c.showNames) { $s.title } else { ([string][char]0x2022) * 8 + ' (hidden)' })); [void]$it.SubItems.Add($s.date.ToString('yyyy-MM-dd HH:mm')); [void]$it.SubItems.Add('{0:N0} KB' -f ($s.size / 1KB))
        [void]$c.savesList.Items.Add($it)
        [void]$c.copyFrom.Items.Add((Get-KlbfSaveName $s)); [void]$c.copyTo.Items.Add((Get-KlbfSaveName $s))
    }
    $c.showNamesBtn.Text = "$(if ($c.showNames) { 'Hide names' } else { 'Show names' })" 
    if (-not $c.saves) { $c.savesNote.Text = $(if ($c.savesRoot) { 'No saves yet.' } else { 'No saves folder yet: it appears after the first save in the game.' }) }
}

# The game folder kept in $c.gamePath; the box shows it only after Show ("(hidden)" otherwise).
function Set-KlbfGamePath($c, [string]$Path) {
    $c.gamePath = $Path
    $c.game.Text = if (-not $Path) { '' } elseif ($c.showPath) { $Path } else { ([string][char]0x2022) * 8 + ' (hidden)' }
    $c.show.Text = if ($c.showPath) { 'Hide' } else { 'Show' }
}

# The screenshot folder kept in $c.shotsDir ('' = Pictures\KLBF Lab); the box shows it only after Show.
function Set-KlbfShotsPath($c, [string]$Path) {
    $c.shotsDir = $Path
    $c.shots.Text = if ($c.showShots) { Get-KlbfShotsDir @{ shotsDir = $Path } } else { ([string][char]0x2022) * 8 + ' (hidden)' }
    $c.shotsShow.Text = if ($c.showShots) { 'Hide' } else { 'Show' }
}

# What a saved profile holds of a choice, to compare (the game and screenshot folders apart).
function Get-KlbfProfileKey($Choice) {
    ($Choice.profile, $Choice.upscaler, $Choice.preset, $Choice.output, $Choice.frameGen, $Choice.advanced) -join '|'
}

# The loaded profile's entry in the list: "name", or "name *" when the settings differ from it.
function Update-KlbfProfileMark($c) {
    for ($i = 0; $i -lt $c.saved.Items.Count; $i++) {
        $plain = [string]$c.saved.Items[$i] -replace ' \*$', ''
        $want = if ($c.loaded -eq $plain -and (Get-KlbfProfileKey (Get-KlbfFormChoice $c)) -ne $c.loadedKey) { "$plain *" } else { $plain }
        if ([string]$c.saved.Items[$i] -ne $want) { $sel = $c.saved.SelectedIndex; $c.saved.Items[$i] = $want; $c.saved.SelectedIndex = -1; $c.saved.SelectedIndex = $sel }
    }
}

function Set-KlbfFormChoice($c, $Choice) {
    if ($Choice.game) { Set-KlbfGamePath $c $Choice.game }
    $c.profX.Checked = $Choice.profile -ne 'S'; $c.profS.Checked = $Choice.profile -eq 'S'
    $c.upFsr.Checked = $Choice.upscaler -ne 'title'; $c.upTitle.Checked = $Choice.upscaler -eq 'title'
    $c.presets[$Choice.preset].Checked = $true
    $c.output.SelectedIndex = $c.output.Items.IndexOf($Choice.output)
    $c.fg.Checked = $Choice.frameGen; $c.adv.Checked = $Choice.advanced
    Update-KlbfForm $c
}

function Get-KlbfFormChoice($c) {
    ConvertTo-KlbfChoice ([pscustomobject]@{
        profile = $(if ($c.profS.Checked) { 'S' } else { 'X' })
        upscaler = $(if ($c.upTitle.Checked) { 'title' } else { 'fsr' })
        preset = @($c.presets.Keys | Where-Object { $c.presets[$_].Checked })[0]
        output = [string]$c.output.SelectedItem
        frameGen = [bool]$c.fg.Checked
        advanced = [bool]$c.adv.Checked
        game = [string]$c.gamePath
        shotsDir = [string]$c.shotsDir
        borderless = [bool]($c.mode.SelectedIndex -eq 1)
        display = $(if ($c.display.SelectedIndex -lt 0 -or $c.display.SelectedIndex -eq $c.primaryIndex) { -1 } else { [int]$c.display.SelectedIndex })
    })
}

# (27/11) The line under Output: why 3840x2160 on a smaller screen, the HUD issue below it.
function Get-KlbfOutputNote([bool]$ProfileS, [string]$Output, [string]$Screen) {
    if ($ProfileS) { return 'Profile S: the title sets its own render size and output.' }
    if ($Output -ne '3840x2160') { return "Below 3840x2160 the game's HUD gets bigger (known issue)." }
    $sw, $sh = ($Screen -split 'x') + @(0, 0) | Select-Object -First 2 | ForEach-Object { [int]$_ }
    if ($Screen -and $sh -lt 2160) { return "3840x2160 recommended: the game's HUD is only the right size at 4K; scaled down to your $Screen screen (sharper; cost set by the FSR preset)." }
    '3840x2160 recommended: the size the game''s HUD is made for.'
}

# What applies to what: Profile S has the title's FSR, render size and output; frame generation needs FSR.
function Update-KlbfForm($c) {
    $s = $c.profS.Checked
    $fsr = $s -or $c.upFsr.Checked
    $c.upFsr.Enabled = -not $s; $c.upTitle.Enabled = -not $s
    foreach ($k in $c.presets.Keys) { $c.presets[$k].Enabled = -not $s -and $fsr }
    $c.output.Enabled = -not $s
    $c.outputNote.Text = Get-KlbfOutputNote $s ([string]$c.output.SelectedItem) $(if ($c.display.SelectedIndex -ge 0) { $m = $c.screens[$c.display.SelectedIndex]; if ($m.mode) { $m.mode } else { $m.bounds } } else { '' })
    $c.fg.Enabled = $fsr
    # (27/11) A control greyed by a rule says why, beside it and in a tooltip (a greyed control shows no
    # tooltip of its own: its group's says it too).
    $why = @{ fg = $(if (-not $fsr) { 'needs AMD FSR' } else { '' }); preset = $(if ($s) { 'set by the game in Profile S' } elseif (-not $fsr) { 'needs AMD FSR' } else { '' }) }
    $c.fg.Text = if ($why.fg) { "Frame generation (FG x2) - $($why.fg)" } else { $c.fgText }
    $c.presetLabel.Text = if ($why.preset) { "$($c.presetText) - $($why.preset):" } else { "$($c.presetText):" }
    $c.tips.SetToolTip($c.optionsGroup, $(if ($why.fg) { 'Frame generation is fed by the AMD FSR upscale: pick AMD FSR to use it.' } else { '' }))
    $c.tips.SetToolTip($c.upscalerGroup, $(if ($s) { 'Profile S: the game sets its own upscaler, render size and output.' } elseif ($why.preset) { 'The presets are AMD FSR''s: pick AMD FSR to choose one.' } else { '' }))
    $c.tips.SetToolTip($c.output, $(if ($s) { 'Set by the game in Profile S.' } else { '' }))
    # Play, bench and shaders need the game files
    $ok = Test-KlbfGameFolder $c.gamePath
    $c.gameDot.ForeColor = Get-KlbfColor $(if ($ok) { 0x22C55E } else { 0xEF4444 })
    $c.gameText.Text = if ($ok) { 'Game files found' } elseif ($c.gamePath) { 'This folder does not contain the game files' } else { 'Choose the folder where your copy of the game is installed (Browse...)' }
    foreach ($b in $c.play, $c.bench, $c.prepare) { $b.Enabled = $ok }
}

# Shows the window: @{ action = 'play'|'prepare'|'bench'; choice; benchPlayer; benchSave } or $null if closed.
function Show-KlbfMenu($Choice, [string]$ShaderStatus = '', [string]$Victory = '', [string]$Data = '', [string]$Game = '', [string]$Cache = '', [int]$Tab = 0) {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    $info = @{ victory = $Victory; data = $Data; game = $Game; cache = $Cache; windows = $(if ($os) { "$($os.Caption) (build $($os.BuildNumber))" } else { '' }) }
    $m = New-KlbfMenuForm $Choice (Get-KlbfHardware) $ShaderStatus $info
    $m.c.tabs.SelectedIndex = $Tab
    # (27/11) Just updated (the older launcher set it before starting these new files)
    if ($env:KLBF_UPDATED) { $ShaderStatus = $env:KLBF_UPDATED; $m.c.status.Text = $ShaderStatus; $env:KLBF_UPDATED = $null }
    # (27/11) Updates: GitHub's latest release asked in a runspace (never blocking; offline or rate-limited:
    # nothing shown); a newer one puts the update line in the header. Installed only on its Update link.
    $dir = if ($Victory) { Split-Path -Parent $Victory } else { '' }
    if ($dir -and $m.c.version) {
        $defs = (('Get-KlbfVersion', 'Get-KlbfUpdate', 'Read-KlbfSums', 'Test-KlbfSums', 'Test-KlbfRunning', 'Switch-KlbfBuild', 'Install-KlbfUpdate') |
            ForEach-Object { "function $_ {$((Get-Item "function:$_").ScriptBlock)}" }) -join "`n"
        $run = { param($script, $a, $b) $p = [powershell]::Create(); $null = $p.AddScript("param(`$a, `$b)`n$defs`n$script").AddArgument($a).AddArgument($b); @{ ps = $p; job = $p.BeginInvoke() } }.GetNewClosure()
        $c = $m.c; $fLine = ${function:Set-KlbfUpdateLine}; $fGameOk = ${function:Test-KlbfGameFolder}
        # Never closed while the files are being replaced
        $c.form.add_FormClosing({ param($s, $e) if ($c.updating -and $c.form.Tag -ne 'updated') { $e.Cancel = $true } }.GetNewClosure())
        $check = & $run 'Get-KlbfUpdate $a' $c.version $null
        $uTick = New-Object System.Windows.Forms.Timer; $uTick.Interval = 500
        $uTick.add_Tick({
            if (-not $check.job.IsCompleted) { return }
            $uTick.Stop()
            $u = try { @($check.ps.EndInvoke($check.job))[0] } catch { $null }
            if ($u -and $u.tag) { & $fLine $c $u }
        }.GetNewClosure())
        $uTick.Start()
        $c.update.add_LinkClicked({
            param($s, $e)
            $u = $c.updateInfo
            if ($e.Link.LinkData -eq 'notes') { & $c.openUrl $u.page; return }
            $ask = if ($c.confirm) { $c.confirm } else { { param($t) [System.Windows.Forms.MessageBox]::Show($t, 'KLBF Lab', 'YesNo', 'Question') -eq 'Yes' } }
            if (-not (& $ask "Update KLBF Lab to $($u.tag)?`r`nThe zip is downloaded from GitHub and checked; your saves and settings are not touched. This window then opens again.")) { return }
            $c.update.Enabled = $false; $c.play.Enabled = $false; $c.updating = $true
            $c.status.Text = "Downloading $($u.tag)$([char]0x2026)"
            $inst = & $run 'Install-KlbfUpdate $a $b' $u $dir
            $iTick = New-Object System.Windows.Forms.Timer; $iTick.Interval = 500
            $iTick.add_Tick({
                if (-not $inst.job.IsCompleted) { return }
                $iTick.Stop(); $c.updating = $false
                $r = try { @($inst.ps.EndInvoke($inst.job))[0] } catch { @{ ok = $false; message = "Update not installed: $_" } }
                if ($r.ok) { $env:KLBF_UPDATED = $r.message; $c.form.Tag = 'updated'; $c.form.Close(); return }
                $c.status.Text = $r.message; $c.update.Enabled = $true; $c.play.Enabled = & $fGameOk $c.gamePath
                [void][System.Windows.Forms.MessageBox]::Show($r.message, 'KLBF Lab', 'OK', 'Warning')
            }.GetNewClosure())
            $iTick.Start()
        }.GetNewClosure())
    }
    # (27/11) A first start from the release zip: the official components fetched while the window is open,
    # on the status line, in a runspace of this process (no window). Play waits for nothing: play.ps1 checks.
    $dir = if ($Victory) { Split-Path -Parent $Victory } else { '' }
    $need = @(if ($dir) { 'agility', 'ffx' | Where-Object { -not (Install-KlbfComponent $_ $dir -CheckOnly) } })
    if ($need) {
        $ps = [powershell]::Create()
        $null = $ps.AddScript("param(`$Need, `$Dir)`nfunction Install-KlbfComponent {${function:Install-KlbfComponent}}`n" +
            '$r = @{}; foreach ($n in $Need) { $r[$n] = Install-KlbfComponent $n $Dir }; $r').AddArgument($need).AddArgument($dir)
        $job = $ps.BeginInvoke()
        $st = $m.c.status
        $st.Text = if ($need -contains 'ffx') { "Downloading AMD FSR (official, from AMD's GitHub)$([char]0x2026)" } else { "Downloading DirectX 12 (official, from Microsoft)$([char]0x2026)" }
        $tick = New-Object System.Windows.Forms.Timer; $tick.Interval = 500
        $tick.add_Tick({
            if (-not $job.IsCompleted) { return }
            $tick.Stop()
            $r = try { @($ps.EndInvoke($job))[0] } catch { @{} }
            $st.Text = if ($need -contains 'ffx' -and -not $r.ffx) { 'AMD FSR unavailable (offline or check failed): the built-in upscaler is used.' }
                elseif ($need -contains 'agility' -and -not $r.agility) { 'DirectX 12 download failed (offline?): Play tries again.' }
                else { $ShaderStatus }
        }.GetNewClosure())
        $tick.Start()
    }
    [void]$m.form.ShowDialog()
    if (-not $m.form.Tag) { return $null }
    @{ action = [string]$m.form.Tag; choice = Get-KlbfFormChoice $m.c; benchPlayer = $m.c.benchPlayer; benchSave = $m.c.benchSave }
}

# "You changed how the game is rendered..." with Prepare now / Later: $true for Prepare now.
function Confirm-KlbfPrepare([string]$Text) {
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    $f = New-Object System.Windows.Forms.Form
    $f.Text = 'KLBF Lab'; Set-KlbfTheme $f; $f.FormBorderStyle = 'FixedDialog'
    $f.MaximizeBox = $false; $f.MinimizeBox = $false; $f.StartPosition = 'CenterScreen'
    $f.ClientSize = New-Object System.Drawing.Size(420, 130)
    $l = New-Object System.Windows.Forms.Label; $l.Text = $Text; $l.SetBounds(16, 14, 388, 56); $f.Controls.Add($l)
    $now = New-Object System.Windows.Forms.Button; $now.Text = 'Prepare now'; $now.DialogResult = 'Yes'; $now.SetBounds(196, 84, 100, 30)
    $later = New-Object System.Windows.Forms.Button; $later.Text = 'Later'; $later.DialogResult = 'No'; $later.SetBounds(304, 84, 100, 30)
    Set-KlbfButton $now $true; Set-KlbfButton $later
    $f.Controls.AddRange(@($now, $later)); $f.AcceptButton = $now; $f.CancelButton = $later
    $f.ShowDialog() -eq 'Yes'
}

# Progress of `victory shaders` from its lines: "N shaders found in <file>" (the total, summed over files),
# "N shaders in the cache, S s" every 1000. @{ total; done } updated in place.
function Update-KlbfPrepareProgress($State, [string]$Line) {
    if ($Line -match '^(\d+) shaders found in ') { $State.total += [int]$Matches[1] }
    elseif ($Line -match '^(\d+) shaders in the cache') { $State.done = [int]$Matches[1] }
}

# (27/11) Progress of `victory warm` from its lines: "... ; M to make, J processes" (the total), "N of M, S s"
# every 100. @{ total; done } updated in place.
function Update-KlbfWarmProgress($State, [string]$Line) {
    if ($Line -match '; (\d+) to make, \d+ processes') { $State.total = [int]$Matches[1] }
    elseif ($Line -match '^(\d+) of (\d+), \d+ s') { $State.done = [int]$Matches[1]; $State.total = [int]$Matches[2] }
}

# The overall bar (0..1000) from the phases' weights, the phase running and how far it is (0..1).
function Get-KlbfPrepareBar([double[]]$Weights, [int]$Phase, [double]$Fraction) {
    $sum = ($Weights | Measure-Object -Sum).Sum
    if ($sum -le 0) { return 0 }
    $before = if ($Phase -gt 0) { ($Weights[0..($Phase - 1)] | Measure-Object -Sum).Sum } else { 0 }
    [int][math]::Round(1000 * ($before + $Weights[$Phase] * [math]::Min(1.0, [math]::Max(0.0, $Fraction))) / $sum)
}

# A process and every process it started (victory's workers, theirs too), ended: the whole tree is listed
# first (a child outlives its parent), then each ended. The number still running after up to 3 s (0: all gone).
function Stop-KlbfTree([int]$Id) {
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Select-Object ProcessId, ParentProcessId)
    $tree = New-Object System.Collections.Generic.List[int]
    $tree.Add($Id)
    for ($i = 0; $i -lt $tree.Count; $i++) {
        foreach ($p in $all) { if ($p.ParentProcessId -eq $tree[$i] -and -not $tree.Contains([int]$p.ProcessId)) { $tree.Add([int]$p.ProcessId) } }
    }
    foreach ($p in $tree) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }
    $left = $tree
    for ($t = 0; $t -lt 30; $t++) {
        $left = @($tree | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
        if (-not $left) { return 0 }
        Start-Sleep -Milliseconds 100
    }
    $left.Count
}

# (27/11) The whole preparation in one window, from 0 to done, before Play or the bench: 1. translating the
# shaders (`victory shaders <rpf>`, unless -NoTranslate), 2. making the pipelines of earlier runs
# (`victory warm` and its workers, when $Book exists). One bar (translation 30 %, pipelines 70 %), the step
# and its count under it; Cancel ends every worker. $true when the translation finished (or was not asked).
function Show-KlbfPrepare([string]$Victory, [string]$Rpf, [string]$Book = '', [switch]$NoTranslate) {
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    $phases = @()
    if (-not $NoTranslate) { $phases += @{ args = 'shaders "' + $Rpf + '"'; weight = 0.3; what = 'Translating shaders'; parse = ${function:Update-KlbfPrepareProgress} } }
    if ($Book -and (Test-Path -LiteralPath $Book)) { $phases += @{ args = 'warm'; weight = 0.7; what = 'Preparing pipelines'; parse = ${function:Update-KlbfWarmProgress} } }
    if (-not $phases) { return $true }
    $f = New-Object System.Windows.Forms.Form
    $f.Text = 'KLBF Lab'; Set-KlbfTheme $f; $f.FormBorderStyle = 'FixedDialog'
    $f.MaximizeBox = $false; $f.MinimizeBox = $false; $f.StartPosition = 'CenterScreen'
    $f.ClientSize = New-Object System.Drawing.Size(460, 140)
    $head = New-Object System.Windows.Forms.Label; $head.Text = 'Preparing the game (once per build; Play starts when it is done)...'; $head.SetBounds(16, 12, 428, 20)
    $bar = New-Object System.Windows.Forms.ProgressBar; $bar.SetBounds(16, 38, 428, 22); $bar.Maximum = 1000
    $last = New-Object System.Windows.Forms.Label; $last.SetBounds(16, 66, 428, 20)
    $cancel = New-Object System.Windows.Forms.Button; $cancel.Text = 'Cancel'; $cancel.SetBounds(344, 98, 100, 30)
    Set-KlbfButton $cancel
    $f.Controls.AddRange(@($head, $bar, $last, $cancel))
    $st = @{ i = 0; total = 0; done = 0; read = $null; ok = $true; p = $null; cancelled = $false }
    $weights = [double[]]@($phases | ForEach-Object { $_.weight })
    $fBar = ${function:Get-KlbfPrepareBar}; $fStop = ${function:Stop-KlbfTree}
    $start = {
        $ph = $phases[$st.i]
        $psi = New-Object System.Diagnostics.ProcessStartInfo($Victory)
        $psi.Arguments = $ph.args
        $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
        $st.total = 0; $st.done = 0
        $st.p = [System.Diagnostics.Process]::Start($psi)
        $null = $st.p.StandardError.ReadToEndAsync()
        $st.read = $st.p.StandardOutput.ReadLineAsync()
        $last.Text = "$($ph.what)..."
    }.GetNewClosure()
    $timer = New-Object System.Windows.Forms.Timer; $timer.Interval = 200
    $timer.add_Tick({
        $ph = $phases[$st.i]
        while ($st.read -and $st.read.IsCompleted) {
            $line = $st.read.Result
            if ($null -eq $line) { $st.read = $null; break }
            & $ph.parse $st $line
            $st.read = $st.p.StandardOutput.ReadLineAsync()
        }
        $frac = if ($st.total -gt 0) { $st.done / $st.total } else { 0 }
        $bar.Value = & $fBar $weights $st.i $frac
        if ($st.total -gt 0) { $last.Text = '{0}: {1} of {2}' -f $ph.what, $st.done, $st.total }
        if (-not $st.read -and $st.p.HasExited) {
            # The translation must succeed; pipelines left over are made while playing.
            if ($ph.args -ne 'warm' -and $st.p.ExitCode -ne 0) { $st.ok = $false }
            if ($st.ok -and $st.i + 1 -lt $phases.Count) { $st.i++; & $start } else { $bar.Value = 1000; $f.Close() }
        }
    }.GetNewClosure())
    $cancel.add_Click({ $st.cancelled = $true; $st.ok = $false; $f.Close() }.GetNewClosure())
    $f.add_Shown({ & $start; $timer.Start() }.GetNewClosure())
    $f.add_FormClosing({ $timer.Stop(); if ($st.p -and -not $st.p.HasExited) { & $fStop $st.p.Id } }.GetNewClosure())
    [void]$f.ShowDialog()
    $st.ok
}
