
$ErrorActionPreference = "Stop"

# OS-dependent bootstrap script for initializing the build system.
# All it does is pulling a python interpreter, and then we can use python to do workspace tasks like building, testing, formatting, etc.

$Root = (Resolve-Path "$PSScriptRoot\..").Path
$Tools = Join-Path $Root "_tools"
$Bin = Join-Path $Tools "bin"
$Pys = Join-Path $Tools "python"
$Cache = Join-Path $Tools "cache\uv"
$Venv = Join-Path $Tools "venv"

New-Item -ItemType Directory -Force -Path $Bin, $Pys, $Cache | Out-Null

$Uv = Join-Path $Bin "uv.exe"
if (-not (Test-Path $Uv)) {
    $env:UV_INSTALL_DIR = $Bin
    $env:UV_NO_MODIFY_PATH = "1"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

$env:UV_CACHE_DIR = $Cache
$env:UV_PYTHON_INSTALL_DIR = $Pys
$env:UV_MANAGED_PYTHON = "1"

& $Uv python install
if (-not (Test-Path $Venv)) { & $Uv venv $Venv }

$Py = Join-Path $Venv "Scripts\python.exe"
$Requirements = Join-Path $Root "tools\requirements.txt"
& $Uv pip install --python $Py -r $Requirements

Write-Host "OK: $Venv"
