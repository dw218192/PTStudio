param(
    [switch]$x,  # Rebuild flag: removes build folder before building
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$BuildType = "Debug"  # Build configuration type, defaults to Debug
)

$ErrorActionPreference = "Stop"

Push-Location

# Remove build directory if -x flag is provided
if ($x -and (Test-Path build)) {
    Write-Host "Rebuild flag (-x) detected. Removing build folder..." -ForegroundColor Yellow
    Remove-Item -Path build -Recurse -Force
}

# Create build directory if missing
if (-not (Test-Path build)) {
    New-Item -ItemType Directory -Path build | Out-Null
}

Set-Location build

Write-Host "Building with configuration: $BuildType" -ForegroundColor Cyan

conan install .. --build=missing --output-folder .

cmake .. `
    "-DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake" `
    -DCMAKE_BUILD_TYPE=$BuildType

cmake --build . --config $BuildType

Pop-Location
