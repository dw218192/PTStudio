param(
    [switch]$x,  # Rebuild flag: removes build folder before building
    [switch]$u,  # Update lock flag: forces regeneration of conan.lock
    [switch]$c,  # Configure only flag: runs conan install and cmake configure, but skips building
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$BuildType = "Debug",  # Build configuration type, defaults to Debug
    [string]$ConanProfile = "default"  # Conan profile, defaults to default
)

$ErrorActionPreference = "Stop"

function Ensure-ConanProfile {
    $profileDir = Join-Path $HOME ".conan2/profiles"

    if (-not (Test-Path $profileDir) -or 
        -not (Get-ChildItem $profileDir -File -Filter "*")) {

        Write-Host "No Conan profiles found. Running 'conan profile detect'..."
        conan profile detect
    }
    else {
        Write-Host "Conan profiles already exist."
    }
}

Push-Location
try {
    $BuildFolder = "_build"

    # Remove build directory if -x flag is provided
    if ($x -and (Test-Path $BuildFolder)) {
        Write-Host "Rebuild flag (-x) detected. Removing build folder..." -ForegroundColor Yellow
        Remove-Item -Path $BuildFolder -Recurse -Force
    }

    # Create build directory if missing
    if (-not (Test-Path $BuildFolder)) {
        New-Item -ItemType Directory -Path $BuildFolder | Out-Null
    }

    Set-Location $BuildFolder
    Ensure-ConanProfile

    # Create logs directory
    $logsDir = "..\_logs"
    if (-not (Test-Path $logsDir)) {
        New-Item -ItemType Directory -Path $logsDir | Out-Null
    }

    if ($c) {
        Write-Host "Configuring with configuration: $BuildType" -ForegroundColor Cyan
    }
    else {
        Write-Host "Building with configuration: $BuildType" -ForegroundColor Cyan
    }

    # Handle lock file generation and usage
    $lockFile = "..\conan.lock"    
    $shouldCreateLock = $u -or -not (Test-Path $lockFile)
    
    if ($shouldCreateLock) {
        if ($u) {
            Write-Host "Update lock flag (-u) detected. Regenerating lock file..." -ForegroundColor Yellow
        }
        else {
            Write-Host "Lock file not found. Generating new lock file..." -ForegroundColor Yellow
        }
        $lockLogFile = Join-Path $logsDir "conan_lock_create.log"
        conan lock create .. --lockfile-out $lockFile | Tee-Object -FilePath $lockLogFile
    }
    else {
        Write-Host "Lock file found. Using existing lock file: $lockFile" -ForegroundColor Green
    }
    
    $installLogFile = Join-Path $logsDir "conan_install.log"
    conan install .. --lockfile $lockFile `
        --output-folder=$BuildType `
        --deployer-folder=deps `
        --deployer=full_deploy `
        --build=missing `
        --profile:host=$ConanProfile `
        --profile:build=$ConanProfile `
        -s build_type=$BuildType | Tee-Object -FilePath $installLogFile

    $configureLogFile = Join-Path $logsDir "cmake_configure.log"
    cmake -S .. -B $BuildType `
        "-DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake" `
        -DCMAKE_BUILD_TYPE=$BuildType `
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON | Tee-Object -FilePath $configureLogFile

    if (-not $c) {
        $buildLogFile = Join-Path $logsDir "cmake_build.log"
        cmake --build $BuildType --config $BuildType | Tee-Object -FilePath $buildLogFile
    }
    else {
        Write-Host "Configure only mode (-c): Skipping build step" -ForegroundColor Yellow
    }
}
finally {
    Pop-Location
}
