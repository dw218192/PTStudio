@echo off
setlocal

set "ROOT=%~dp0"
set "VENV_PY=%ROOT%_tools\venv\Scripts\python.exe"
set "BOOTSTRAP=%ROOT%tools\bootstrap.ps1"
REM Check if bootstrapping is needed by checking if the venv Python exists
if not exist "%VENV_PY%" (
    echo Bootstrapping repository tooling...
    powershell -ExecutionPolicy ByPass -File "%BOOTSTRAP%"
    if errorlevel 1 (
        echo Bootstrap failed!
        exit /b 1
    )
)

REM Invoke the Python package module
pushd "%ROOT%tools"
"%VENV_PY%" -m repo_tools %*
set EXIT_CODE=%ERRORLEVEL%
popd
exit /b %EXIT_CODE%

endlocal

