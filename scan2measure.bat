@echo off
rem scan2measure launcher - double-click from Windows Explorer.
rem Resolves its own location, translates to a WSL path, and execs the AppImage.

setlocal

rem %~dp0 is the directory of this .bat file (with trailing backslash)
set "REPO_DIR_WIN=%~dp0"
rem Strip trailing backslash for wslpath
if "%REPO_DIR_WIN:~-1%"=="\" set "REPO_DIR_WIN=%REPO_DIR_WIN:~0,-1%"

rem Translate Windows path to a WSL path (e.g. C:\foo -> /mnt/c/foo, or \\wsl$\... -> /...)
for /f "usebackq delims=" %%I in (`wsl.exe wslpath -a "%REPO_DIR_WIN%"`) do set "REPO_DIR_WSL=%%I"

if "%REPO_DIR_WSL%"=="" (
    echo ERROR: failed to translate "%REPO_DIR_WIN%" to a WSL path. Is WSL installed?
    pause
    exit /b 1
)

echo Starting scan2measure from %REPO_DIR_WSL% ...

wsl.exe bash -l -c "'%REPO_DIR_WSL%/app/release/scan2measure.AppImage'" 2> "%TEMP%\scan2measure_err.log"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo Exit code: %RC%
    echo Error occurred. Check %TEMP%\scan2measure_err.log
    pause
)

endlocal
