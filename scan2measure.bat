@echo off
rem scan2measure launcher - double-click from Windows Explorer.
rem Resolves its own location, translates to a WSL path, and execs the AppImage.

setlocal EnableDelayedExpansion

rem %~dp0 is the directory of this .bat file (with trailing backslash)
set "REPO_DIR_WIN=%~dp0"
rem Strip trailing backslash
if "%REPO_DIR_WIN:~-1%"=="\" set "REPO_DIR_WIN=%REPO_DIR_WIN:~0,-1%"

rem Translate Windows path -> WSL path.
rem Special case: \\wsl.localhost\<distro>\... and \\wsl$\<distro>\...
rem must be parsed directly, because `wsl.exe wslpath -a` mistranslates
rem these UNC paths (treats the WSL share as a /mnt/c subdirectory).
set "REPO_DIR_WSL="

if /i "!REPO_DIR_WIN:~0,16!"=="\\wsl.localhost\" (
    set "_rest=!REPO_DIR_WIN:~16!"
    rem _rest now: <distro>\<path-with-backslashes>
    for /f "tokens=1,* delims=\" %%A in ("!_rest!") do set "_path=%%B"
    rem replace backslashes with forward slashes and prepend /
    set "REPO_DIR_WSL=/!_path:\=/!"
) else if /i "!REPO_DIR_WIN:~0,7!"=="\\wsl$\" (
    set "_rest=!REPO_DIR_WIN:~7!"
    for /f "tokens=1,* delims=\" %%A in ("!_rest!") do set "_path=%%B"
    set "REPO_DIR_WSL=/!_path:\=/!"
) else (
    rem Standard Windows path: ask wslpath
    for /f "usebackq delims=" %%I in (`wsl.exe wslpath -a "!REPO_DIR_WIN!"`) do set "REPO_DIR_WSL=%%I"
)

if "!REPO_DIR_WSL!"=="" (
    echo ERROR: failed to translate "!REPO_DIR_WIN!" to a WSL path. Is WSL installed?
    pause
    exit /b 1
)

echo Starting scan2measure from !REPO_DIR_WSL! ...

rem Delegate to scripts/launch-appimage.sh - it handles WSLg env defaults
rem (DISPLAY / WAYLAND_DISPLAY / XDG_RUNTIME_DIR) and execs the AppImage.
rem Going through a real bash script avoids cmd.exe quoting hell when
rem trying to embed bash parameter expansion inline.
wsl.exe bash -l "!REPO_DIR_WSL!/scripts/launch-appimage.sh" 2> "%TEMP%\scan2measure_err.log"
set "RC=!ERRORLEVEL!"

if not "!RC!"=="0" (
    echo Exit code: !RC!
    echo Error occurred. Check %TEMP%\scan2measure_err.log
    pause
)

endlocal
