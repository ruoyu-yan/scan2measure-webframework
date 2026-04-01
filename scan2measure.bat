@echo off
rem scan2measure launcher — double-click from Windows Explorer
echo Starting scan2measure...
wsl.exe bash -l -c "/home/ruoyu/scan2measure-webframework/app/release/scan2measure.AppImage" 2> "%TEMP%\scan2measure_err.log"
echo Exit code: %ERRORLEVEL%
if %ERRORLEVEL% neq 0 (
    echo Error occurred. Check %TEMP%\scan2measure_err.log
    pause
)
pause
