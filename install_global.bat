@echo off
REM BrainBox Global Command Installer
REM Creates 'brainbox' command that works from anywhere

set BRAINBOX_DIR=%~dp0
set SCRIPT_DIR=%USERPROFILE%\bin

REM Create user bin directory if it doesn't exist
if not exist "%SCRIPT_DIR%" mkdir "%SCRIPT_DIR%"

REM Create brainbox.bat in user bin
echo @echo off > "%SCRIPT_DIR%\brainbox.bat"
echo python "%BRAINBOX_DIR%brainbox_cli.py" %%* >> "%SCRIPT_DIR%\brainbox.bat"

REM Add user bin to PATH if not already there
echo.
echo Adding %SCRIPT_DIR% to PATH...
setx PATH "%PATH%;%SCRIPT_DIR%" >nul 2>&1

echo.
echo ✅ BrainBox global command installed!
echo.
echo Now you can type 'brainbox' from anywhere.
echo.
echo Restart your terminal and try:
echo   brainbox --status
echo   brainbox
echo.
pause