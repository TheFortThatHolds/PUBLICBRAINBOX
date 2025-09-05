@echo off
REM Install BrainBox as global CLI command
REM Run this once to enable typing 'brainbox' from anywhere

echo 🧠 Installing BrainBox Global CLI...
echo.

REM Get the current directory (where BrainBox is installed)
set BRAINBOX_PATH=%~dp0
set BRAINBOX_PATH=%BRAINBOX_PATH:~0,-1%

echo BrainBox installation path: %BRAINBOX_PATH%
echo.

REM Create the global brainbox.bat command
set CLI_PATH=%USERPROFILE%\brainbox.bat

echo @echo off > "%CLI_PATH%"
echo python "%BRAINBOX_PATH%\brainbox_cli.py" %%* >> "%CLI_PATH%"

echo Created global command at: %CLI_PATH%
echo.

REM Add to PATH if not already there
echo %PATH% | findstr /i "%USERPROFILE%" >nul
if errorlevel 1 (
    echo Adding %USERPROFILE% to your PATH...
    echo.
    echo NOTE: You may need to restart your terminal for PATH changes to take effect.
    setx PATH "%PATH%;%USERPROFILE%"
) else (
    echo %USERPROFILE% is already in your PATH
)

echo.
echo ✅ Installation complete!
echo.
echo To test, open a new terminal and type:
echo    brainbox
echo.
echo Your AI spine will activate from any directory.
pause