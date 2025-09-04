@echo off
echo Cleaning embarrassing test data...
taskkill /f /im python.exe /t 2>nul
timeout /t 2 /nobreak >nul
rmdir /s /q brainbox_data 2>nul
mkdir brainbox_data 2>nul
echo Data cleaned! Safe for GitHub now.
pause