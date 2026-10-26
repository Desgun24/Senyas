@echo off
cd /d "%~dp0"
if not exist "%~dp0.runtime\python\pythonw.exe" goto install
start "" "%~dp0.runtime\python\pythonw.exe" "%~dp0scripts\launch_desktop.py"
exit /b 0
:install
call "%~dp0Install Senyas.cmd"
exit /b
