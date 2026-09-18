@echo off
setlocal
cd /d "%~dp0"

py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 goto fallback

py -3.12 run.py %*
exit /b %errorlevel%

:fallback
python run.py %*
exit /b %errorlevel%
