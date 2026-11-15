@echo off
cd /d "%~dp0"
set "COMPARE_PYTHON=D:\miniconda3\envs\msirost\python.exe"
if not exist "%COMPARE_PYTHON%" set "COMPARE_PYTHON=python"
"%COMPARE_PYTHON%" compare_models.py %*
pause
