@echo off
cd /d "%~dp0"
set "TRAIN_PYTHON=D:\miniconda3\envs\msirost\python.exe"
if not exist "%TRAIN_PYTHON%" set "TRAIN_PYTHON=python"
"%TRAIN_PYTHON%" train.py %*
pause
