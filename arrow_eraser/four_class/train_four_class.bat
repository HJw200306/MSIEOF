@echo off
cd /d "%~dp0"
if defined ARROW_PYTHON (
  "%ARROW_PYTHON%" train.py --classes 4 %*
) else (
  python train.py --classes 4 %*
)
pause
