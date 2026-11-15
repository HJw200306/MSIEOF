@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "ANNOTATOR=solid\annotation\seg_annotator\run.py"
set "ANNOTATION_DATA=solid\annotation\annotation_v1"
set "MSIEOF_PY=D:\miniconda3\envs\msirost\python.exe"

if exist "%MSIEOF_PY%" (
  "%MSIEOF_PY%" "%ANNOTATOR%" --data-dir "%ANNOTATION_DATA%" %*
) else (
  python "%ANNOTATOR%" --data-dir "%ANNOTATION_DATA%" %*
)

if errorlevel 1 pause
