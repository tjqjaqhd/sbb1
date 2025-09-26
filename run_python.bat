@echo off
REM UTF-8 Environment Setup
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Run Python with UTF-8 support
python %*
