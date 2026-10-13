@echo off
REM Diagnostics (INT8 speed, analog relative error vs Theorem 1, noise controls). Run AFTER run_main.bat.
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
python cpd_diag.py
echo ===== DIAG DONE =====
