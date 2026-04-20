@echo off
REM Unattended ablation runner — invoked by Windows Scheduled Task.
REM Logs stdout+stderr to ablation_run.log; the Python script also writes
REM per-batch entries to ablation_progress.log.

cd /d C:\Users\antoi\Documents\forge

"C:\Users\antoi\AppData\Local\Programs\Python\Python313\python.exe" ^
  scripts\run_ablation.py ^
  --models-dir "C:\Users\antoi\tools\models" ^
  >> ablation_run.log 2>&1
