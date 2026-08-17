@echo off
(for %%A in (%*) do @echo %%~A)>"%FORGE_BOOTSTRAP_HANDOFF_LOG%"
echo powershell handoff stdout
echo powershell handoff stderr 1>&2
exit /b %FORGE_BOOTSTRAP_HANDOFF_STATUS%
