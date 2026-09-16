@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "PYTHON=%NSAMDR_PYTHON_EXE%"
if defined PYTHON if not exist "%PYTHON%" if exist "%ROOT%\%PYTHON%" set "PYTHON=%ROOT%\%PYTHON%"
if not defined PYTHON set "PYTHON=%ROOT%\artifacts\nsamdr\python-env\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%ROOT%\artifacts\nsamdr\python-env-cpu\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

if /I "%~1"=="gui" goto :gui
if /I "%~1"=="diversity" goto :diversity
if /I "%~1"=="battleship-diversity" goto :battleship_diversity
if /I "%~1"=="family-metrics" goto :family_metrics
if /I "%~1"=="stage2-status" goto :stage2_status
if /I "%~1"=="stage2-probe" goto :stage2_probe
if /I "%~1"=="stage2-summary" goto :stage2_summary
if /I "%~1"=="stage2-audit" goto :stage2_audit
goto :cli

:gui
"%PYTHON%" -u "%ROOT%\tools\nsamdr\gui\nsamdr_v16_lowimpact_monitored_workflow_gui.py"
exit /b %ERRORLEVEL%

:diversity
shift
set "FORWARD_ARGS="
goto :collect_diversity_args

:collect_diversity_args
if "%~1"=="" goto :run_diversity
set FORWARD_ARGS=%FORWARD_ARGS% "%~1"
shift
goto :collect_diversity_args

:run_diversity
"%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\discover_nsamdr_v16_raven_diversity.py" --repo-root "%ROOT%" %FORWARD_ARGS%
exit /b %ERRORLEVEL%

:battleship_diversity
shift
set "FORWARD_ARGS="
goto :collect_battleship_diversity_args

:collect_battleship_diversity_args
if "%~1"=="" goto :run_battleship_diversity
set FORWARD_ARGS=%FORWARD_ARGS% "%~1"
shift
goto :collect_battleship_diversity_args

:run_battleship_diversity
"%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\discover_nsamdr_v16_caldari_battleship_diversity.py" --repo-root "%ROOT%" %FORWARD_ARGS%
exit /b %ERRORLEVEL%

:family_metrics
shift
set "FORWARD_ARGS="
goto :collect_family_metrics_args

:collect_family_metrics_args
if "%~1"=="" goto :run_family_metrics
set FORWARD_ARGS=%FORWARD_ARGS% "%~1"
shift
goto :collect_family_metrics_args

:run_family_metrics
"%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\analyze_nsamdr_v16_multiregion_families.py" --repo-root "%ROOT%" %FORWARD_ARGS%
exit /b %ERRORLEVEL%

:stage2_status
"%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\inspect_nsamdr_v16_stage2_state.py" --repo-root "%ROOT%"
exit /b %ERRORLEVEL%

:stage2_probe
"%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\open_nsamdr_v16_stage2_probe.py" --repo-root "%ROOT%" --open
exit /b %ERRORLEVEL%

:stage2_summary
shift
set "FORWARD_ARGS="
goto :collect_stage2_summary_args

:collect_stage2_summary_args
if "%~1"=="" goto :run_stage2_summary
set FORWARD_ARGS=%FORWARD_ARGS% "%~1"
shift
goto :collect_stage2_summary_args

:run_stage2_summary
"%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\summarize_nsamdr_v16_stage2_run.py" --repo-root "%ROOT%" %FORWARD_ARGS%
exit /b %ERRORLEVEL%

:stage2_audit
shift
set "FORWARD_ARGS="
goto :collect_stage2_audit_args

:collect_stage2_audit_args
if "%~1"=="" goto :run_stage2_audit
set FORWARD_ARGS=%FORWARD_ARGS% "%~1"
shift
goto :collect_stage2_audit_args

:run_stage2_audit
"%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\audit_nsamdr_v16_stage2_family_difficulty.py" --repo-root "%ROOT%" %FORWARD_ARGS%
exit /b %ERRORLEVEL%

:cli
"%PYTHON%" -u "%ROOT%\tools\nsamdr\nsamdr_cli.py" %*
exit /b %ERRORLEVEL%
