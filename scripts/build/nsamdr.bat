@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "PYTHON=%NSAMDR_PYTHON_EXE%"
if defined PYTHON if not exist "%PYTHON%" if exist "%ROOT%\%PYTHON%" set "PYTHON=%ROOT%\%PYTHON%"
if not defined PYTHON set "PYTHON=%ROOT%\artifacts\nsamdr\python-env\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%ROOT%\artifacts\nsamdr\python-env-cpu\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

set "COMMAND=%~1"
if /I "%COMMAND%"=="gui" goto :gui
if /I "%COMMAND%"=="eve-census" goto :collect
if /I "%COMMAND%"=="structure-audit" goto :collect
if /I "%COMMAND%"=="boundary-profile-audit" goto :collect
if /I "%COMMAND%"=="authored-prior-corpus" goto :collect
if /I "%COMMAND%"=="structure-conditioning-probe" goto :collect
if /I "%COMMAND%"=="full-broad-probe" goto :collect
if /I "%COMMAND%"=="memorization-probe" goto :collect
if /I "%COMMAND%"=="interference-probe" goto :collect
if /I "%COMMAND%"=="stage2-status" goto :collect
if /I "%COMMAND%"=="stage2-probe" goto :collect
if /I "%COMMAND%"=="stage2-summary" goto :collect
goto :cli

:gui
"%PYTHON%" -u "%ROOT%\tools\nsamdr\gui\nsamdr_v16_structure_workflow_gui.py"
exit /b %ERRORLEVEL%

:collect
shift
set "FORWARD_ARGS="
:collect_loop
if "%~1"=="" goto :dispatch
set FORWARD_ARGS=%FORWARD_ARGS% "%~1"
shift
goto :collect_loop

:dispatch
if /I "%COMMAND%"=="eve-census" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\scan_eve_authored_corpus.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="structure-audit" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\audit_nsamdr_v16_structure_support.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="boundary-profile-audit" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\audit_nsamdr_v16_boundary_profiles.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="authored-prior-corpus" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\prepare_nsamdr_v16_authored_prior_corpus.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="structure-conditioning-probe" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\probe_nsamdr_v16_structure_conditioning.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="full-broad-probe" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\probe_nsamdr_v16_full_broad.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="memorization-probe" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\probe_nsamdr_v16_memorization.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="interference-probe" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\probe_nsamdr_v16_interference.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="stage2-status" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\inspect_nsamdr_v16_stage2_state.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="stage2-probe" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\open_nsamdr_v16_stage2_probe.py" --repo-root "%ROOT%" --open %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
if /I "%COMMAND%"=="stage2-summary" (
  "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\summarize_nsamdr_v16_stage2_run.py" --repo-root "%ROOT%" %FORWARD_ARGS%
  exit /b %ERRORLEVEL%
)
exit /b 2

:cli
"%PYTHON%" -u "%ROOT%\tools\nsamdr\nsamdr_cli.py" %*
exit /b %ERRORLEVEL%
