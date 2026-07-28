@echo off
setlocal
set "NSAMDR_BUILD_ONLY=1"
call "%~dp0run_nsamdr_realship_dx11.bat" %*
exit /b %ERRORLEVEL%
