@echo off
setlocal EnableExtensions
set "NSAMDR_BUILD_ONLY=1"
call "%~dp0run_nsamdr_obj_preview_dx11.bat"
exit /b %ERRORLEVEL%
