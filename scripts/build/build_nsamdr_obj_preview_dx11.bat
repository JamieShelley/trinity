@echo off
setlocal EnableExtensions
call "%~dp0verify_and_clean_nsamdr_layout.bat"
if errorlevel 1 exit /b %ERRORLEVEL%
set "NSAMDR_BUILD_ONLY=1"
call "%~dp0run_nsamdr_obj_preview_dx11.bat"
exit /b %ERRORLEVEL%
