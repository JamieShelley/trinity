@echo off
rem Compatibility alias: the default NSAMDR preview is now the real OBJ path.
call "%~dp0run_nsamdr_obj_preview_dx11.bat" %*
exit /b %ERRORLEVEL%
