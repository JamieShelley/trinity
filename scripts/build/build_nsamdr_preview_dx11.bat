@echo off
rem Compatibility alias: build the Granny-free real OBJ preview.
call "%~dp0build_nsamdr_obj_preview_dx11.bat" %*
exit /b %ERRORLEVEL%
