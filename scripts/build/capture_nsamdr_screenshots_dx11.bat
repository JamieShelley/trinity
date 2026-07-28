@echo off
call "%~dp0_run_tests.bat" trinitydev dx11 "*StretchAwareDetail*" screenshots
exit /b %ERRORLEVEL%
