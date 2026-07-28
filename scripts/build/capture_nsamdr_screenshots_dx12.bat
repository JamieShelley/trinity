@echo off
call "%~dp0_run_tests.bat" trinitydev dx12 "*StretchAwareDetail*" screenshots
exit /b %ERRORLEVEL%
