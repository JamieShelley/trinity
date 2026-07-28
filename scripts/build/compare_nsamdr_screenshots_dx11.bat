@echo off
call "%~dp0_run_tests.bat" trinitydev dx11 "*StretchAwareDetail*" compare
exit /b %ERRORLEVEL%
