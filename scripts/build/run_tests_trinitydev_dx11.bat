@echo off
call "%~dp0_run_tests.bat" trinitydev dx11 "*" normal
exit /b %ERRORLEVEL%
