@echo off
call "%~dp0_run_tests.bat" %*
exit /b %ERRORLEVEL%
