@echo off
setlocal EnableExtensions
rem Clear, dedicated entry point for the Granny-SDK-free real EVE asset test.
call "%~dp0run_nsamdr_eve_asset_dx11.bat" %*
exit /b %ERRORLEVEL%
