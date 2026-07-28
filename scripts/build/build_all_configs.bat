@echo off
setlocal EnableExtensions

call "%~dp0build_debug_dx11.bat" || exit /b %ERRORLEVEL%
call "%~dp0build_debug_dx12.bat" || exit /b %ERRORLEVEL%
call "%~dp0build_release_dx11.bat" || exit /b %ERRORLEVEL%
call "%~dp0build_release_dx12.bat" || exit /b %ERRORLEVEL%
call "%~dp0build_internal_dx11.bat" || exit /b %ERRORLEVEL%
call "%~dp0build_internal_dx12.bat" || exit /b %ERRORLEVEL%
call "%~dp0build_trinitydev_dx11.bat" || exit /b %ERRORLEVEL%
call "%~dp0build_trinitydev_dx12.bat" || exit /b %ERRORLEVEL%

echo All Windows configurations and rendering backends built successfully.
exit /b 0
