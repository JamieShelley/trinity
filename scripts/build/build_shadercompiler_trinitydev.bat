@echo off
call "%~dp0_build_config.bat" trinitydev stub ShaderCompiler build shader
exit /b %ERRORLEVEL%
