@echo off
call "%~dp0_build_config.bat" trinitydev stub ShaderCompilerTest build shader
exit /b %ERRORLEVEL%
