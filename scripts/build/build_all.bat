@echo off
rem Builds all TrinityDev targets: DX11, DX12, stub, tests, and ShaderCompiler.
call "%~dp0_build_config.bat" trinitydev both ALL build shader
exit /b %ERRORLEVEL%
