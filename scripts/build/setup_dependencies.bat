@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
if not exist "%ROOT%\.git" (
    echo ERROR: "%ROOT%" is not a Git clone.
    echo A ZIP download cannot initialise the required Git submodules.
    echo Clone the repository with Git, then copy these scripts into it.
    exit /b 2
)

where git.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: git.exe was not found in PATH.
    exit /b 3
)

pushd "%ROOT%" || exit /b 4

echo Synchronising submodule definitions...
git submodule sync --recursive
if errorlevel 1 goto :failed

echo Initialising vcpkg and the Carbon vcpkg registry...
rem The repository records SSH URLs. This command temporarily rewrites GitHub
rem SSH URLs to HTTPS without changing the user's global Git configuration.
git -c url."https://github.com/".insteadOf="git@github.com:" ^
    submodule update --init --recursive

if errorlevel 1 goto :failed

popd
echo Dependencies initialised successfully.
exit /b 0

:failed
set "RESULT=%ERRORLEVEL%"
popd
echo ERROR: Submodule initialisation failed with exit code %RESULT%.
exit /b %RESULT%
