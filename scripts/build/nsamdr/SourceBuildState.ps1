param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Prepare', 'Commit')]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$BuildDir,

    [string]$Context = '',

    [switch]$ForceClean
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$StampPath = Join-Path $BuildDir '.trinity-source-state.sha256'
$PendingPath = Join-Path $BuildDir '.trinity-source-state.pending'

function Invoke-GitLines {
    param(
        [Parameter(Mandatory = $true)][string]$WorkingTree,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$AllowExitCodeOne
    )

    # Windows PowerShell 5.1 exposes text written by native programs to stderr as
    # ErrorRecord objects. With the script-wide ErrorActionPreference set to Stop,
    # harmless Git warnings (for example LF/CRLF conversion notices) otherwise abort
    # source-state calculation before LASTEXITCODE can be checked.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $nativeOutput = @(& git.exe -C $WorkingTree @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $stdout = New-Object System.Collections.Generic.List[string]
    $stderr = New-Object System.Collections.Generic.List[string]
    foreach ($record in $nativeOutput) {
        if ($record -is [System.Management.Automation.ErrorRecord]) {
            $stderr.Add([string]$record.Exception.Message)
        } else {
            $stdout.Add([string]$record)
        }
    }

    if ($exitCode -ne 0 -and -not ($AllowExitCodeOne -and $exitCode -eq 1)) {
        $diagnostics = New-Object System.Collections.Generic.List[string]
        foreach ($line in $stdout) { $diagnostics.Add($line) }
        foreach ($line in $stderr) { $diagnostics.Add($line) }
        $joined = ($diagnostics -join [Environment]::NewLine)
        throw "git $($Arguments -join ' ') failed in '$WorkingTree' with exit code $exitCode.`n$joined"
    }

    # Only stdout forms part of the deterministic source signature. Successful Git
    # warnings are intentionally excluded so they neither stop the build nor change
    # the fingerprint.
    return $stdout.ToArray()
}

function Test-IgnoredStatePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = $RelativePath.Replace('\', '/').TrimStart([char[]]'./')
    return (
        $normalized -match '^\.cmake-build-' -or
        $normalized -match '^artifacts/' -or
        $normalized -match '^\.vs/' -or
        $normalized -match '^out/' -or
        $normalized -match '^build/' -or
        $normalized -match '/__pycache__/' -or
        $normalized -match '\.pyc$'
    )
}

function Add-RepositoryState {
    param(
        [Parameter(Mandatory = $true)][string]$WorkingTree,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][System.Text.StringBuilder]$Manifest,
        [Parameter(Mandatory = $true)][hashtable]$Visited
    )

    $fullPath = [System.IO.Path]::GetFullPath($WorkingTree)
    $visitKey = $fullPath.ToLowerInvariant()
    if ($Visited.ContainsKey($visitKey)) {
        return
    }
    $Visited[$visitKey] = $true

    [void]$Manifest.AppendLine("repo=$Label")
    [void]$Manifest.AppendLine("path=$fullPath")

    $head = (Invoke-GitLines -WorkingTree $fullPath -Arguments @('rev-parse', 'HEAD') | Select-Object -First 1)
    [void]$Manifest.AppendLine("head=$head")

    $trackedDiff = Invoke-GitLines -WorkingTree $fullPath -Arguments @('diff', '--binary', '--no-ext-diff', 'HEAD', '--', '.')
    [void]$Manifest.AppendLine('tracked-diff-begin')
    foreach ($line in $trackedDiff) {
        [void]$Manifest.AppendLine([string]$line)
    }
    [void]$Manifest.AppendLine('tracked-diff-end')

    $untrackedFiles = Invoke-GitLines -WorkingTree $fullPath -Arguments @('-c', 'core.quotepath=false', 'ls-files', '--others', '--exclude-standard')
    foreach ($relativePathRaw in ($untrackedFiles | Sort-Object)) {
        $relativePath = [string]$relativePathRaw
        if ([string]::IsNullOrWhiteSpace($relativePath) -or (Test-IgnoredStatePath -RelativePath $relativePath)) {
            continue
        }

        $absolutePath = Join-Path $fullPath $relativePath
        if (Test-Path -LiteralPath $absolutePath -PathType Leaf) {
            $fileHash = (Get-FileHash -LiteralPath $absolutePath -Algorithm SHA256).Hash.ToLowerInvariant()
            $fileLength = (Get-Item -LiteralPath $absolutePath).Length
            [void]$Manifest.AppendLine("untracked=$relativePath|$fileLength|$fileHash")
        } else {
            [void]$Manifest.AppendLine("untracked-missing=$relativePath")
        }
    }

    $gitModulesPath = Join-Path $fullPath '.gitmodules'
    if (Test-Path -LiteralPath $gitModulesPath -PathType Leaf) {
        $moduleLines = Invoke-GitLines -WorkingTree $fullPath -Arguments @('config', '--file', $gitModulesPath, '--get-regexp', '^submodule\..*\.path$') -AllowExitCodeOne
        foreach ($moduleLineRaw in $moduleLines) {
            $moduleLine = [string]$moduleLineRaw
            if ($moduleLine -notmatch '^\S+\s+(.+)$') {
                continue
            }

            $moduleRelativePath = $Matches[1]
            $moduleFullPath = Join-Path $fullPath $moduleRelativePath
            if (-not (Test-Path -LiteralPath $moduleFullPath -PathType Container)) {
                [void]$Manifest.AppendLine("submodule-missing=$moduleRelativePath")
                continue
            }

            $moduleGitMarker = Join-Path $moduleFullPath '.git'
            if (-not (Test-Path -LiteralPath $moduleGitMarker)) {
                [void]$Manifest.AppendLine("submodule-uninitialised=$moduleRelativePath")
                continue
            }

            $isWorkTree = Invoke-GitLines -WorkingTree $moduleFullPath -Arguments @('rev-parse', '--is-inside-work-tree')
            if (($isWorkTree | Select-Object -First 1) -eq 'true') {
                Add-RepositoryState -WorkingTree $moduleFullPath -Label "$Label/$moduleRelativePath" -Manifest $Manifest -Visited $Visited
            } else {
                [void]$Manifest.AppendLine("submodule-uninitialised=$moduleRelativePath")
            }
        }
    }
}

function Get-SourceStateSignature {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$BuildContext
    )

    $manifest = New-Object System.Text.StringBuilder
    [void]$manifest.AppendLine('schema=2')
    [void]$manifest.AppendLine("context=$BuildContext")

    $visited = @{}
    Add-RepositoryState -WorkingTree $Root -Label '.' -Manifest $manifest -Visited $visited

    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($manifest.ToString())
        $hashBytes = $sha256.ComputeHash($bytes)
        return ([System.BitConverter]::ToString($hashBytes)).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha256.Dispose()
    }
}

if ($Mode -eq 'Commit') {
    if (-not (Test-Path -LiteralPath $PendingPath -PathType Leaf)) {
        throw "Pending source-state file was not found: $PendingPath"
    }

    Move-Item -LiteralPath $PendingPath -Destination $StampPath -Force
    Write-Host 'Recorded source state for the successful configure/build.'
    exit 0
}

$signature = Get-SourceStateSignature -Root $RepoRoot -BuildContext $Context
$previousSignature = ''
if (Test-Path -LiteralPath $StampPath -PathType Leaf) {
    $previousSignature = (Get-Content -LiteralPath $StampPath -Raw).Trim()
}

$mustClean = $ForceClean.IsPresent -or [string]::IsNullOrWhiteSpace($previousSignature) -or ($previousSignature -ne $signature)

if ($mustClean) {
    if ($ForceClean.IsPresent) {
        Write-Host 'Clean build requested explicitly.'
    } elseif ([string]::IsNullOrWhiteSpace($previousSignature)) {
        Write-Host 'No successful source-state stamp exists for this build directory.'
    } else {
        Write-Host 'Git/source state changed since the last successful build.'
    }

    if (Test-Path -LiteralPath $BuildDir -PathType Container) {
        Write-Host "Removing build directory: $BuildDir"
        Remove-Item -LiteralPath $BuildDir -Recurse -Force
    }
} else {
    Write-Host 'Git/source state exactly matches the last successful build; reusing the build directory.'
}

New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
Set-Content -LiteralPath $PendingPath -Value $signature -Encoding ASCII -NoNewline
Write-Host "Source state: $signature"
exit 0
