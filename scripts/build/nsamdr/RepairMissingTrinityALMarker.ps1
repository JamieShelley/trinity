param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath($RepoRoot)
$git = Get-Command git.exe -ErrorAction SilentlyContinue
if ($null -eq $git) {
    throw 'git.exe is required to verify and repair the tracked TrinityAL source tree.'
}

$insideWorkTree = & $git.Source -C $repo rev-parse --is-inside-work-tree 2>$null
if ($LASTEXITCODE -ne 0 -or $insideWorkTree -ne 'true') {
    throw "'$repo' is not a Git working tree."
}

# The NSAMDR directory is owned by this override and may contain uncommitted
# files.  Everything else below trinityal is upstream CarbonEngine source.
# Query only tracked files that are absent from the worktree.  This does not
# compare file hashes and does not overwrite any file that currently exists.
$deletedOutput = & $git.Source -C $repo ls-files --deleted -- trinityal
if ($LASTEXITCODE -ne 0) {
    throw "git ls-files failed with exit code $LASTEXITCODE."
}

$missing = @(
    $deletedOutput |
        ForEach-Object { $_.Trim() } |
        Where-Object {
            $_ -and
            -not $_.StartsWith('trinityal/tests/nsamdr/', [System.StringComparison]::OrdinalIgnoreCase)
        }
)

if ($missing.Count -eq 0) {
    Write-Host 'Tracked TrinityAL source tree is complete.'
    exit 0
}

if ($CheckOnly) {
    Write-Error ('Missing tracked TrinityAL files: {0}' -f ($missing -join ', '))
    exit 1
}

Write-Host ('Restoring {0} missing tracked TrinityAL file(s) only...' -f $missing.Count)
$batchSize = 64
for ($offset = 0; $offset -lt $missing.Count; $offset += $batchSize) {
    $end = [Math]::Min($offset + $batchSize - 1, $missing.Count - 1)
    $batch = @($missing[$offset..$end])
    $arguments = @('-C', $repo, 'restore', '--source=HEAD', '--worktree', '--') + $batch
    & $git.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git restore failed with exit code $LASTEXITCODE."
    }
}

$notRestored = @()
foreach ($relativePath in $missing) {
    $absolutePath = Join-Path $repo ($relativePath -replace '/', '\')
    if (-not (Test-Path -LiteralPath $absolutePath -PathType Leaf)) {
        $notRestored += $relativePath
    }
}
if ($notRestored.Count -ne 0) {
    throw "Git completed without restoring: $($notRestored -join ', ')"
}

Write-Host ('Restored missing tracked TrinityAL files: {0}' -f $missing.Count)
Write-Host 'Existing files were not overwritten. No commit was created. No source-state comparison was performed.'
exit 0
