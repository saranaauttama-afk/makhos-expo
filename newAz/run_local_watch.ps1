param(
    [Parameter(Mandatory = $true)]
    [string]$DriveDir,

    [switch]$NoTsVerify,

    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$pythonArgs = @(
    (Join-Path $scriptDir 'eval_local.py'),
    '--drive-dir', $DriveDir,
    '--watch',
    '--poll-seconds', $PollSeconds
)

if (-not $NoTsVerify) {
    $pythonArgs += '--ts-verify'
}

Write-Host "Starting newAz local watch..." -ForegroundColor Cyan
Write-Host "  DriveDir   : $DriveDir"
Write-Host "  TS Verify  : $(-not $NoTsVerify)"
Write-Host "  Poll       : $PollSeconds s"
Write-Host ""

Set-Location $repoRoot
python @pythonArgs
