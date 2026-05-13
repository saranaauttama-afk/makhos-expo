Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Label,
    [Parameter(Mandatory = $true)]
    [string[]]$Command
  )

  Write-Host ""
  Write-Host "==> $Label"
  Write-Host ("    " + ($Command -join " "))

  & $Command[0] $Command[1..($Command.Length - 1)]
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Label (exit $LASTEXITCODE)"
  }
}

Invoke-Step -Label "Perft" -Command @("npm.cmd", "run", "test:perft")
Invoke-Step -Label "Gate Report" -Command @("npm.cmd", "run", "gate:ai:report")
Invoke-Step -Label "Gate Repeat" -Command @("npm.cmd", "run", "gate:ai:repeat")
