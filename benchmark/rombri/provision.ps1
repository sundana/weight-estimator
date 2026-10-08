<#
.SYNOPSIS
  Windows entry point for the Phase E ROMBRL provisioning (delegates to WSL2).

.DESCRIPTION
  D4RL / mujoco-py are not installable on native Windows, so this wrapper converts the
  repo path to its WSL form and runs benchmark/rombri/provision.sh inside the default
  WSL distribution. Requires WSL2 (`wsl --install`) with a Linux distro.

.EXAMPLE
  pwsh benchmark/rombri/provision.ps1
  pwsh benchmark/rombri/provision.ps1 -DryRun
  pwsh benchmark/rombri/provision.ps1 -TorchIndex https://download.pytorch.org/whl/cu121
#>
[CmdletBinding()]
param(
  [switch]$DryRun,
  [string]$TorchIndex = "",
  [string]$Distro = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
  throw "WSL not found. Install WSL2 first: run 'wsl --install' in an elevated shell, reboot, then retry."
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$wslArgs = @()
if ($Distro) { $wslArgs += @("-d", $Distro) }

# Pass a forward-slash Windows path: backslashes get eaten by wsl.exe argument parsing.
$winPath = $here -replace '\\', '/'
$linuxPath = [string](& wsl @wslArgs wslpath -a "$winPath" 2>$null)
$linuxPath = $linuxPath.Trim()
if ([string]::IsNullOrWhiteSpace($linuxPath)) {
  throw "Could not convert '$winPath' to a WSL path. Is a WSL distro installed and running?"
}

$envBits = @()
if ($DryRun)     { $envBits += "ROMBRI_DRY_RUN=1" }
if ($TorchIndex) { $envBits += "ROMBRI_TORCH_INDEX=$TorchIndex" }
$envPrefix = if ($envBits.Count -gt 0) { ($envBits -join " ") + " " } else { "" }

$command = "${envPrefix}bash '$linuxPath/provision.sh'"
Write-Host "[provision.ps1] WSL path: $linuxPath"
Write-Host "[provision.ps1] command:  $command"

& wsl @wslArgs bash -lc $command
exit $LASTEXITCODE
