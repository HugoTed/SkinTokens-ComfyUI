$ErrorActionPreference = "Stop"

$PluginRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $PluginRoot

Write-Host "[TokenRig] Installing worker in $PluginRoot"
python bootstrap.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "[TokenRig] Worker install complete."
