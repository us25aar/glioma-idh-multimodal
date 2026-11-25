param([switch]$InstallDependencies)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
    $InstallDependencies = $true
}
if ($InstallDependencies) {
    & $projectPython -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& $projectPython -m glioma_idh download
if ($LASTEXITCODE -ne 0) { throw 'Metadata download or verification failed.' }
& $projectPython scripts/run_checks.py
if ($LASTEXITCODE -ne 0) { throw 'Scientific validity checks failed.' }
& $projectPython -m glioma_idh week1 --config configs/utsw_week1.json
if ($LASTEXITCODE -ne 0) { throw 'Clinical experiment failed.' }
