param(
  [switch]$InstallExtras
)

$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
if (-not (Test-Path $python)) {
  $command = Get-Command python -ErrorAction SilentlyContinue
  if ($command -and $command.Source -notmatch 'WindowsApps') {
    $python = $command.Source
  }
}

if (-not (Test-Path $python)) {
  throw "Python 3.12 was not found. Install it or update scripts/bootstrap.ps1 with your local path."
}

if (-not (Test-Path '.venv')) {
  & $python -m venv .venv
}

$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $venvPython -m pip install --upgrade pip

if ($InstallExtras) {
  & $venvPython -m pip install -e '.[translate,tts,dev]'
} else {
  & $venvPython -m pip install -e .
}

Write-Host "Bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1"
