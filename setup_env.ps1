# Tao va cai dat mot moi truong Python 3.12 tai thu muc goc.
$ErrorActionPreference = "Stop"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Khong tim thay Python Launcher (py). Hay cai Python 3.12 va bat Add to PATH."
}
$version = & py -3.12 --version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Khong tim thay Python 3.12. Phien ban hien tai: $version"
}
Write-Host "Su dung $version"
if (-not (Test-Path ".venv")) { & py -3.12 -m venv .venv }
$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
Write-Host "`nDa cai dat xong trong .venv (Git da bo qua thu muc nay)."
Write-Host "Kich hoat: .\.venv\Scripts\Activate.ps1"
Write-Host "Kiem tra: python --version"
