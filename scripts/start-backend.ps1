param(
    [string]$Profile = "dev",
    [string]$ConfigFile = "config.yaml",
    [int]$Port = 8000,
    [int]$Workers = 1,
    [switch]$SeedDevUser
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $workspaceRoot

$configPath = if ([System.IO.Path]::IsPathRooted($ConfigFile)) {
    [System.IO.Path]::GetFullPath($ConfigFile)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $ConfigFile))
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Config file not found: $configPath"
}
if ($Workers -lt 1) {
    throw "Workers must be at least 1"
}

$env:FITNESS_CONFIG_FILE = $ConfigFile
$env:FITNESS_PROFILE = $Profile
Remove-Item Env:ENV_FILE -ErrorAction SilentlyContinue
Remove-Item Env:RAG_STARTUP_INDEX -ErrorAction SilentlyContinue

if ($SeedDevUser) {
    if ($Profile -ne "dev") {
        throw "-SeedDevUser can only be used with the dev profile"
    }
    $env:FITNESS_COACH_ALLOW_DEV_SEED = "1"
    python scripts/seed_dev_user.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} else {
    Remove-Item Env:FITNESS_COACH_ALLOW_DEV_SEED -ErrorAction SilentlyContinue
}

Write-Host "[start] Config: $configPath"
Write-Host "[start] Profile: $Profile"
Write-Host "[start] Backend: http://127.0.0.1:$Port"
Write-Host "[start] Workers: $Workers"
Write-Host "[start] Reload: disabled (systemd/gunicorn-like local run)"

# Gunicorn is Unix-oriented and is not a reliable Windows process manager.
# Uvicorn workers provide the equivalent local process model on Windows;
# production continues to use systemd + gunicorn.
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port --workers $Workers
