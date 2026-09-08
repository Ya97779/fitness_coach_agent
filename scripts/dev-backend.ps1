param(
    [string]$Profile = "dev",
    [string]$ConfigFile = "config.yaml",
    [string]$EnvFile,
    [switch]$WithRagIndex
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $workspaceRoot

# config.yaml owns profile selection, non-secret settings, and feature flags.
# This wrapper only selects the profile and starts the loopback-only server.
$configPath = if ([System.IO.Path]::IsPathRooted($ConfigFile)) {
    [System.IO.Path]::GetFullPath($ConfigFile)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $ConfigFile))
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Config file not found: $configPath."
}

$env:FITNESS_CONFIG_FILE = $ConfigFile
$env:FITNESS_PROFILE = $Profile

# ENV_FILE remains a legacy/CI escape hatch. Normal selection is by profile.
if ($EnvFile) {
    $env:ENV_FILE = $EnvFile
} else {
    Remove-Item Env:ENV_FILE -ErrorAction SilentlyContinue
}

if ($WithRagIndex) {
    $env:RAG_STARTUP_INDEX = "true"
} else {
    Remove-Item Env:RAG_STARTUP_INDEX -ErrorAction SilentlyContinue
}

if ($Profile -eq "dev" -and -not $EnvFile) {
    $env:FITNESS_COACH_ALLOW_DEV_SEED = "1"
    python scripts/seed_dev_user.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} else {
    Remove-Item Env:FITNESS_COACH_ALLOW_DEV_SEED -ErrorAction SilentlyContinue
    Write-Host "[dev] Skip development user seeding (profile=$Profile, explicit env file=$([bool]$EnvFile))"
}

Write-Host "[dev] Backend: http://127.0.0.1:8000"
Write-Host "[dev] API docs: http://127.0.0.1:8000/docs"
Write-Host "[dev] Config: $configPath"
Write-Host "[dev] Profile: $Profile"
Write-Host "[dev] RAG startup indexing override: $([bool]$WithRagIndex)"
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
