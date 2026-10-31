# 내 PC 데이터 서버 켜기 — .venv 의 python 으로 local_server/app.py 를 127.0.0.1:LOCAL_PORT 에 띄운다.  Windows PowerShell 5.1.
#   .\local_server\run_local.ps1                 # scripts\server.env 의 UNIVUS_API_KEY · LOCAL_PORT 를 읽는다
#   .\local_server\run_local.ps1 -DataDir data\f9 -ResultsDir results
# 먼저 데이터가 있어야 한다:  .\.venv\Scripts\python.exe src\prepare_f9.py --n 1500 --seed 42
param([string]$DataDir = "", [string]$ResultsDir = "")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host ".venv 가 없습니다 — 먼저 .\setup_local.ps1" -ForegroundColor Red; exit 1 }

$envFile = Join-Path $Root "scripts\server.env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -Encoding UTF8) {
        $t = ($line -split "#", 2)[0].Trim()
        if ($t -match "^(UNIVUS_API_KEY|LOCAL_PORT)=(.*)$") { Set-Item -Path ("env:" + $matches[1]) -Value $matches[2].Trim().Trim('"').Trim("'") }
    }
} else { Write-Host "scripts\server.env 가 없어 기본값(포트 8601, 인증 없음)으로 뜹니다" -ForegroundColor Yellow }
$env:PYTHONUTF8 = "1"
$pyArgs = @("local_server\app.py")
if ($DataDir) { $pyArgs += @("--data-dir", $DataDir) }
if ($ResultsDir) { $pyArgs += @("--results-dir", $ResultsDir) }
Write-Host "local_server 시작 — 포트 $(if ($env:LOCAL_PORT) { $env:LOCAL_PORT } else { 8601 }) · Ctrl+C 로 종료" -ForegroundColor Cyan
& $py @pyArgs
