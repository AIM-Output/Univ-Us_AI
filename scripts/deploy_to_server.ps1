# 현재 브랜치의 **추적 파일**을 pod 의 REMOTE_DIR 로 보낸다 (git archive → scp → tar).  Windows PowerShell 5.1.
#   .\scripts\deploy_to_server.ps1                     # 모델 브랜치(gpu/*)에서 실행
#   .\scripts\deploy_to_server.ps1 -IncludeLocalEnv    # config.local.env(UNIVUS_API_KEY·HF_TOKEN) 도 같이 보낸다
#   .\scripts\deploy_to_server.ps1 -AllowCommon        # main/common 브랜치여도 보낸다 (config.env 가 없어 서버는 못 돈다)
# 설정은 scripts\server.env (없으면 server.env.example 을 복사해서 채운다). rsync 가 없어 tar 한 덩어리로 보낸다.
param([switch]$IncludeLocalEnv, [switch]$AllowCommon)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Read-EnvFile($path) {
    $h = @{}
    if (-not (Test-Path $path)) { return $h }
    foreach ($line in Get-Content $path -Encoding UTF8) {
        $t = ($line -split "#", 2)[0].Trim()
        if ($t -match "^([A-Za-z_][A-Za-z0-9_]*)=(.*)$") { $h[$matches[1]] = $matches[2].Trim().Trim('"').Trim("'") }
    }
    return $h
}
$cfg = Read-EnvFile (Join-Path $Root "scripts\server.env")
if ($cfg.Count -eq 0) { Write-Host "scripts\server.env 가 없습니다 — copy scripts\server.env.example scripts\server.env 후 채우세요" -ForegroundColor Red; exit 1 }
$SshHost = $cfg["SSH_HOST"]; $SshPort = $cfg["SSH_PORT"]; $SshUser = $cfg["SSH_USER"]
$Key = $cfg["SSH_KEY"]; if (-not [IO.Path]::IsPathRooted($Key)) { $Key = Join-Path $Root $Key }
$Remote = if ($cfg["REMOTE_DIR"]) { $cfg["REMOTE_DIR"] } else { "~/univus-train" }
$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"; $scp = "C:\Windows\System32\OpenSSH\scp.exe"
if (-not (Test-Path $Key)) { Write-Host "SSH 키가 없습니다: $Key" -ForegroundColor Red; exit 1 }

$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
if (($branch -eq "main" -or $branch -eq "common") -and -not $AllowCommon) {
    Write-Host "지금 브랜치가 '$branch' 입니다. 서버에는 config.env 가 있는 모델 브랜치(gpu/qwen3-4b 등)를 보내세요. 그래도 보내려면 -AllowCommon" -ForegroundColor Yellow
    exit 1
}
$dirty = (& git status --porcelain --untracked-files=no)
if ($dirty) { Write-Host "⚠️ 커밋되지 않은 변경이 있습니다 — git archive 는 HEAD(커밋된 내용)만 보냅니다" -ForegroundColor Yellow }

$tar = Join-Path $env:TEMP ("univus-train-" + ($branch -replace "[^A-Za-z0-9]", "_") + ".tar")
Write-Host "== git archive HEAD ($branch) → $tar"
& git archive --format=tar -o $tar HEAD
if ($LASTEXITCODE -ne 0) { exit 1 }

$dest = "$SshUser@$SshHost"
Write-Host "== scp → ${dest}:/tmp/univus-train.tar"
& $scp -P $SshPort -i $Key -o StrictHostKeyChecking=accept-new $tar "${dest}:/tmp/univus-train.tar"
if ($LASTEXITCODE -ne 0) { Write-Host "scp 실패" -ForegroundColor Red; exit 1 }

$remoteCmd = "mkdir -p $Remote && tar -xf /tmp/univus-train.tar -C $Remote && rm -f /tmp/univus-train.tar && cd $Remote && chmod +x *.sh gpu_server/*.sh 2>/dev/null; echo '$branch' > .deployed_branch && echo DEPLOYED && ls"
Write-Host "== 풀기 → $Remote"
& $ssh -p $SshPort -i $Key -o StrictHostKeyChecking=accept-new $dest $remoteCmd
if ($LASTEXITCODE -ne 0) { Write-Host "원격 풀기 실패" -ForegroundColor Red; exit 1 }

if ($IncludeLocalEnv) {
    $le = Join-Path $Root "config.local.env"
    if (Test-Path $le) {
        Write-Host "== config.local.env 전송 (UNIVUS_API_KEY · HF_TOKEN 등)"
        & $scp -P $SshPort -i $Key $le "${dest}:$Remote/config.local.env"
    } else { Write-Host "config.local.env 가 없어 건너뜀 — UNIVUS_API_KEY=... 한 줄이라도 만들어 두면 서버 인증이 켜집니다" -ForegroundColor Yellow }
}
Remove-Item $tar -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "다음 (pod 에서):  ssh -p $SshPort -i $Key $dest" -ForegroundColor Green
Write-Host "   cd $Remote; bash gpu_server/setup_server.sh      # 처음 1회 + pod 재시작 후" -ForegroundColor Green
Write-Host "   bash gpu_server/run_server.sh start" -ForegroundColor Green
Write-Host "그다음 (내 PC):   .\scripts\tunnel.ps1  →  .\local_server\run_local.ps1" -ForegroundColor Green
