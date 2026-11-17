# SSH 터널 — 내 PC ↔ pod 사이의 유일한 길 (pod 는 SSH 30562 외에 열린 포트가 없다고 가정).  Windows PowerShell 5.1.
#   .\scripts\tunnel.ps1          # 끊기면 5초 뒤 다시 연결. Ctrl+C 로 종료
#   -L SERVER_PORT : 내 PC 의 127.0.0.1:8600 → pod 의 gpu_server      (WE-Meet .env LLM_MAIN_BASE_URL=http://127.0.0.1:8600/v1)
#   -R LOCAL_PORT  : pod 의 127.0.0.1:8601 → 내 PC 의 local_server     (학습 데이터 내려받기 · 결과 올리기)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
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
if ($cfg.Count -eq 0) { Write-Host "scripts\server.env 가 없습니다 — server.env.example 을 복사해서 채우세요" -ForegroundColor Red; exit 1 }
$Key = $cfg["SSH_KEY"]; if (-not [IO.Path]::IsPathRooted($Key)) { $Key = Join-Path $Root $Key }
$sp = if ($cfg["SERVER_PORT"]) { $cfg["SERVER_PORT"] } else { "8600" }
$lp = if ($cfg["LOCAL_PORT"]) { $cfg["LOCAL_PORT"] } else { "8601" }
$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$dest = "$($cfg['SSH_USER'])@$($cfg['SSH_HOST'])"

Write-Host "터널: 127.0.0.1:$sp → pod:$sp (gpu_server)   pod:$lp → 127.0.0.1:$lp (local_server)   Ctrl+C 로 종료" -ForegroundColor Cyan
while ($true) {
    $t = Get-Date -Format "HH:mm:ss"
    Write-Host "[$t] 연결 중 $dest ..."
    & $ssh -N -p $cfg["SSH_PORT"] -i $Key `
        -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes `
        -L "${sp}:127.0.0.1:${sp}" -R "${lp}:127.0.0.1:${lp}" $dest
    $code = $LASTEXITCODE
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 터널 끊김 (exit $code) — 5초 뒤 재연결. 포트가 이미 쓰인다면(ExitOnForwardFailure) 다른 터널을 먼저 끄세요" -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
