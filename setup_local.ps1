# 로컬(Windows) 개발 환경 — .venv 생성 + torch(CUDA) + requirements.  RTX 4060 8GB 기준으로 스모크/소형 실험용.
#   .\setup_local.ps1
#   .\setup_local.ps1 -Python 3.12      # 쓸 파이썬 버전 지정 (py 런처)
#
# torch 는 파이썬 버전에 따라 받을 수 있는 CUDA 휠이 다르다 (2026-10 확인):
#   3.12 이상 → cu129 index 의 최신 torch
#   3.10/3.11 → cu128 index 의 torch 2.9.1 까지 (그 뒤 버전은 cp310 휠이 없다)
# --index-url 을 쓴다 (--extra-index-url 이면 PyPI 의 CPU 휠로 떨어질 수 있다).
param([string]$Python = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"

function Find-Python {
    param([string]$Want)
    $cands = if ($Want) { @($Want) } else { @("3.12", "3.13", "3.11", "3.10") }
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"      # py 런처가 없는 버전은 stderr 에 쓴다 — Stop 이면 여기서 끊긴다
    try {
        foreach ($v in $cands) {
            $null = & py "-$v" -c "import sys" 2>&1
            if ($LASTEXITCODE -eq 0) { return $v }
        }
    } finally { $ErrorActionPreference = $old }
    return $null
}

$ver = Find-Python $Python
if (-not $ver) {
    Write-Host "py 런처로 3.10 이상 파이썬을 찾지 못했습니다. https://www.python.org 에서 3.12 를 설치하고 다시 실행하세요." -ForegroundColor Red
    exit 1
}
Write-Host "== Python $ver 사용"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    & py "-$ver" -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit 1 }
    Write-Host "   .venv 생성"
} else { Write-Host "   .venv 이미 있음" }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) { exit 1 }

$minor = [int]$ver.Split(".")[1]
if ($minor -ge 12) {
    Write-Host "== torch (cu129, 최신)"
    & $py -m pip install --index-url https://download.pytorch.org/whl/cu129 torch
} else {
    Write-Host "== torch 2.9.1 (cu128) — Python 3.$minor 은 그 뒤 CUDA 휠이 없다. 최신 torch 가 필요하면 3.12 설치 후 -Python 3.12"
    & $py -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1"
}
if ($LASTEXITCODE -ne 0) { Write-Host "torch 설치 실패 — 파이썬 버전과 CUDA 드라이버를 확인하세요" -ForegroundColor Red; exit 1 }

Write-Host "== requirements.txt"
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== 확인"
& $py -c @"
import importlib.metadata as md, torch
for n in ('torch','transformers','peft','trl','datasets','accelerate','bitsandbytes','huggingface_hub'):
    try: print(f'  {n:16} {md.version(n)}')
    except md.PackageNotFoundError: print(f'  {n:16} 없음')
print('  cuda.is_available()', torch.cuda.is_available())
if torch.cuda.is_available():
    f,t = torch.cuda.mem_get_info(); print(f'  GPU {torch.cuda.get_device_name(0)}  {f/1024**3:.1f}/{t/1024**3:.1f} GiB  bf16={torch.cuda.is_bf16_supported()}')
"@
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host ""
Write-Host "STATUS=DONE   다음:  .\smoke_test.ps1" -ForegroundColor Green
