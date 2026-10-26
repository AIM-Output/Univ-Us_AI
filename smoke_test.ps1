# 스모크 테스트 (Windows PowerShell 5.1) — smoke_test.sh 와 같은 일을 한다.
#   .\smoke_test.ps1
#   $env:SMOKE_MODEL_ID = "조직/작은모델"; .\smoke_test.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"

function Step($cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "STATUS=SMOKE_FAILED ($cmd)" -ForegroundColor Red; exit 1 }
}

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv 가 없습니다 — 먼저 .\setup_local.ps1" -ForegroundColor Red; exit 1 }
$model = if ($env:SMOKE_MODEL_ID) { $env:SMOKE_MODEL_ID } else { "Qwen/Qwen2.5-0.5B-Instruct" }
$out = "outputs\smoke"; $data = "data\f9_smoke"

$probe = & $py -c "import torch;c=torch.cuda.is_available();print(int(c),'bf16' if (c and torch.cuda.is_bf16_supported()) else ('fp16' if c else 'fp32'))"
$parts = $probe.Trim().Split(" ")
$cuda = $parts[0]; $dtype = $parts[1]
Write-Host "== 스모크: $model · cuda=$cuda dtype=$dtype"

Step "& $py src\common.py"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null
Step "& $py src\prepare_f9.py --smoke --out $data --seed 1"
Step "& $py src\train.py --model_id $model --data_dir $data --output_dir $out --load_4bit $cuda --compute_dtype $dtype --max_seq 2304 --batch 1 --grad_accum 1 --max_steps 8 --max_train_rows 40 --save_steps 4 --logging_steps 1 --lora_r 8"
Step "& $py src\eval.py --model_id $model --data $data\eval.jsonl --out $out\eval_base.md --label '학습 전' --limit 12 --max_new_tokens 96"
Step "& $py src\eval.py --model_id $model --adapter $out\adapter --data $data\eval.jsonl --out $out\eval_adapter.md --label '학습 후(8 step)' --limit 12 --max_new_tokens 96"

Write-Host ""
Write-Host "| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |"
Write-Host "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
Get-ChildItem "$out\summary_row_*.md" | ForEach-Object { Get-Content $_.FullName -Encoding UTF8 }
Write-Host ""
Write-Host "STATUS=SMOKE_OK ($out)" -ForegroundColor Green
