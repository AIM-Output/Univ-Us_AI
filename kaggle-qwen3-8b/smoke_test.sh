#!/usr/bin/env bash
# 스모크 테스트 — 파이프라인이 끝까지 도는지만 본다 (정확도는 보지 않는다).
#   60건 합성 → 작은 모델 8 step 학습 → 평가 12건 × 2회(어댑터 없이/있게) → 비교표
#   bash smoke_test.sh
#   SMOKE_MODEL_ID=조직/작은모델 bash smoke_test.sh     # 기본값은 0.5B 급 instruct (config.local.env 로도 가능)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUTF8=1
[ -f config.local.env ] && { set -a; source ./config.local.env; set +a; }
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=python
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -n "${SCRATCH_DIR:-}" ] && [ -x "$SCRATCH_DIR/.venv/bin/python" ]; then PY="$SCRATCH_DIR/.venv/bin/python"
else PY=python; fi
MODEL="${SMOKE_MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT="outputs/smoke"; DATA="data/f9_smoke"

read -r CUDA DTYPE <<< "$("$PY" -c 'import torch;c=torch.cuda.is_available();print(int(c),"bf16" if (c and torch.cuda.is_bf16_supported()) else ("fp16" if c else "fp32"))')"
echo "== 스모크: $MODEL · cuda=$CUDA dtype=$DTYPE · python=$PY"
"$PY" src/common.py
rm -rf "$OUT"; mkdir -p "$OUT"
"$PY" src/prepare_f9.py --smoke --out "$DATA" --seed 1
"$PY" src/train.py --model_id "$MODEL" --data_dir "$DATA" --output_dir "$OUT" \
  --load_4bit "$CUDA" --compute_dtype "$DTYPE" --max_seq 2304 --batch 1 --grad_accum 1 \
  --max_steps 8 --max_train_rows 40 --save_steps 4 --logging_steps 1 --lora_r 8
"$PY" src/eval.py --model_id "$MODEL" --data "$DATA/eval.jsonl" --out "$OUT/eval_base.md" --label "학습 전" --limit 12 --max_new_tokens 96
"$PY" src/eval.py --model_id "$MODEL" --adapter "$OUT/adapter" --data "$DATA/eval.jsonl" --out "$OUT/eval_adapter.md" --label "학습 후(8 step)" --limit 12 --max_new_tokens 96
echo; echo "| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |"
echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
cat "$OUT"/summary_row_*.md
echo; echo "STATUS=SMOKE_OK  ($OUT)"
