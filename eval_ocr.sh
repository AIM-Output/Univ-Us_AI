#!/usr/bin/env bash
# Donut OCR 평가 — 같은 validation 행으로 학습 전(MODEL_ID) · 학습 후($OCR_OUT_DIR/model) 를 채점해 compare.md 로.
#   bash eval_ocr.sh                     # base → finetuned → compare 전부
#   bash eval_ocr.sh --batch 2           # 추가 인자는 매 평가에 그대로 (OOM 이면 --batch 2 → 1)
#   bash eval_ocr.sh base|finetuned|compare [인자…]   # 한 단계만
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUTF8=1
[ -f config.env ] || { echo "config.env 가 없습니다 — gpu/donut-ocr 브랜치에서 실행하세요"; exit 1; }
_env_before="$(env)"
set -a; source ./config.env; [ -f config.local.env ] && source ./config.local.env; set +a
while IFS='=' read -r k v; do [[ "$k" =~ ^(MODEL_ID|OCR_[A-Z_]+|HF_[A-Z_]+|IMAGE_H|IMAGE_W|MAX_LEN|EVAL_ROWS)$ ]] && export "$k=$v"; done <<< "$_env_before"
: "${MODEL_ID:?config.env 의 MODEL_ID 가 비어 있습니다}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_HUB_CACHE TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$(dirname "${HF_HUB_CACHE:-$HOME/.cache/x}")/datasets_cache}"   # 홈을 채우지 않게

PY=""
for c in "${OCR_PY:-}" "$(dirname "${HF_HUB_CACHE:-/nonexistent/x}")/.venv/bin/python" .venv/bin/python .venv/Scripts/python.exe; do
  [ -n "$c" ] && [ -x "$c" ] && { PY="$c"; break; }
done
if [ -z "$PY" ]; then
  [ "${TARGET:-local}" = "gpu-server" ] && { echo "venv python 을 못 찾았습니다 — OCR_PY=/경로/python bash eval_ocr.sh"; exit 1; }
  PY=python
fi
mkdir -p "$OCR_OUT_DIR"
LOG="$OCR_OUT_DIR/eval_console.log"
FT="$OCR_OUT_DIR/model"

STAGE="all"
case "${1:-}" in base|finetuned|compare) STAGE="$1"; shift ;; esac

if [ "$STAGE" = all ] || [ "$STAGE" = base ]; then
  echo "== (1/3) 학습 전: $MODEL_ID"
  "$PY" src/eval_ocr.py --model "$MODEL_ID" --tag base "$@" 2>&1 | tee -a "$LOG"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = finetuned ]; then
  [ -f "$FT/config.json" ] || { echo "학습된 모델이 없습니다: $FT — bash train_ocr.sh 가 먼저"; exit 1; }
  echo "== (2/3) 학습 후: $FT"
  "$PY" src/eval_ocr.py --model "$FT" --tag finetuned "$@" 2>&1 | tee -a "$LOG"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = compare ]; then
  echo "== (3/3) 비교"
  "$PY" src/eval_ocr.py --compare 2>&1 | tee -a "$LOG"
fi
