#!/usr/bin/env bash
# Donut OCR 학습 — config.env → config.local.env → 셸 env 순으로 읽고 src/train_ocr.py 를 돌린다 (train.sh 와 같은 꼴).
#   bash train_ocr.sh --prepare               # 데이터 점검만 (src/prepare_ocr.py → $OCR_OUT_DIR/data_stats.md)
#   bash train_ocr.sh --smoke                 # 16행 · 4 step — 서버에서 가장 먼저: s/step · GPU 메모리
#   bash train_ocr.sh                         # 본 학습 → $OCR_OUT_DIR/model · run.json
#   BATCH=1 GRAD_ACCUM=16 bash train_ocr.sh   # OOM 이면 (셸 env 가 config.env 보다 우선)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUTF8=1
[ -f config.env ] || { echo "config.env 가 없습니다 — gpu/donut-ocr 브랜치에서 실행하세요"; exit 1; }
_env_before="$(env)"
set -a; source ./config.env; [ -f config.local.env ] && source ./config.local.env; set +a
while IFS='=' read -r k v; do [[ "$k" =~ ^(MODEL_ID|OCR_[A-Z_]+|HF_[A-Z_]+|IMAGE_H|IMAGE_W|MAX_LEN|BATCH|GRAD_ACCUM|LR|WARMUP_STEPS|MAX_TRAIN_ROWS|MAX_STEPS|TIME_BUDGET_MIN|SAVE_STEPS|EVAL_ROWS)$ ]] && export "$k=$v"; done <<< "$_env_before"
: "${MODEL_ID:?config.env 의 MODEL_ID 가 비어 있습니다}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_HUB_CACHE TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$(dirname "${HF_HUB_CACHE:-$HOME/.cache/x}")/datasets_cache}"   # 홈을 채우지 않게

# venv: OCR_PY → (HF_HUB_CACHE 옆) .venv → 저장소 .venv → python
PY=""
for c in "${OCR_PY:-}" "$(dirname "${HF_HUB_CACHE:-/nonexistent/x}")/.venv/bin/python" .venv/bin/python .venv/Scripts/python.exe; do
  [ -n "$c" ] && [ -x "$c" ] && { PY="$c"; break; }
done
if [ -z "$PY" ]; then
  [ "${TARGET:-local}" = "gpu-server" ] && { echo "venv python 을 못 찾았습니다 — OCR_PY=/경로/python bash train_ocr.sh"; exit 1; }
  PY=python
fi
mkdir -p "$OCR_OUT_DIR" "$OCR_CKPT_DIR"

if [ "${1:-}" = "--prepare" ]; then
  shift
  "$PY" src/prepare_ocr.py "$@" 2>&1 | tee -a "$OCR_OUT_DIR/prepare.log"
  exit 0
fi
LOG="$OCR_OUT_DIR/train_console.log"; for a in "$@"; do [ "$a" = "--smoke" ] && LOG="$OCR_OUT_DIR/smoke_console.log"; done
echo "== 학습: $MODEL_ID · $PY · 로그 $LOG"
"$PY" src/train_ocr.py "$@" 2>&1 | tee -a "$LOG"
echo "== 끝. 다음:  bash eval_ocr.sh"
