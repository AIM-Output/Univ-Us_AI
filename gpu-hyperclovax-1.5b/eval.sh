#!/usr/bin/env bash
# 평가 — 같은 eval.jsonl 로 (1) 베이스만 (2) 베이스+어댑터 를 채점해 compare.md 한 표로 모은다.
#   bash eval.sh                    # outputs/<RUN_NAME>/eval_base.md · eval_adapter.md · compare.md
#   bash eval.sh --limit 50         # 추가 인자는 eval.py 로
#   RUN_NAME=run2 bash eval.sh      # 환경변수가 config.env 보다 우선
#   BASE_MODEL_ID=조직/모델-Base bash eval.sh   # 3자 비교의 'Base 모델' 행까지 (선택)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUTF8=1
[ -f config.env ] || { echo "config.env 가 없습니다 — 모델 브랜치에서 실행하세요"; exit 1; }
_env_before="$(env)"
set -a; source ./config.env; [ -f config.local.env ] && source ./config.local.env; set +a
# 셸에서 준 환경변수가 파일보다 우선 — 파일을 읽기 전 값으로 되돌린다 (train.sh 와 같은 규칙)
while IFS='=' read -r k v; do [[ "$k" =~ ^(MODEL_ID|RUN_NAME|OUTPUT_DIR|DATA_DIR|LOAD_4BIT|COMPUTE_DTYPE|BASE_MODEL_ID)$ ]] && export "$k=$v"; done <<< "$_env_before"
: "${MODEL_ID:?MODEL_ID 비어 있음}"
RUN_NAME="${RUN_NAME:-run1}"; DATA_DIR="${DATA_DIR:-data/f9}"; OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
if [ "${TARGET:-local}" = "gpu-server" ]; then
  SCRATCH_DIR="${SCRATCH_DIR:-$HOME/.venv/univus-train}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH_DIR/hf_cache}"     # 토큰은 ~/.cache/huggingface/token 에 그대로 (HF_HOME 은 건드리지 않는다)
  export HF_XET_CACHE="${HF_XET_CACHE:-$SCRATCH_DIR/xet_cache}"     # xet 청크 캐시는 HF_HOME(홈)을 따라가므로 따로 overlay 로
  [ -x "$SCRATCH_DIR/.venv/bin/python" ] || { echo "venv 가 없습니다 ($SCRATCH_DIR/.venv) — pod 재시작 후라면 정상입니다. 먼저:  bash setup.sh"; exit 1; }
  PY="$SCRATCH_DIR/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ]; then PY=python
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY=python; fi
[ -f "$DATA_DIR/eval.jsonl" ] || { echo "$DATA_DIR/eval.jsonl 없음 — bash train.sh 가 먼저 만든다"; exit 1; }
ADAPTER="$OUTPUT_DIR/adapter"
# 학습과 같은 적재 방식으로 채점한다 (hyperclovax 는 LOAD_4BIT=0 → bf16)
COMMON=(--data "$DATA_DIR/eval.jsonl" --load_4bit "${LOAD_4BIT:-1}" --compute_dtype "${COMPUTE_DTYPE:-auto}" "$@")

rows=(); files=()
if [ -n "${BASE_MODEL_ID:-}" ]; then
  echo "== (0/2) Base 모델: $BASE_MODEL_ID"
  "$PY" src/eval.py --model_id "$BASE_MODEL_ID" --out "$OUTPUT_DIR/eval_basemodel.md" --label "학습 전 Base" "${COMMON[@]}"
  rows+=("$OUTPUT_DIR/summary_row_학습_전_Base.md"); files+=(eval_basemodel.md)
fi
echo "== (1/2) 학습 전: $MODEL_ID"
"$PY" src/eval.py --model_id "$MODEL_ID" --out "$OUTPUT_DIR/eval_base.md" --label "학습 전 Instruct" "${COMMON[@]}"
rows+=("$OUTPUT_DIR/summary_row_학습_전_Instruct.md"); files+=(eval_base.md)
if [ -d "$ADAPTER" ]; then
  echo "== (2/2) 학습 후: $MODEL_ID + $ADAPTER"
  "$PY" src/eval.py --model_id "$MODEL_ID" --adapter "$ADAPTER" --out "$OUTPUT_DIR/eval_adapter.md" --label "학습 후" "${COMMON[@]}"
  rows+=("$OUTPUT_DIR/summary_row_학습_후.md"); files+=(eval_adapter.md)
else
  echo "   어댑터 없음($ADAPTER) — 학습 전 행만 만든다"
fi

{
  echo "# F9 비교표 — $RUN_NAME ($MODEL_ID)"; echo
  echo "> $(date '+%F %T') · 같은 평가셋 \`$DATA_DIR/eval.jsonl\` · 같은 프롬프트(render_chat) · greedy · 모든 assistant 턴(teacher forcing)"; echo
  echo "| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |"
  echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
  cat "${rows[@]}"
  echo; echo "상세·도구별·실패 사례: ${files[*]} (같은 폴더, .json 도 함께)"
} > "$OUTPUT_DIR/compare.md"
echo; cat "$OUTPUT_DIR/compare.md"
