#!/usr/bin/env bash
# 학습 실행 — config.env(브랜치별) → config.local.env(내 PC, gitignore) 순으로 읽고 src/train.py 를 돌린다.
#   bash train.sh                 # 데이터 없으면 prepare_f9.py 먼저 → 학습
#   bash train.sh --resume        # 추가 인자는 train.py 로 그대로 넘어간다
#   MODEL_ID=... bash train.sh    # 환경변수로도 덮어쓸 수 있다
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUTF8=1
HERE="$PWD"

# ── 설정 읽기 ───────────────────────────────────────────────
[ -f config.env ] || { echo "config.env 가 없습니다 — 모델 브랜치(gpu/* · kaggle/*)에서 실행하거나 config.env.example 을 복사하세요"; exit 1; }
_env_before="$(env)"
set -a; source ./config.env; [ -f config.local.env ] && source ./config.local.env; set +a
# 셸에서 준 환경변수가 파일보다 우선 — 파일을 읽기 전 값으로 되돌린다
while IFS='=' read -r k v; do [[ "$k" =~ ^(MODEL_ID|RUN_NAME|OUTPUT_DIR|DATA_DIR|MAX_SEQ|BATCH|GRAD_ACCUM|LOAD_4BIT|COMPUTE_DTYPE|EPOCHS|LR|SAVE_STEPS|LORA_R|HF_PUSH_REPO|SEED|N_ROWS)$ ]] && export "$k=$v"; done <<< "$_env_before"
: "${MODEL_ID:?config.env 의 MODEL_ID 가 비어 있습니다}"
RUN_NAME="${RUN_NAME:-run1}"
DATA_DIR="${DATA_DIR:-data/f9}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"

# ── 서버(gpu-server)면 영속 경로로 — data·outputs 는 PERSIST_DIR, venv·HF 캐시는 SCRATCH_DIR ──
if [ "${TARGET:-local}" = "gpu-server" ]; then
  PERSIST_DIR="${PERSIST_DIR:-$HOME/univus-train-out}"; SCRATCH_DIR="${SCRATCH_DIR:-$HOME/.venv/univus-train}"
  mkdir -p "$PERSIST_DIR/data" "$PERSIST_DIR/outputs" "$PERSIST_DIR/logs"
  for d in data outputs; do
    if [ ! -L "$d" ]; then [ -e "$d" ] && mv "$d" "$PERSIST_DIR/${d}_moved_$(date +%s)"; ln -s "$PERSIST_DIR/$d" "$d"; fi
  done
  # 가중치 캐시만 overlay 로 — HF_HOME 을 바꾸면 로그인 토큰($HF_HOME/token)까지 overlay 로 가서 재시작 때 사라진다
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH_DIR/hf_cache}"
  [ -x "$SCRATCH_DIR/.venv/bin/python" ] || { echo "venv 가 없습니다 ($SCRATCH_DIR/.venv) — pod 재시작 후라면 정상입니다. 먼저:  bash setup.sh"; exit 1; }
  PY="$SCRATCH_DIR/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ]; then PY=python
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY=python; fi

# ── 데이터 ─────────────────────────────────────────────────
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "== 데이터 없음 → 합성 데이터 생성 (seed ${SEED:-42}, ${N_ROWS:-1500}건)"
  "$PY" src/prepare_f9.py --out "$DATA_DIR" --seed "${SEED:-42}" --n "${N_ROWS:-1500}"
fi

# ── 학습 ───────────────────────────────────────────────────
PUSH=(); [ -n "${HF_PUSH_REPO:-}" ] && PUSH=(--push_to_hub "$HF_PUSH_REPO")
echo "== 학습: $MODEL_ID → $OUTPUT_DIR  (4bit=${LOAD_4BIT:-1} ${COMPUTE_DTYPE:-bf16} max_seq=${MAX_SEQ:-2560})"
mkdir -p "$OUTPUT_DIR"
"$PY" src/train.py --model_id "$MODEL_ID" --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" \
  --load_4bit "${LOAD_4BIT:-1}" --compute_dtype "${COMPUTE_DTYPE:-bf16}" --max_seq "${MAX_SEQ:-2560}" \
  --batch "${BATCH:-1}" --grad_accum "${GRAD_ACCUM:-8}" --lora_r "${LORA_R:-16}" \
  --epochs "${EPOCHS:-3}" --lr "${LR:-2e-4}" --save_steps "${SAVE_STEPS:-200}" --seed "${SEED:-42}" \
  ${PUSH[@]+"${PUSH[@]}"} "$@" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
echo "== 끝. 다음:  bash eval.sh"
