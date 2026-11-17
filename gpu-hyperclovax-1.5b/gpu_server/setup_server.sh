#!/usr/bin/env bash
# ============================================================
# GPU 서버(pod) 환경 — gpu_server/app.py 와 train.sh 가 **같은 venv** 를 쓴다.
#   처음 1회 + pod 재시작 후마다:   bash gpu_server/setup_server.sh
#   오래 걸리면 세션 분리:          setsid nohup bash gpu_server/setup_server.sh > ~/univus-train-setup.log 2>&1 < /dev/null &
#
# 서버는 Kubernetes pod (서버-작업환경-설정.md 1장, 2026-10-30 실측 홈 여유 3.1GB):
#   /home/jovyan   CephFS 10GB  남음   → PERSIST_DIR (data · outputs · models(병합본) · logs)
#   ~/.venv/*      overlay      사라짐 → SCRATCH_DIR (.venv · hf_cache · merge_tmp)
# sudo/apt 없음. 재시작 뒤 venv 와 가중치가 없는 것은 정상이고 이 스크립트가 다시 만든다.
# ============================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f config.env ] || { echo "config.env 가 없습니다 — 모델 브랜치(gpu/qwen3-4b 등)를 배포했는지 확인"; echo "STATUS=FAILED"; exit 1; }
set -a; source ./config.env; [ -f config.local.env ] && source ./config.local.env; set +a
PERSIST_DIR="${PERSIST_DIR:-$HOME/univus-train-out}"
SCRATCH_DIR="${SCRATCH_DIR:-$HOME/.venv/univus-train}"
VENV="$SCRATCH_DIR/.venv"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH_DIR/hf_cache}"     # 토큰(~/.cache/huggingface/token)은 홈에 그대로
# pip 휠 캐시(~/.cache/pip, torch 휠만 수 GB)와 hf-xet 청크 캐시(~/.cache/huggingface/xet, HF_HOME 을 따라감)가 홈 10GB 를 채운다 → overlay 로
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_DIR/pip_cache}" HF_XET_CACHE="${HF_XET_CACHE:-$SCRATCH_DIR/xet_cache}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"
fail() { echo; echo "!!! $1 !!!"; echo "STATUS=FAILED"; exit 1; }

echo "=============================================="
echo " 시작       : $(date '+%F %T')   브랜치 $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo " 모델       : ${MODEL_ID:-?}   run ${RUN_NAME:-?}"
echo " 영속(남음) : $PERSIST_DIR"
echo " 소멸(재설치): $SCRATCH_DIR"
echo "=============================================="

echo; echo "### [0/5] 경로 ###"
mkdir -p "$PERSIST_DIR"/{data,outputs,models,logs} "$SCRATCH_DIR" || fail "폴더 생성 실패"
for d in data outputs; do
  if [ ! -L "$d" ]; then [ -e "$d" ] && mv "$d" "$PERSIST_DIR/${d}_moved_$(date +%s)"; ln -s "$PERSIST_DIR/$d" "$d"; fi
  echo "  $d -> $(readlink -f "$d")"
done
df -h "$HOME" "$SCRATCH_DIR" 2>/dev/null | sed 's/^/  /'

echo; echo "### [1/5] venv ###"
if [ ! -x "$VENV/bin/python" ]; then echo "  venv 없음 → 새로 만든다 (pod 재시작 후라면 정상)"; python3 -m venv "$VENV" || fail "venv 생성 실패"; fi
"$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel || fail "pip 업그레이드 실패"
echo "  $("$VENV/bin/python" --version 2>&1)  @ $VENV"

echo; echo "### [2/5] torch (CUDA) ###"
if "$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "  이미 CUDA torch 있음: $("$VENV/bin/python" -c 'import torch;print(torch.__version__)')"
else
  "$VENV/bin/pip" install --index-url "$TORCH_INDEX" torch || fail "torch 설치 실패 ($TORCH_INDEX)"
fi

echo; echo "### [3/5] requirements (학습 + flask/requests) ###"
"$VENV/bin/pip" install -r requirements.txt nvitop || fail "requirements 설치 실패"

echo; echo "### [4/5] GPU 검증 ###"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | sed 's/^/  /'
"$VENV/bin/python" - <<'PY' || fail "GPU 검증 실패"
import importlib.metadata as md, torch
for n in ("torch","transformers","peft","trl","bitsandbytes","flask","requests"):
    try: print(f"  {n:14} {md.version(n)}")
    except md.PackageNotFoundError: print(f"  {n:14} 없음")
if not torch.cuda.is_available(): raise SystemExit("  !!! torch 가 GPU 를 못 본다")
free, total = torch.cuda.mem_get_info(); name = torch.cuda.get_device_name(0)
print(f"  GPU {name}  {free/1024**3:.1f}/{total/1024**3:.1f} GiB  bf16={torch.cuda.is_bf16_supported()}")
if total/1024**3 < 9: print("  ⚠️ 9GiB 미만 — config.env 의 BATCH·LORA_R 을 줄일 것")
x = torch.randn(512, 512, device="cuda"); (x @ x).sum().item(); print("  matmul OK")
PY

echo; echo "### [5/5] Hugging Face ###"
if [ "${HF_GATED:-0}" = "1" ]; then
  if [ -n "${HF_TOKEN:-}" ] || "$VENV/bin/python" -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
    echo "  HF 로그인 확인됨"
  else
    echo "  !!! ${MODEL_ID} 는 gated 모델 — https://huggingface.co/${MODEL_ID} 약관 동의 후  $VENV/bin/hf auth login  (토큰은 홈에 남음)"
    fail "HF 토큰 없음"
  fi
fi
if [ "${PREFETCH:-1}" = "1" ] && [ -n "${MODEL_ID:-}" ]; then
  echo "  가중치 미리 받기 → $HF_HUB_CACHE (overlay — 재시작 후 다시 받는 게 정상)"
  "$VENV/bin/python" - <<PY || fail "가중치 다운로드 실패 (네트워크·토큰·약관)"
from huggingface_hub import snapshot_download
print("  ", snapshot_download("${MODEL_ID}", ignore_patterns=["*.pth", "*.bin", "original/*", "*.gguf", "*.onnx"]))
PY
fi

echo
echo "=============================================="
echo " overlay: venv $(du -sh "$VENV" 2>/dev/null | cut -f1) · HF 캐시 $(du -sh "$HF_HUB_CACHE" 2>/dev/null | cut -f1) · pip 캐시 $(du -sh "$PIP_CACHE_DIR" 2>/dev/null | cut -f1) · xet 캐시 $(du -sh "$HF_XET_CACHE" 2>/dev/null | cut -f1)"
echo " 홈(영속) — 10GB 한도, 병합본 nf4 ≈2.8GB 가 들어가려면 3GB 이상 비어 있어야 한다:"
df -h "$HOME" | sed 's/^/   /'
if [ "$(df -Pk "$HOME" | awk 'NR==2{print $4}')" -lt 3145728 ]; then
  echo " ⚠️ 홈 여유가 3GB 미만 — /model/merge 는 병합본을 overlay(MODELS_FALLBACK_DIR=$SCRATCH_DIR/models)에 두고, pod 재시작 후 다시 만들어야 한다."
  echo "    비울 곳:  du -xsh ~/.cache/* ~/.cache/huggingface/* ~/univus-train-out/outputs/*/checkpoint-* 2>/dev/null | sort -rh | head"
fi
echo " STATUS=DONE   $(date '+%F %T')"
echo
echo " 서버 켜기:  bash gpu_server/run_server.sh start      (= nohup $VENV/bin/python gpu_server/app.py --port ${SERVER_PORT:-8600} > $PERSIST_DIR/logs/server.log 2>&1 &)"
echo " API 키   :  config.local.env 에 UNIVUS_API_KEY=... 한 줄 (내 PC 의 scripts/server.env 와 같은 값)"
echo "=============================================="
