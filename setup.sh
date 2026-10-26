#!/usr/bin/env bash
# ============================================================
# 학교 GPU 서버 학습 환경 — 처음 1회 + **pod 재시작 후마다** 다시 실행한다.
#
#   bash setup.sh
#   PREFETCH=0 bash setup.sh                 가중치 미리 받기 생략 (train 때 받음)
#
# 서버는 Kubernetes pod (서버-작업환경-설정.md 1장):
#   /home/jovyan        CephFS 10GB   — 남음   → PERSIST_DIR (data · outputs · logs)
#   ~/.venv/*           overlay       — 사라짐 → SCRATCH_DIR (.venv ~8GB · hf_cache 가중치)
# 재시작 뒤 venv 와 가중치가 없는 것은 **정상**이다. 이 스크립트가 다시 만든다. sudo/apt 는 쓰지 않는다.
#
# 오래 걸립니다(torch ~3GB + 가중치). 세션과 분리해서:
#   setsid nohup bash setup.sh > ~/univus-train-setup.log 2>&1 < /dev/null &
#   tail -f ~/univus-train-setup.log        (STATUS=DONE 이면 성공, FAILED 면 실패)
# ============================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; source ./config.env; [ -f config.local.env ] && source ./config.local.env; set +a
PERSIST_DIR="${PERSIST_DIR:-$HOME/univus-train-out}"
SCRATCH_DIR="${SCRATCH_DIR:-$HOME/.venv/univus-train}"
VENV="$SCRATCH_DIR/.venv"
export HF_HOME="$SCRATCH_DIR/hf_cache"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"   # 팀 requirements-gpu.txt 와 같은 CUDA 세대

fail() { echo; echo "!!! $1 !!!"; echo "STATUS=FAILED"; exit 1; }

echo "=============================================="
echo " 시작      : $(date '+%F %T')"
echo " 브랜치    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')  ·  모델 ${MODEL_ID:-?}"
echo " 영속(남음): $PERSIST_DIR"
echo " 소멸(재설치): $SCRATCH_DIR"
echo "=============================================="

# ── 0. 경로 ─────────────────────────────────────────────────
echo; echo "### [0/5] 경로 ###"
mkdir -p "$PERSIST_DIR/data" "$PERSIST_DIR/outputs" "$PERSIST_DIR/logs" "$SCRATCH_DIR" || fail "폴더 생성 실패"
for d in data outputs; do
  if [ ! -L "$d" ]; then [ -e "$d" ] && mv "$d" "$PERSIST_DIR/${d}_moved_$(date +%s)"; ln -s "$PERSIST_DIR/$d" "$d"; fi
  echo "  $d -> $(readlink -f "$d")"
done
if [ -d "$VENV" ]; then echo "  venv 이미 있음: $VENV"; else echo "  venv 없음 → 새로 만든다 (pod 재시작 후라면 정상)"; fi
df -h "$HOME" "$SCRATCH_DIR" 2>/dev/null | sed 's/^/  /'

# ── 1. venv ─────────────────────────────────────────────────
echo; echo "### [1/5] venv ###"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || fail "venv 생성 실패"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel || fail "pip 업그레이드 실패"
echo "  python $("$VENV/bin/python" --version 2>&1)"

# ── 2. torch (CUDA) ─────────────────────────────────────────
echo; echo "### [2/5] torch ###"
if "$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "  이미 CUDA torch 있음: $("$VENV/bin/python" -c 'import torch;print(torch.__version__)')"
else
  "$VENV/bin/pip" install --index-url "$TORCH_INDEX" torch || fail "torch 설치 실패 ($TORCH_INDEX)"
fi

# ── 3. 패키지 ───────────────────────────────────────────────
echo; echo "### [3/5] requirements ###"
"$VENV/bin/pip" install -r requirements.txt nvitop || fail "requirements 설치 실패"

# ── 4. GPU 검증 ─────────────────────────────────────────────
echo; echo "### [4/5] GPU ###"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null | sed 's/^/  /'
[ -x "$VENV/bin/nvitop" ] && "$VENV/bin/nvitop" -1 2>/dev/null | head -8 | sed 's/^/  /'
"$VENV/bin/python" - <<'PY' || fail "GPU 검증 실패"
import importlib.metadata as md, torch
for n in ("torch","transformers","peft","trl","datasets","accelerate","bitsandbytes","huggingface_hub"):
    try: print(f"  {n:16} {md.version(n)}")
    except md.PackageNotFoundError: print(f"  {n:16} 없음")
if not torch.cuda.is_available():
    raise SystemExit("  !!! torch 가 GPU 를 못 본다 — nvidia-smi · 드라이버 · TORCH_INDEX 확인")
free, total = torch.cuda.mem_get_info()
name = torch.cuda.get_device_name(0)
print(f"  GPU   {name}")
print(f"  VRAM  {free/1024**3:.1f} GiB free / {total/1024**3:.1f} GiB total  ·  bf16={torch.cuda.is_bf16_supported()}")
if "MIG" in name: print("  MIG 슬라이스 — 카드 전체가 아니라 위 total 만큼만 쓸 수 있다")
if total/1024**3 < 9: print("  ⚠️ 9GiB 미만 — config.env 의 MAX_SEQ 를 줄이거나 LOAD_4BIT=1 로")
if not torch.cuda.is_bf16_supported(): print("  ⚠️ bf16 미지원 GPU — config.env COMPUTE_DTYPE=fp16 로 바꿀 것")
x = torch.randn(1024, 1024, device="cuda"); (x @ x).sum().item(); print("  matmul OK")
PY

# ── 5. HF 토큰·가중치 ─────────────────────────────────────────
echo; echo "### [5/5] Hugging Face ###"
if [ "${HF_GATED:-0}" = "1" ]; then
  if [ -n "${HF_TOKEN:-}" ] || "$VENV/bin/python" -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
    echo "  HF 로그인 확인 — gated 모델 ${MODEL_ID} 접근 가능 여부는 다운로드 때 판정된다"
  else
    echo "  !!! ${MODEL_ID} 는 gated 모델입니다. 아래를 먼저 하세요:"
    echo "      1) https://huggingface.co/${MODEL_ID} 에서 약관 동의(Agree)"
    echo "      2) 토큰 발급(read) → 아래 중 하나"
    echo "         $VENV/bin/huggingface-cli login            (홈에 저장돼 재시작 후에도 남음)"
    echo "         echo 'HF_TOKEN=hf_xxx' >> config.local.env  (gitignore)"
    fail "HF 토큰 없음 (gated 모델)"
  fi
fi
if [ "${PREFETCH:-1}" = "1" ] && [ -n "${MODEL_ID:-}" ]; then
  echo "  가중치 미리 받기 → $HF_HOME  (재시작 후면 다시 받는 게 정상)"
  "$VENV/bin/python" - <<PY || fail "가중치 다운로드 실패 (네트워크·토큰·약관 확인)"
from huggingface_hub import snapshot_download
p = snapshot_download("${MODEL_ID}", ignore_patterns=["*.pth", "*.bin", "original/*", "*.gguf", "*.onnx"])
print("  ", p)
PY
fi

echo
echo "=============================================="
echo " venv 크기 : $(du -sh "$VENV" 2>/dev/null | cut -f1)   HF 캐시: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
echo " 홈 남은 용량: $(df -h "$HOME" | awk 'NR==2{print $4}')  (10GB 한도 — outputs 체크포인트는 save_total_limit=2)"
echo " STATUS=DONE"
echo " 종료: $(date '+%F %T')"
echo
echo " 다음:  bash train.sh        (세션 분리:  setsid nohup bash train.sh > $PERSIST_DIR/logs/train.log 2>&1 < /dev/null &)"
echo "=============================================="
