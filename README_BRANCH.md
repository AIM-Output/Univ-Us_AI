# gpu/hyperclovax-1.5b — HyperCLOVAX SEED 1.5B · 학교 GPU 서버 · bf16 LoRA

| 항목 | 값 |
| --- | --- |
| 모델 | `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` — 한국어 특화, 초경량. **HF gated(auto)**: 모델 페이지에서 약관 동의 + `HF_TOKEN` |
| 환경 | 학교 서버 A100 MIG `1g.10gb` — 가용 9.5GiB, bf16 |
| 방식 | **4-bit 없이** bf16 LoRA · r=16 · MAX_SEQ 2560 · **batch 1** × accum 8 · completion-only 손실 |
| 산출물 | `outputs/hcx-1.5b-f9/adapter/` |
| 주의 | 템플릿이 `tools=` 를 모를 수 있다 → `render_chat` 이 자동으로 **폴백 모드**(system 에 도구 JSON 주입). `train.py` 로그의 `템플릿 모드` 로 확인 |

## 메모리 예산

| 항목 | 크기 |
| --- | --- |
| 1.5B 가중치 bf16 | ~3GB |
| 활성화(2560 토큰 × batch 1) + logits + LoRA + 옵티마이저 | 3~5GB |
| **합계** | **6~8GB / 9.5GiB** ✅ (4-bit 불필요) |

**토큰 예산:** system + 도구 11개 정의 ≈ **1,950 토큰**(폴백 모드는 도구 JSON 이 system 에 들어가 비슷하거나 조금 더 길다) → `MAX_SEQ` 는 **2304 이상**.
`BATCH=2` 는 안 된다 — seq 2560 × batch 2 의 logits(어휘 크기 × 토큰 수, fp32) 가 9.5GiB 를 넘긴다.
OOM 이면 `LORA_R` 16→8 → 그래도면 `LOAD_4BIT=1`. `MAX_SEQ` 는 내리지 않는다.

## 왜 이 모델인가

GPU 문서 4장: 9.5GiB 에서 돌 수 있는 **한국어 특화 공개 모델**은 HyperCLOVA X SEED 계열뿐이다. 1.5B 는 tool calling 신뢰도가 낮을 수 있어
Qwen3-4B 와 **같은 평가셋으로 비교**하는 것이 이 브랜치의 목적이다. 수집 브랜치 `training/README.md` 가 적은 `…-Instruct-8B` 는 HF 에 없다(404).

## 실행 순서

```bash
huggingface-cli login     # 1회 — 토큰은 홈(~/.cache/huggingface)에 남아 재시작 후에도 유지
bash setup.sh             # gated 확인 → venv → 가중치
bash train.sh
bash eval.sh
```

영속 경로·홈 10GB 한도·`save_total_limit=2` 는 `gpu/qwen3-4b` 의 README_BRANCH 와 같다.
