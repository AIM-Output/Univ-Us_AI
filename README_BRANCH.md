# kaggle/qwen3-8b — Qwen3-8B · 캐글 T4×2 · QLoRA (fp16)

| 항목 | 값 |
| --- | --- |
| 모델 | `Qwen/Qwen3-8B` — 9.5GiB 서버에서는 학습이 안 되는 **정확도 상한** 후보 |
| 환경 | 캐글 노트북 GPU **T4 × 2 (16GB each)**. 세션 12h · 주 30h · 종료 시 디스크 초기화 |
| 방식 | 4-bit(nf4) QLoRA · r=16 · **fp16 compute** · MAX_SEQ 2560 · batch 1 × accum 8 · `SAVE_STEPS=50` · completion-only 손실 |
| 산출물 | HF model repo `HF_PUSH_REPO` (private) — **/kaggle/working 은 세션과 함께 사라진다** |

## 메모리 예산

| 항목 | 크기 |
| --- | --- |
| 8B 가중치 4-bit | ~5GB |
| 활성화(2560 토큰, GC 켬) + 오버헤드 + LoRA/옵티마이저 | 4~6GB |
| **합계** | **9~11GB / 16GB** ✅ (한 장만 쓴다 — 소형 모델 학습은 분할 불필요, 개발계획서 4.4) |

**토큰 예산:** system + 도구 11개 정의 ≈ **1,950 토큰** → `MAX_SEQ` 는 **2304 이상**(기본 2560). 내리면 정답이 잘려 `train.py` 가 행을 버린다.
OOM 이면 `LORA_R` 16→8. **`COMPUTE_DTYPE` 은 fp16 고정** — T4·P100 은 bf16 이 없어 bf16 으로 두면 시작 즉시 중단된다.

## 실행 순서

1. [setup_kaggle.md](setup_kaggle.md) 대로 계정·Secrets(`HF_TOKEN`)·데이터 업로드
2. `config.env` 의 `HF_PUSH_REPO` `HF_DATA_REPO` 를 본인 HF id 로
3. `kaggle/univus_f9_train.ipynb` 를 노트북으로 올려 실행 (또는 `kaggle kernels push -p kaggle/`)
4. 끝나면 HF repo 에 `adapter/*` `run.json` `eval_*.md` `compare.md` 가 있어야 한다 — **없으면 결과가 없는 것이다**

## 세션 제약 대응 (개발계획서 6.2)

| 제약 | 대응 |
| --- | --- |
| 12시간 | 1500건 × 3 epoch 는 T4 에서 1~2시간. 데이터가 커지면 `EPOCHS` 를 1 로 나눠 돌리고 매번 push |
| 디스크 초기화 | 노트북 마지막 셀이 어댑터·평가 결과를 HF 에 올린다. 중간에 끊기면 체크포인트도 사라진다 |
| 주 30시간 | 한 번 학습 설계를 할당 안에. 무제한 반복 불가 |
