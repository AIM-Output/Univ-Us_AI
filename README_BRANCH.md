# gpu/qwen3-4b — Qwen3-4B · 학교 GPU 서버 · QLoRA

| 항목 | 값 |
| --- | --- |
| 모델 | `Qwen/Qwen3-4B` (tool calling 템플릿 내장 → `render_chat` 네이티브 모드) |
| 환경 | 학교 서버 A100 MIG `1g.10gb` — **가용 9.5GiB**, bf16 지원 |
| 방식 | 4-bit(nf4) QLoRA · LoRA r=16 · bf16 compute · MAX_SEQ 2560 · batch 1 × accum 8 · completion-only 손실 |
| 산출물 | `outputs/qwen3-4b-f9/adapter/` (→ `$PERSIST_DIR/outputs/…`, pod 재시작 후에도 남음) |

## 메모리 예산 (GPU-환경과-모델-선정.md 1장)

| 항목 | 크기 |
| --- | --- |
| 4B 가중치 4-bit | ~2.5GB |
| 활성화(2560 토큰, gradient checkpointing 켬) + 런타임 오버헤드 | 3~5GB |
| LoRA 파라미터·옵티마이저(8-bit paged) | <1GB |
| **합계** | **6~8GB / 9.5GiB** ✅ |

**토큰 예산:** system + 도구 11개 정의만 **약 1,950 토큰** → 한 행 2,000~2,200 토큰. `MAX_SEQ` 는 **2304 이상**(기본 2560). 더 줄이면 정답이 잘려 `train.py` 가 그 행을 버린다.
OOM 이 나면 `config.env` 에서 **이 순서로**: `LORA_R` 16→8 → (`BATCH` 는 이미 1) → 그래도면 `tools/f9_tools.json` 의 description 을 줄여 토큰을 아낀다(단, 평가 프롬프트도 같이 바뀌므로 이전 결과와 비교 불가).
학습 중에는 추론 서버를 내려야 한다 — 같은 9.5GiB 를 나눠 쓸 수 없다.

## 실행 순서

```bash
bash setup.sh     # 처음 1회 + pod 재시작 후 (venv·가중치 재설치가 정상 동작)
bash train.sh     # 데이터 없으면 합성 생성 → 학습 → outputs/qwen3-4b-f9/adapter
bash eval.sh      # 학습 전·후 비교표 → outputs/qwen3-4b-f9/compare.md
```

긴 작업은 세션과 분리: `setsid nohup bash train.sh > $HOME/univus-train-out/logs/train.log 2>&1 < /dev/null &`

## 영속성 — 홈 10GB 한도

| 경로 | 남나 | 내용 |
| --- | :---: | --- |
| `$PERSIST_DIR` = `~/univus-train-out` | ✅ | `data/` `outputs/` `logs/` — `train.sh` 가 저장소의 `data` `outputs` 를 여기로 심볼릭 링크 |
| `$SCRATCH_DIR` = `~/.venv/univus-train` | ❌ | `.venv/`(~8GB) · `hf_cache/`(가중치 ~8GB) — 재시작 후 `setup.sh` |

홈이 10GB 뿐이라 **체크포인트가 쌓이면 꽉 찬다** → `train.py` 가 `save_total_limit=2` 로 최근 2개만 남긴다. 끝난 run 의 `checkpoint-*` 는 지워도 된다(어댑터는 `adapter/` 에 따로 있다).

## 3자 비교의 Base 행

`BASE_MODEL_ID=Qwen/Qwen3-4B-Base bash eval.sh` 로 학습 전 Base 모델 행까지 한 표에 넣는다(개발계획서 7.2).
