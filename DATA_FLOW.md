# 데이터 흐름

> **기계 사이를 오가는 것은 JSONL 텍스트와 LoRA 어댑터(수십 MB)뿐입니다.**
> 사용자 개인 데이터(일정·성적·로그인)는 이 저장소에 들어오지 않습니다. 학습 데이터는 합성 + 공개 데이터입니다.

```
[내 PC / common]                    [HF Hub (private)]              [학습 환경]
tools/f9_tools.json ─┐
                     ├─ prepare_f9.py ─→ data/f9/train.jsonl ──┐
(선택) 공개 한국어    │                   data/f9/eval.jsonl    │
function-calling ────┘   --extra                                │
JSONL                                                           │        gpu/* 브랜치
                                                                ├──────────────────────▶ 학교 서버 (코드는 rsync/scp 로 복사 — 원격 없음,
                                                                │   data 는 PERSIST_DIR)  bash train.sh
                                                                │
                                                                │        kaggle/* 브랜치
                                              dataset repo ◀────┘   노트북이 snapshot_download
                                                   │                 → python src/train.py
                                                   ▼
                                                                    outputs/<run>/adapter  (LoRA)
                                              model repo  ◀──────── --push_to_hub (캐글은 필수)
                                                   │
                                                   ▼
                              [서빙 — 이 저장소 범위 밖]
                              학교 서버: 어댑터 merge → OpenAI 호환 서버
                                        → WE-Meet .env  LLM_MAIN_BASE_URL=http://<서버>:<포트>/v1
```

## 단계별

| # | 단계 | 입력 → 출력 | 어디서 | 명령 |
| :---: | --- | --- | --- | --- |
| 1 | **데이터 생성** | `tools/f9_tools.json` → `data/f9/train.jsonl` · `eval.jsonl` · `README.md` | 어디서든 (CPU) | `python src/prepare_f9.py --n 1500 --seed 42` |
| 2 | (선택) **외부 데이터 합치기** | 공개 한국어 function-calling JSONL → train 에만 추가 | 〃 | `python src/prepare_f9.py --extra 외부.jsonl` |
| 3 | **학습** | train.jsonl → `outputs/<RUN_NAME>/adapter/` + `run.json` | gpu: 서버(코드는 rsync/scp 로 복사 — README 3장) · kaggle: 노트북 | `bash train.sh` / 노트북 실행 |
| 4 | **어댑터 보관** | adapter/ → HF model repo (private). `HF_PUSH_REPO` 가 있으면 **학습 중 체크포인트도** `last-checkpoint/` 로 올라간다(hub_strategy=checkpoint) | 〃 | `HF_PUSH_REPO=<id>/<repo>` 또는 `--push_to_hub` |
| 5 | **평가** | eval.jsonl + (어댑터 유무) → `outputs/<RUN_NAME>/eval_base.md` · `eval_adapter.md` · `compare.md` — 행의 assistant 턴 전부를 채점 | 학습한 곳 (GPU) | `bash eval.sh` |
| 6 | **서빙** | adapter + base → merge → OpenAI 호환 서버 | 학교 서버 | 범위 밖 (WE-Meet 는 base_url 만 바꾼다) |

### 1단계 — 평가셋은 여기서 갈라진다

`prepare_f9.py` 는 질문을 **템플릿 패밀리**(같은 말투 묶음) 단위로 만들고, 패밀리 몇 개를 **통째로 eval 로** 보냅니다.
행을 무작위로 섞어 나누면 학습 때 본 말투가 평가에 그대로 나와 점수가 부풀기 때문입니다.
같은 `--seed` 면 같은 파일이 나옵니다. **평가셋을 바꾸면 이전 실험과 비교할 수 없습니다.**

`--extra` 로 넣은 외부 데이터는 **train 에만** 들어갑니다. 평가셋은 우리 도구 11개 기준으로 고정합니다.

### 3단계 — 학습 텍스트는 한 가지 모양

학습·평가가 같은 `render_chat()` 을 씁니다.

1. 토크나이저의 chat template 에 `tools=` 를 넘겨 본다. 렌더 결과에 첫 도구 이름이 들어 있으면 **네이티브 모드**.
2. 안 되면(템플릿이 `tools` 를 모르거나 에러) 도구 목록을 JSON 으로 system 메시지에 넣는 **폴백 모드**.
3. 어느 모드든 assistant 의 도구 호출은 **`<tool_call>{"name":…,"arguments":{…}}</tool_call>`** 문자열로 통일합니다. 모델이 달라도 정답 형식이 같아야 평가가 같은 파서로 됩니다.

손실은 기본 **completion-only** 입니다. assistant 턴마다 `(prompt, completion)` 쌍을 만들어 정답 토큰에만 손실을 두고, 두 단계 행(도구 호출 → 결과 → 답변)은 쌍 2개가 됩니다. `--loss full` 로 전체 텍스트 손실로 바꿀 수 있습니다.

**토큰 예산:** system 메시지 + 도구 11개 정의가 약 **1,950 토큰**이라 한 행이 2,000~2,200 토큰입니다. `MAX_SEQ` 는 **2304 이상**이어야 하고(기본 2560), 넘는 행은 `train.py` 가 잘라서 학습하지 않고 **버립니다**(버린 수를 로그와 `run.json` 에 남김).

### 서버의 영속 경로 — 무엇이 남고 무엇이 사라지나

학교 서버는 Kubernetes pod 입니다(서버-작업환경-설정.md 1장). 홈은 10GB 뿐이고 venv 는 overlay 라 재시작 때 사라집니다.

| 변수 (gpu/* `config.env`) | 기본값 | 영속 | 담는 것 |
| --- | --- | :---: | --- |
| `PERSIST_DIR` | `$HOME/univus-train-out` | ✅ | `data/`(JSONL) · `outputs/`(어댑터·run.json·eval.md) · `logs/` |
| `SCRATCH_DIR` | `$HOME/.venv/univus-train` | ❌ | `.venv/`(~8GB) · `hf_cache/`(모델 가중치, Qwen3-4B bf16 ≈ 8GB) — `HF_HUB_CACHE` 로 지정 |
| (HF 로그인 토큰) | `~/.cache/huggingface/token` | ✅ | `hf auth login` 이 쓰는 곳. **`HF_HOME` 은 바꾸지 않는다** — 바꾸면 토큰까지 overlay 로 가서 재시작 때 사라진다. 대안: `config.local.env` 에 `HF_TOKEN=` |

pod 가 재시작되면 `bash setup.sh` 를 다시 돌립니다. venv 재설치와 가중치 재다운로드가 **정상 동작**입니다. 어댑터는 `PERSIST_DIR` 에 있어 남습니다.
홈 10GB 한도 때문에 체크포인트는 `save_total_limit=2` 로 제한합니다.

### 캐글 — 세션이 끝나면 디스크가 비워진다

| 제약 | 대응 |
| --- | --- |
| 세션 12시간 | `EPOCHS=1`·`SAVE_STEPS=50`. `HF_PUSH_REPO` 가 있으면 학습 중 체크포인트가 Hub 의 `last-checkpoint/` 로 올라간다. **`--resume` 은 세션을 넘지 못한다** — 다음 세션에서 이어 가려면 먼저 `hf download <repo> --include "last-checkpoint/*" --local-dir outputs/<RUN_NAME>` 로 받아 `checkpoint-N` 이름으로 둔 뒤 `--resume` |
| 디스크 초기화 | 학습 끝에 **반드시** `--push_to_hub` 로 어댑터를 HF model repo 에 올린다. 노트북이 자동으로 한다 |
| 데이터 반입 | `data/f9/*.jsonl` 을 HF dataset repo(private) 에 올려 두고 노트북이 `snapshot_download` |
| bf16 없음 | `COMPUTE_DTYPE=fp16`. bf16 이 남아 있어도 `train.py` 가 `is_bf16_supported()` 로 감지해 fp16 으로 바꾸고 로그에 남긴다 |

### 6단계 이후 — WE-Meet 와 만나는 지점

WE-Meet 의 F1·F11·F12·C4 와 앞으로 만들 F9 백엔드는 전부 `LLM_MAIN_*` 슬롯만 읽습니다.
서빙 서버가 `/v1/chat/completions` 와 `tools` 파라미터를 받으면 앱 코드는 바뀌지 않습니다.
서빙과 학습은 같은 9.5GiB 를 나눠 쓸 수 없으므로 **학습 중에는 서빙을 내리고, 서빙 중에는 학습하지 않습니다.**
