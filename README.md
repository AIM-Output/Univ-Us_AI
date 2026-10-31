# Univ-Us_Train — F9 자연어 대화(Function Calling) 모델 자체 학습

유니버스(WE-Meet) 데스크톱 앱의 **F9 자연어 대화** 기능이 쓸 LLM 을 **외부 API 대신 직접 파인튜닝**해서
학교 GPU 서버에 OpenAI 호환 서버로 띄우기 위한 저장소입니다.

WE-Meet 쪽 코드는 손대지 않습니다. 앱은 `LLM_MAIN_BASE_URL · LLM_MAIN_API_KEY · LLM_MAIN_MODEL` 세 값만 읽으므로
학습한 모델을 OpenAI 호환 서버로 띄우고 `.env` 의 `LLM_MAIN_BASE_URL` 을 그 서버로 바꾸면 그대로 붙습니다.

> **이 저장소는 로컬 전용(원격 없음)입니다.** 데이터·학습 결과는 git 에 올라가지 않습니다(`.gitignore`).
> 어댑터와 데이터는 HF Hub(private) 로 주고받습니다 → [DATA_FLOW.md](DATA_FLOW.md)

---

## 1. 왜 F9 인가

| 근거 | 내용 |
| --- | --- |
| 지금 학습 가능 | AI 에이전트 개발계획서 3.1 — F9 는 **공개 한국어 function calling 데이터가 있어 지금 학습 가능한 유일한 기능**. 강의계획서 추출(P1)은 데이터가 없어 막혔다 |
| 범위가 작다 | 도구가 **11개로 고정**돼 있다 (WE-Meet 요구사항정의서 F9 §5). 모델이 할 일은 도구 고르기 + 인자 채우기뿐 |
| 틀려도 막힌다 | 날짜 문자열은 **백엔드가 다시 검증**하고, 쓰기 도구는 **미리보기 → 사용자 확인** 뒤에만 실행된다(2단계 프로토콜). 모델 오류가 바로 캘린더에 들어가지 않는다 |
| 포트폴리오 | 학습 전 Base / 학습 없는 Instruct / 학습 후 모델을 **같은 평가셋**으로 재서 표로 낸다(개발계획서 7.2). 이 표가 산출물이다 |

F9 가 **하지 않는 것**(요구사항정의서 F9-D4·R23): 숫자 계산. 졸업학점·우선순위·배치는 규칙 코드가 이미 한다. 모델은 도구가 준 값만 말한다.

---

## 2. 학습 환경 두 곳

| | 학교 GPU 서버 | 캐글 |
| --- | --- | --- |
| GPU | A100 80GB 의 **MIG `1g.10gb` 슬라이스 1개 — 실제 9.5GiB** | T4 16GB × 2 (또는 P100 16GB) |
| 정밀도 | **bf16 지원** (Ampere) | **bf16 없음 → fp16** (Turing). `config.env` 는 fp16 으로 두고, 혹시 bf16 이 남아 있어도 `train.py` 가 감지해 fp16 으로 자동 전환한다 |
| 세션 | 상시. 단, pod 재시작 시 overlay(venv·HF 캐시) 소멸 | **12시간** 제한, **주 30시간** GPU 할당 |
| 디스크 | 홈 `/home/jovyan` **10GB** 만 영속. venv(~8GB)·모델 가중치는 홈에 못 들어간다 → overlay 에 두고 재설치 전제 | 세션 종료 시 **전부 초기화** → 어댑터는 반드시 HF Hub 에 push |
| 학습 가능 크기 | QLoRA 4-bit 기준 **4B 급까지**. 8B 는 컨텍스트 짧게 해도 OOM 위험 | QLoRA 4-bit 로 **8B 가능** |
| 서빙 | **여기서 한다** (학습과 동시에는 불가 — VRAM 경합) | 불가 |

> 두 환경은 직접 통신하지 않습니다. **HF Hub 가 유일한 통로**입니다. 올리지 않은 것은 없는 것과 같습니다(개발계획서 6.3).

---

## 3. 브랜치 지도

| 브랜치 | 모델 | 환경 | 4-bit | compute dtype | MAX_SEQ | 메모 |
| --- | --- | --- | :---: | :---: | ---: | --- |
| `main` | — | — | — | — | — | **문서만** (이 README · DATA_FLOW) |
| `common` | — | — | — | — | — | **공통 코드** — `src/` `tools/` `train.sh` `eval.sh` `smoke_test.*`. 모델 브랜치는 전부 여기서 갈라진다 |
| `gpu/qwen3-4b` | `Qwen/Qwen3-4B` | 학교 서버 | ✅ | bf16 | 2560 | 기본 후보. tool calling 템플릿 내장 |
| `gpu/hyperclovax-1.5b` | `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | 학교 서버 | ❌ (bf16 LoRA 로 충분) | bf16 | 2560 (batch 1) | 한국어 특화·초경량. **HF gated** — 약관 동의 + `hf auth login`(토큰은 홈 `~/.cache/huggingface/token` 에 남는다) 또는 `config.local.env` 의 `HF_TOKEN=` |
| `kaggle/qwen3-8b` | `Qwen/Qwen3-8B` | 캐글 T4×2 | ✅ | **fp16** | 2560 | 정확도 상한. 세션 제한 때문에 `SAVE_STEPS` 짧고 HF push 필수 |

> **MAX_SEQ 는 2304 아래로 내리지 않습니다.** system 메시지 + 도구 11개 정의만 **약 1,950 토큰**이라(실측, Qwen 토크나이저) 한 행이 2,000~2,200 토큰입니다.
> 더 짧게 잡으면 정답(assistant 턴)이 잘리고, `train.py` 는 그런 행을 학습에서 **버립니다**. OOM 은 `BATCH`·`LORA_R` 로 잡습니다.

> 수집 브랜치의 `training/README.md` 가 후보로 적었던 `HyperCLOVAX-SEED-Text-Instruct-8B` 는 **HF 에 존재하지 않습니다**(2026-10-26 확인 — 404).
> 네이버가 공개한 SEED Text 계열은 0.5B·1.5B 이고, 8B 급은 Omni(멀티모달)뿐입니다. 그래서 8B 는 Qwen3 로만 둡니다.

### 브랜치 쓰는 법

```bash
git switch gpu/qwen3-4b          # 모델 브랜치로 이동 → 그 브랜치의 config.env · setup.sh 사용
bash setup.sh                    # 환경 준비 (서버) — 처음 1회 + pod 재시작 후
bash train.sh                    # 데이터 생성(없으면) → 학습 → outputs/<RUN_NAME>/adapter
bash eval.sh                     # 학습 전·후 같은 평가셋으로 채점 → outputs/<RUN_NAME>/compare.md
```

- **공통 코드(`src/` 등)를 고칠 때는 `common` 에서 고치고** 각 모델 브랜치에서 `git merge common` 합니다. 모델 브랜치에서 직접 고치면 다른 브랜치가 못 받습니다.
- 모델 브랜치에서 바꾸는 것은 **`config.env` · `setup.sh` · `README_BRANCH.md`** 뿐입니다.
- 모델을 바꿔 보려면 `config.env` 의 `MODEL_ID` 한 줄만 바꿉니다. `src/` 안에는 모델 이름이 없습니다.
- 내 PC 에서만 쓰는 값(경로·토큰·실험용 모델)은 `config.local.env` 에 씁니다(gitignore). `RUN_NAME=run2 bash train.sh` 처럼 환경변수로 줘도 됩니다(파일보다 우선).

### 서버로 코드 보내기 — 이 저장소에는 원격이 없다

`git pull` 할 곳이 없으므로 **브랜치를 체크아웃한 폴더를 그대로 복사**합니다. `.git` `data` `outputs` `.venv` 는 빼고 보냅니다.

```bash
# 내 PC (Git Bash) — 서버 접속 설정은 docs/SSH-서버-접속-가이드.md 의 host 별칭을 그대로 쓴다
git switch gpu/qwen3-4b
rsync -av --exclude .git --exclude data --exclude outputs --exclude .venv --exclude __pycache__ ./ jnu-gpu:~/Univ-Us_Train/
#   rsync 가 없으면:  scp -r config.env setup.sh train.sh eval.sh requirements.txt src tools jnu-gpu:~/Univ-Us_Train/
```

나중에 GitHub 원격을 붙이면 서버에서 `git clone -b gpu/qwen3-4b <주소>` 로 바꿉니다. 그때까지는 **복사한 시점의 코드**가 서버에서 도는 것이므로, `common` 을 고쳤으면 다시 보내야 합니다.

### 운영 구조 — Flask 서버 2개 + SSH 터널 (2026-10-30 결정)

pod 는 홈이 **3.1GB** 남았고 바깥으로 열린 포트가 SSH 뿐이라, **pod 에는 코드와 병합한 4-bit 가중치(Qwen3-4B ≈2.8GB)만** 두고 데이터·결과는 내 PC 가 듭니다.

| 어디 | 무엇 | 파일 |
| --- | --- | --- |
| pod | `gpu_server` (Flask :8600) — 학습 지휘 · 어댑터 병합(nf4) · OpenAI 호환 추론. 상태 `idle/training/merging/serving` 중 하나 | [`gpu_server/`](gpu_server/README.md) |
| 내 PC | `local_server` (Flask :8601) — `data/f9` 제공 · `results/<run>` 수신 | [`local_server/`](local_server/README.md) |
| 내 PC | SSH 터널 `-L 8600` `-R 8601` · 코드 배포(git archive → scp) | `scripts/tunnel.ps1` · `scripts/deploy_to_server.ps1` |

WE-Meet 는 `.env` 의 `LLM_MAIN_BASE_URL=http://127.0.0.1:8600/v1` 만 바꾸면 됩니다. 절차 전체는 [gpu_server/README.md](gpu_server/README.md).

### 로컬(Windows, RTX 4060 8GB)에서 돌려 보기

```powershell
git switch common
.\setup_local.ps1                # .venv 생성 + torch(cu129) + requirements
.\smoke_test.ps1                 # 60건 합성 → 0.5B 급 모델 8 step 학습 → 평가 2회 (파이프라인 검증용)
```

---

## 4. 평가 원칙 (개발계획서 5.4 · 7.2 · 7.3)

| 원칙 | 이 저장소에서 |
| --- | --- |
| 평가셋은 학습에 절대 사용 금지 | `prepare_f9.py` 가 생성 시점에 `train.jsonl` / `eval.jsonl` 을 **템플릿 패밀리 단위**로 분리한다. 외부 데이터(`--extra`)는 train 에만 들어간다 |
| 평가셋 고정 | 같은 `--seed` 면 같은 파일. 바꾸면 모델 간 비교가 깨진다 |
| 3자 비교 | `eval.py` 를 **어댑터 없이 / 어댑터 있게** 두 번 돌려 같은 표에 넣는다(`eval.sh`). Base 모델 행은 `BASE_MODEL_ID=Qwen/Qwen3-4B-Base bash eval.sh` 처럼 한 번에 |
| 동일 프롬프트 | 학습·평가가 같은 `render_chat()` 을 쓴다 — 템플릿이 다르면 모델 비교가 아니다 |
| 모든 턴 채점 | 행의 **assistant 턴 전부**를 채점한다(teacher forcing). 두 단계 행은 도구 턴 + 답변 턴, update/delete 행은 list_events 턴 + update/delete 턴 |
| 실패 사례 보존 | `eval.md` 에 실패 10건을 남긴다. 성공률만 남기면 다음에 뭘 고칠지 모른다 |

지표(턴 단위): **도구 선택 정확도 · 인자 정확도(필드별) · 거절 정확도(도구를 부르면 안 되는 질문) · 텍스트 턴 정확도(거절 + 도구 결과 뒤 답변) · 턴 정확도 · 응답 지연**.

---

## 5. 관련 문서

| 문서 | 위치 | 왜 보나 |
| --- | --- | --- |
| 요구사항정의서 F9 §5 인터페이스 계약 | `../26-WE-Meet-Project/요구사항정의서.md` | 도구 11개·2단계 프로토콜·시간 해석 규칙 — `tools/f9_tools.json` 의 원본 |
| AI 에이전트 개발계획서 | `../Univ_Us_collect_ai-agent_logan0741/AI_에이전트_개발계획서.md` | 3.1 F9 우선 이유 · 5.4 평가셋 규칙 · 6.2 세션 제약 · 7.2 비교 실험 |
| GPU 환경과 모델 선정 | `../Univ_Us_collect_ai-agent_logan0741/docs/GPU-환경과-모델-선정.md` | 9.5GiB 예산표 · 후보 모델 |
| 서버 작업환경 설정 | `../Univ_Us_collect_ai-agent_logan0741/docs/서버-작업환경-설정.md` | pod 재시작 시 무엇이 사라지나 · venv 재설치 |
| 데이터 흐름 | [DATA_FLOW.md](DATA_FLOW.md) | 어디서 무엇이 만들어져 어디로 가나 |
