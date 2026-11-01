# Univ-Us_Train 인수인계 (HANDOFF)

> **이 문서 하나로 로컬 PC 에서든 학교 GPU 서버에서든 Claude Code 에게 작업을 이어서 시킬 수 있게 쓴 문서입니다.**
> 마지막 갱신: 2026-11-02 · 작성: logan0741 + Claude Code
> 상세 설계는 [README.md](README.md) · [DATA_FLOW.md](DATA_FLOW.md) · [gpu_server/README.md](gpu_server/README.md) 에 있고, 이 문서는 **맥락 · 현재 상태 · 다음 할 일**을 모읍니다.

---

## 0. 이어서 시키는 법

| 어디서 | 어떻게 |
| --- | --- |
| **로컬 PC** | `C:\Project\Univ-Us_AI\Univ-Us_Train` 에서 Claude Code 를 열고 → `HANDOFF.md 읽고 6장 현재 상태 확인한 뒤 8장 N번부터 이어가줘` |
| **GPU 서버** | `ssh -p 30562 -i private.pem jovyan@168.131.30.102` → `cd ~/univus-train` → Claude Code 실행 → 같은 요청 |
| **GPU 서버 작업 기록** | pod 의 Claude 가 한 일은 `~/univus-train-out/POD_SESSION_LOG_<날짜>.md` 에 남긴다(배포에 덮어써지지 않는 영속 폴더). 로컬에서 이어갈 때 이 파일도 읽는다 — 2026-11-01 기록은 enum 근거·8 step 스모크·발견한 불일치를 담고 있고, 로컬이 2026-11-02 에 전부 반영했다 |

서버의 `~/univus-train` 은 **로컬 레포의 `gpu/qwen3-4b` 브랜치를 `git archive` 로 보낸 사본**이라 `.git` 이 없습니다.
서버에서 코드를 고쳤으면 **로컬 레포에 같은 수정을 반영하고 다시 배포**해야 남습니다(서버 쪽 수정은 다음 배포 때 덮어써짐).

작업 원칙 (사용자 지시):
- **시킨 것만 한다.** 제안·다음 단계 덧붙이지 않는다.
- 비밀값(`config.local.env` · `scripts/server.env` · `private.pem` · HF 토큰)은 **절대 커밋하지 않는다.**
- 모델 이름은 `config.env` 의 `MODEL_ID` 한 줄에만. `src/` 안에는 모델 이름이 없다.
- 공통 코드는 `common` 브랜치에서 고치고 모델 브랜치에 `git merge common`.
- 서버에서 지우는 작업은 **목록을 보여주고 확인받은 뒤**에만.

---

## 1. 프로젝트 맥락 (왜 이걸 하나)

| 저장소 | 주인 | 내용 | 우리와의 관계 |
| --- | --- | --- | --- |
| `AIM-Output/26-WE-Meet-Project` | 김하람(팀장) | **실제 제품.** Tauri 데스크톱 앱 + FastAPI + Next.js. C0~C4 · F1~F8 · F11~F13 이 **규칙 기반**으로 구현 완료(v0.2.0) | AI 가 들어갈 자리를 비워 둠. LLM 은 `LLM_MAIN_BASE_URL / API_KEY / MODEL` 세 값만 읽음 |
| `logan0741/Univ-Us_AI` | 우리(AI팀) | 수집 크롤러(collect 브랜치) · Donut OCR 골격(pic 브랜치) · 문서 | core/doc/chat 에이전트 코드는 **없음**. 강의계획서는 2026-09-13 제품에서 **삭제**돼 doc/pic 방향은 폐기 |
| `Univ-Us_Train` (이 레포, 로컬 전용) | 우리 | **F9 자연어 대화(Function Calling) 모델 자체 학습 + 서빙** | WE-Meet 요구사항정의서 F9 §5 계약(도구 11개 · 2단계 미리보기)을 그대로 따름 |

**WE-Meet 가 AI팀 몫으로 남긴 것**: ① F9 Function Calling (화면만 있고 백엔드 없음) ② F4 강의자료 RAG(요약·문제·질문) ③ `LLM_MAIN` 모델 선정.
이 레포는 ①과 ③을 다룹니다. ②는 아직 시작 안 함.

**모델 결정 (2026-10-30)**: **Qwen/Qwen3-4B**. 이유 — 서버 9.5GiB 에서 학습(QLoRA)과 서빙을 둘 다 할 수 있는 유일한 크기, tool calling 템플릿 내장, 병합 후 4-bit 약 2.8GB. 비교용으로 Qwen3-8B(캐글)·HyperCLOVAX-SEED 1.5B 브랜치가 있음.
**학습이 필요한 이유**: API 대신 자체 모델을 쓰기로 했고, 소형 모델은 한국어 도구 선택·인자 채우기 정확도가 낮아 LoRA 로 올린다. 포트폴리오 산출물은 "학습 전 Base / 학습 없는 Instruct / 학습 후" 3자 비교표.

---

## 2. 저장소 구조와 브랜치

| 브랜치 | 내용 | 모델 | 환경 |
| --- | --- | --- | --- |
| `main` | 문서만 (README · DATA_FLOW · HANDOFF) | — | — |
| `common` | 공통 코드: `src/` `tools/` `train.sh` `eval.sh` `smoke_test.*` `gpu_server/` `local_server/` `scripts/` | — | — |
| `gpu/qwen3-4b` | **주 브랜치.** `config.env` `setup.sh` `README_BRANCH.md` | Qwen/Qwen3-4B · 4-bit · bf16 · MAX_SEQ 2560 | 학교 서버 |
| `gpu/hyperclovax-1.5b` | 〃 | HyperCLOVAX-SEED-Text-Instruct-1.5B · bf16 LoRA · HF gated | 학교 서버 |
| `kaggle/qwen3-8b` | `config.env` `setup_kaggle.md` `kaggle/*.ipynb` | Qwen/Qwen3-8B · 4-bit · **fp16**(T4) · EPOCHS 1 | 캐글 |

```
Univ-Us_Train/
├── HANDOFF.md  CLAUDE.md  README.md  DATA_FLOW.md
├── config.env            ← 브랜치별 (모델·메모리·경로). 비밀값은 config.local.env (gitignore)
├── tools/f9_tools.json   ← WE-Meet F9 도구 11개 스키마 (category·kinds enum 은 2026-11-02 에 C1 실제 값과 동기화)
├── src/common.py         ← 렌더(chat template + tools) · <tool_call> 파서 · 공용 유틸
├── src/prepare_f9.py     ← 한국어 합성 데이터 생성 → data/f9/{train,eval}.jsonl (평가셋 템플릿 패밀리 단위 분리)
├── src/train.py          ← QLoRA/LoRA, completion-only 손실, max_seq 초과 행 제거, HF push
├── src/eval.py           ← 모든 assistant 턴 채점 → 비교표 (도구 선택·인자·거절·텍스트 턴·지연)
├── train.sh  eval.sh     ← config.env → config.local.env → 셸 env 순으로 읽고 실행
├── gpu_server/app.py     ← Flask: 학습 오케스트레이션 + 병합 + OpenAI 호환 /v1/chat/completions
│   setup_server.sh  run_server.sh  README.md
├── local_server/app.py   ← Flask(내 PC): data/f9 제공 + 학습 결과 수신(results/)
└── scripts/deploy_to_server.ps1  tunnel.ps1  server.env(.example)
```

---

## 3. 환경 사실 (2026-10-31 실측)

### 로컬 PC (Windows 11)
- 레포 `C:\Project\Univ-Us_AI\Univ-Us_Train`, venv `.venv` = Python 3.10 · torch 2.9.1+cu128 · transformers 5.20 · trl 1.14 · peft 0.21 · bitsandbytes 0.50 · flask
- GPU RTX 4060 Laptop 8GB (스모크 테스트·서버 코드 로컬 테스트용)
- SSH 키 `C:\Project\Univ-Us_AI\private.pem` · 접속값 `scripts\server.env` (gitignore, API 키 포함)
- **주의**: Bash 도구에서 한글 파일명이 깨질 수 있음 → Read 도구 사용. 인터넷이 가끔 끊김 → HF 는 `HF_HUB_OFFLINE=1` 로 캐시 사용

### 학교 GPU 서버 (Kubernetes pod)
| 항목 | 값 |
| --- | --- |
| 접속 | `ssh -p 30562 -i private.pem jovyan@168.131.30.102` (외부에서 열린 포트는 **SSH 뿐**이라고 가정) |
| GPU | A100 80GB 의 MIG `1g.10gb` 슬라이스 1개 = **9.5GiB**, bf16 지원 |
| 홈 `/home/jovyan` | 10GB **영속**. **2026-10-31 정리 후 9.7GB 여유** (사용 384MB = `.vscode-server/data` 설정 281MB · `.codex` 90MB · 캐시 14MB). 정리 전엔 `.vscode-server` 가 8.8GB 였음 — 서버 빌드·확장·캐시 8.6GB 를 overlay `/home/jovyan_venv/.venv/_trash_20261031/` 로 옮김(pod 재시작 전까지 복구 가능, 재시작 시 자동 소멸). **VS Code Remote-SSH 로 다시 접속하면 서버는 자동 재설치되고 확장(Python·Claude Code 등)은 원격에 다시 설치해야 함** |
| overlay (`/`, `~/.venv/*`) | 824GB 여유, **pod 재시작 시 소멸** |
| 코드 | `~/univus-train` (git archive 사본, 브랜치 `gpu/qwen3-4b`, `.deployed_branch` 에 기록) |
| venv | `~/.venv/univus-train/.venv` — Python 3.11 · torch 2.13+cu129 · transformers 5.22 · trl 1.17 · peft 0.21 · bnb 0.50 · flask (overlay) |
| 가중치 캐시 | `~/.venv/univus-train/hf_cache` — Qwen3-4B 7.6GB 선다운로드 완료 (overlay) |
| 영속 산출물 | `~/univus-train-out/{data,outputs,logs,models}` — 레포의 `data` `outputs` 가 여기로 심볼릭 링크 |
| API 키 | `~/univus-train/config.local.env` 의 `UNIVUS_API_KEY` = 로컬 `scripts/server.env` 와 같은 값 |
| 로그 | 설치 `~/univus-train-out/logs/setup.log` · 서버 `server.log` · 학습 `<run>.log` |
| 팀 venv | `~/.venv/univus-ai` (7.9GB, 팀 공용, 건드리지 않음) |

**pod 가 재시작되면**: venv·가중치·overlay 의 병합본이 사라짐 → `cd ~/univus-train && bash gpu_server/setup_server.sh` 재실행(약 7분) → 서버 재기동 → 병합본은 어댑터에서 다시 병합. 홈의 어댑터·데이터·로그는 남음.

---

## 4. 데이터 흐름

### 학습
```
[내 PC]                                        [GPU 서버 pod]
tools/f9_tools.json ─ prepare_f9.py ─▶ data/f9/train.jsonl · eval.jsonl
local_server (Flask :8601)  ◀── ssh -R 8601 ──  gpu_server /train/start
   GET /data/manifest · /data/*.jsonl  ─────────▶  sha256 검증 → ~/univus-train-out/data/f9/
                                                   bash train.sh  (QLoRA, config.env)
                                                   bash eval.sh   (학습 전·후 비교표)
   POST /results/<run>/…  ◀────────────────────── adapter/ · run.json · eval_*.md · compare.md 업로드
   results/<run>/ 에 사본 저장                      업로드 성공 후 checkpoint-* 삭제 (홈 예산)
```
### 병합 · 서빙
```
gpu_server /model/merge {run_name}  →  어댑터 + 베이스 → nf4 병합본
     저장 위치: 홈 ~/univus-train-out/models/<run>-merged-nf4 (≈2.8GB, 영속)  — 홈이 부족하면 overlay 로 대체(비영속, 재시작 후 재병합)
gpu_server /model/load {source: merged|adapter}  →  serving 상태
[내 PC] WE-Meet 앱 ── http://127.0.0.1:8600/v1/chat/completions ── ssh -L 8600 ──▶ gpu_server
        .env:  LLM_MAIN_BASE_URL=http://127.0.0.1:8600/v1  LLM_MAIN_API_KEY=<UNIVUS_API_KEY>  LLM_MAIN_MODEL=anything
```
- 기계 사이를 오가는 것은 **JSONL 텍스트와 어댑터(≈132MB)** 뿐. 사용자 개인 데이터는 이 레포에 들어오지 않는다.
- **학습과 서빙은 같은 9.5GiB 를 못 나눠 쓴다** → gpu_server 가 상태기계(`idle|training|merging|loading|serving`)로 막는다.
- 요청마다 system + 도구 11개 ≈ **1,950 토큰**이 붙는다 → MAX_SEQ 2560, 프롬프트 8192 토큰 초과는 400.

---

## 5. 운영 절차 (명령 모음)

```powershell
# ── 내 PC ──
git switch gpu/qwen3-4b ; git merge common          # 공통 코드 바뀌었으면
.\scripts\deploy_to_server.ps1                      # 브랜치 추적 파일 → pod ~/univus-train
.\scripts\tunnel.ps1                                # -L 8600 / -R 8601 (켜 둔 채로)
.\local_server\run_local.ps1                        # 데이터 서버 :8601 (켜 둔 채로)
python src\prepare_f9.py --n 1500 --seed 42         # data/f9 생성 (평가셋 고정: seed 바꾸지 말 것)
```
```bash
# ── pod ──
cd ~/univus-train
bash gpu_server/setup_server.sh                     # 처음 1회 + pod 재시작 후
bash gpu_server/run_server.sh start|status|stop     # Flask :8600 (127.0.0.1)
```
```bash
# ── 내 PC 에서 터널로 호출 (KEY = server.env 의 UNIVUS_API_KEY) ──
curl -H "Authorization: Bearer $KEY" http://127.0.0.1:8600/health
curl -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"run_name":"qwen3-4b-f9"}' http://127.0.0.1:8600/train/start
curl -H "Authorization: Bearer $KEY" http://127.0.0.1:8600/train/status
curl -X POST ... -d '{"run_name":"qwen3-4b-f9"}'                 http://127.0.0.1:8600/model/merge
curl -X POST ... -d '{"source":"merged"}'                        http://127.0.0.1:8600/model/load
curl -X POST ... -d '{"model":"x","messages":[{"role":"user","content":"이번 주 마감 뭐 남았어?"}],"tools":[...]}' http://127.0.0.1:8600/v1/chat/completions
```
전체 설명과 응답 예시: [gpu_server/README.md](gpu_server/README.md).

---

## 6. 현재 상태 (2026-11-02 기준 체크리스트)

| # | 항목 | 상태 | 비고 |
| :---: | --- | :---: | --- |
| 1 | 레포·브랜치 5개 구성, 스모크 테스트(로컬 4060) | ✅ | 2026-10-26 |
| 2 | 학습·평가 코드 아키텍트 검증 | ✅ APPROVED | 2026-10-26 |
| 3 | Flask 서버 2개 + 배포·터널 스크립트 작성, 로컬 테스트 58항목 | ✅ | 2026-10-31 |
| 4 | 서버 코드 아키텍트 검증 | ✅ APPROVED (3차) | 1차 REJECTED(7건) → 2차 REJECTED(블로커 2건) → 3차 APPROVED 2026-10-31. 비차단 후속 7건은 7장 끝에 기록 |
| 5 | pod 의 옛 `~/Univ-Us_AI` 클론 삭제 | ✅ | 커밋 안 된 작업은 로컬 scratchpad `server_backup_20261030` 에 백업 |
| 6 | pod 에 코드 배포 (`~/univus-train`, gpu/qwen3-4b) | ✅ | 2026-10-31 2차 수정본 재배포 완료 (`.deployed_branch` = gpu/qwen3-4b) |
| 7 | pod 환경 설치 (`setup_server.sh`) | ✅ STATUS=DONE | venv 7.6GB · Qwen3-4B 캐시 7.6GB (overlay). pip 캐시가 홈을 채워 삭제함 → 스크립트가 pip·xet 캐시를 overlay 로 보내도록 수정됨 |
| 8 | pod 에 `config.local.env` (API 키) | ✅ | |
| 9 | gpu_server 기동 + 터널 `/health` | ✅ | 2026-10-31 06:36 `run_server.sh start` → 내 PC 에서 `http://127.0.0.1:8600/health` 200 (state idle, VRAM 9.39GiB 여유) · pod 에서 역터널로 `http://127.0.0.1:8601/data/manifest` 200 (sha256 포함, 무인증 401). **터널·로컬 서버는 내 PC 에서 켜 둔 동안만 유효** — 다시 켤 때 `scripts\tunnel.ps1` + `local_server\run_local.ps1` |
| 10 | 첫 학습 run (`qwen3-4b-f9`) | 🔄 | 학습 쌍 1,568 × 3 epoch = 588 step. pod 스모크 실측 약 34 s/step → 학습 약 5.5h + 평가(315건 × 2회) 약 1h. **2026-11-02 02:33(내 PC 시각) 시작** — 데이터 받기 OK, 예상 완료 09:00~10:00. 확인: `GET /train/status`. 시작 명령: 내 PC 에서 `POST /train/start {"run_name":"qwen3-4b-f9"}` |
| 11 | 병합 → 적재 → WE-Meet `.env` 연결 | ⬜ | 홈 1.3GB 라 병합본은 overlay 대체 예상 |
| 12 | 3자 비교표(Base / Instruct / 학습 후) | ⬜ | `BASE_MODEL_ID=Qwen/Qwen3-4B-Base bash eval.sh` |
| 13 | `tools/f9_tools.json` 의 `category`·`kinds` enum 을 WE-Meet C1 실제 값과 맞추기 | ✅ | 2026-11-02. category = C1 `config.py` CATEGORIES(personal·study·team·etc), kinds = 백엔드 `main.py` CALENDAR_SOURCES 의 kind(deadline·academic·class·exam·user·study, todo 제거). `data/f9` 같은 seed 로 재생성 → pod 재배포. 같은 날 2차: 종일 일정 `end` 를 exclusive(다음 날, C1-R14)로 고쳐 다시 재생성(train 1,568쌍 / eval 358쌍, 종일 57건 모두 end=start+1) |
| 14 | F4 RAG (parse/index/summarize/generate/ask) | ⬜ | 아직 시작 안 함 |
| 15 | pod 홈 정리 ("처음 구축" 기준) | ✅ | 2026-10-31 홈 1.3GB → **9.7GB 여유**. `.vscode-server` 빌드·확장·캐시 8.6GB 를 overlay 임시 폴더로 이동(설정 `data/User`·`Machine` 유지), pip·xet 캐시 제거. (11-01 VS Code Remote 재접속으로 서버·확장 2GB 재설치 → 여유 8GB) |
| 16 | pod 실기기 스모크 — 새 데이터 받기 → `train.sh --max_steps 4 --max_train_rows 16` → `eval.sh --limit 12` | ✅ | 2026-11-02. Qwen3-4B nf4 가 MIG 9.5GiB 에서 적재·학습·평가 끝까지 OK. 16행 = 20쌍, 토큰 p50 2032 / max 2274 (<2560, 버림 0), 약 34 s/step, loss 1.83→0.55. 데이터 양방향 전송(내려받기 sha256 일치 · `/results` 업로드) 확인. 산출물 `~/univus-train-out/outputs/qwen3-4b-smoke/`, 로그 `~/univus-train-out/logs/smoke.log`, 스크립트 `~/univus-train-out/smoke_run.sh` |
| 17 | 평가 거절 채점 보정 (`src/eval.py`) | ✅ | 2026-11-02. 거절 턴은 도구 미호출 + 거절 표현(`common.REFUSAL_MARKERS`)이 있어야 맞음. 이전엔 텍스트만 있으면 맞음이라 학습 전 모델의 '도와드리겠다' 도 100% 로 잡혔다(pod 로그 §5 지적). 학습 데이터의 거절 문장 176건 전부 마커 포함 확인 |

---

## 7. 알려진 제약 · 주의

- **학습 중 pod 에서 GPU 작업 금지** — 같은 MIG 9.5GiB 를 나누면 OOM. `train.py`·`eval.py`·스모크 모두 안 됨. 진행은 내 PC 에서 `GET /train/status` 로만 본다. 학습 중 터널·로컬 서버가 잠깐 끊겨도 학습은 계속되지만, **끝날 때 결과 업로드**가 있으니 그때는 켜져 있어야 한다(꺼져 있으면 `ok: partial`, 어댑터는 pod `outputs/` 에 남음).
- 로컬 PC 에서 만든 `data/f9/*.jsonl` 은 LF 고정(`common.write_jsonl`) — pod 에서 재생성한 것과 sha256 이 같다.
- **홈 10GB**: 2026-10-31 정리로 9.7GB 여유 → 병합본(≈2.8GB)은 홈 `~/univus-train-out/models/` 에 영구 보관 가능(기본 `MODELS_DIR`). 홈을 다시 채우는 범인은 pip 캐시·hf-xet 캐시·`.vscode-server` 이므로, 설치 스크립트는 캐시를 overlay 로 보내고, VS Code Remote 는 꼭 필요한 확장만 설치할 것. overlay 의 옛 팀 venv `~/.venv/univus-ai`(7.9GB)는 홈과 무관해 그대로 둠(재시작 시 소멸).
- 학습 후 홈에 남는 것: 어댑터 ≈132MB · run.json · 평가 md. `checkpoint-*` 는 업로드 성공 후 자동 삭제.
- `/train/stop` 은 프로세스 그룹 전체를 종료해야 함(수정 중). 서버 종료 전 학습 중이면 먼저 `/train/stop`.
- WE-Meet C4 브리핑의 `LLM_TIMEOUT=3`초는 nf4 4B 가 못 맞출 수 있음 → 측정 후 C4 쪽 조정 필요.
- `huggingface-cli` 는 없음, CLI 는 `hf` (`hf auth login`, `hf repos create`, `hf upload`, `hf download`).
- `.gitignore` 는 줄 끝 주석을 지원하지 않음(한 번 당한 버그).
- 캐글 브랜치: T4 는 bf16 없음(fp16), 세션 12h, 디스크 초기화 → 어댑터 HF push 필수, EPOCHS=1 부터.
- 평가셋은 `--seed 42` 로 고정. 바꾸면 이전 실험과 비교 불가.

**아키텍트 3차 검증에서 남긴 비차단 후속 7건** (`gpu_server/app.py` · `run_server.sh`, 시키면 처리):
1. `app.py` 병합 추정 함수 docstring 에 모델 이름 언급 → 문구만 수정
2. TERM 이 train.sh 와 eval.sh 사이(또는 다운로드 중)에 오면 `proc=None` 이라 `stopped` 가 안 찍혀 eval.sh 가 고아로 시작될 수 있음 → 핸들러에서 `TRAIN["stopped"]=True` 를 무조건 설정하고 `_run_logged` 가 시작 전에 확인
3. `ensure_group_gone` 이 예외(PermissionError)를 내면 `TRAIN["proc"]` 이 남음 → try/finally 로 None 처리
4. `run_server.sh stop` 의 KILL 대기 30초가 핸들러 최악 30초와 겹침 → 45초로
5. `run_server.sh` 의 `curl -m 25` 가 `/train/stop` 최악 시간보다 짧음 → `-m 50`
6. `/model/merge` 응답의 `out_dir` 이 overlay 로 떨어진 경우에도 홈 경로를 보여줌 → 실제 위치는 `result.location`
7. `/health` 의 model 블록에 `persistent`·`path` 없음 → 추가

---

## 8. 다음 할 일 (순서)

1. ~~서버 코드 2차 검증 지적 반영 → 커밋 → 병합 → 재배포~~ ✅ (2026-10-31)
2. ~~서버 기동 → 터널 → `/health` 200~~ ✅ (2026-10-31) — 다시 켤 때: pod `run_server.sh start`, 내 PC `tunnel.ps1` + `run_local.ps1`
3. 🔄 **2026-11-02 02:33 시작** (`run_name` qwen3-4b-f9). 데이터는 `data/f9`(seed 42, train 1,185 행 = 학습 쌍 1,568 / eval 315 행 = 358 쌍 — enum 동기화 + 종일 end exclusive 반영). 내 PC 에서 `/train/start {"run_name":"qwen3-4b-f9"}` → `/train/status` 로 loss 확인 (수 시간). **학습 중에는 터널과 로컬 서버를 끄지 말 것** (데이터 다운로드는 시작 직후 한 번, 결과 업로드는 끝날 때)
4. `/model/merge` → `/model/load {"source":"merged"}` → `/v1/chat/completions` 도구 호출 확인 → WE-Meet `.env` 연결
5. `results/qwen3-4b-f9/compare.md` 로 3자 비교표 정리 (Base 행 추가)
6. 그다음 F4 RAG 착수 여부 결정

---

## 9. 기록 (결정과 그 이유)

| 날짜 | 결정 | 이유 |
| --- | --- | --- |
| 2026-10-26 | 학습 대상 = F9 Function Calling (OCR 제외) | 도구 11개 고정, 공개 데이터 가능, 날짜는 백엔드가 재검증 → 소형 모델로 닿는 과제 |
| 2026-10-26 | 환경별 브랜치 분리 (서버 bf16 / 캐글 fp16) | T4 는 bf16 미지원, 디스크 초기화 등 제약이 다름 |
| 2026-10-30 | 모델 = Qwen3-4B | 9.5GiB 에서 학습·서빙 동시 가능한 유일 크기, tool calling 내장 |
| 2026-10-30 | 서버는 학습·추론, 데이터는 로컬 Flask, 서버 간 통신 | 서버 홈 용량이 작음 |
| 2026-10-30 | 병합본은 nf4 단일 체크포인트 | LoRA 어댑터만으로는 추론 불가, 베이스 8GB 는 홈에 못 둠 |
| 2026-10-31 | pod 의 옛 클론만 삭제, 도구 캐시는 유지 | 사용자 선택 |
| 2026-11-02 | 도구 enum 을 WE-Meet 코드 실제 값으로 (category `etc`, kinds 에서 `todo` 제거) | 임시값으로 학습하면 모델이 앱에 없는 `appointment`·`other` 를 내보낸다. 할 일은 C1 에서 kind=user + isTodo 라 별도 kind 가 없다 |
| 2026-11-02 | 종일 일정 `end` = 다음 날(exclusive) 로 고친 뒤 학습 시작 (pod 로그 §7-1 의 두 갈래 중 '고친 뒤') | WE-Meet C1-R14 규칙. end=start 로 학습하면 모델이 앱 규칙과 다른 값을 내보낸다. 고치는 비용은 재생성 수 초 + 재배포 1분 |
| 2026-11-02 | 거절 턴 채점에 거절 표현 조건 추가 | 학습 전 모델이 거절하지 않았는데 100% 로 잡히면 Base/Instruct 행이 부풀려져 비교표 신뢰가 떨어진다(논문 약점) |
