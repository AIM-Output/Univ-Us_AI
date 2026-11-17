# wemeet/f9-chat-integration — WE-Meet 백엔드에 F9 자연어 대화(Function Calling) 붙이기

WE-Meet 제품 저장소(`AIM-Output/26-WE-Meet-Project`)에 **F9 자연어 대화**를 끼워 넣은 코드 그대로입니다.
내 PC 의 설치본에서 2026-11-02 에 실제로 돌려 본 상태이며(서빙 테스트 13문장 → [docs/results/f9-qwen3-4b/serving_test.md](../../blob/main/docs/results/f9-qwen3-4b/serving_test.md)),
WE-Meet 저장소에는 커밋되지 않았기 때문에 **적용용 패치**로 여기에 둡니다.

| 기준 커밋 | `8243747` — `버전 0.2.0 — C4 브리핑 · F11 장학 · F12 대외활동 · 새 로고` ([patches/BASE_COMMIT.txt](patches/BASE_COMMIT.txt)) |
| --- | --- |
| 패치 | [patches/wemeet-f9-chat-integration.patch](patches/wemeet-f9-chat-integration.patch) — 기존 파일 12개 수정 (`git apply --check` 통과 확인) |
| 새 파일 | `F9_Chat_agent/`(기능 폴더 전체) · `univ_us_local/backend/app/chat.py`(라우터 연결) · `univ_us_local/frontend/src/lib/chat.ts` |
| 모델 서버 | 이 저장소 `gpu/qwen3-4b` 브랜치의 `gpu_server/`(OpenAI 호환 `/v1/chat/completions`) + Release `weights-2026-11-13` 의 가중치 — 절차는 [main README](../../blob/main/README.md) "백엔드에 적용하기" |

## 적용 순서 (WE-Meet 저장소 루트에서)

```bash
# 0) 이 브랜치를 옆에 받는다
git clone -b wemeet/f9-chat-integration --single-branch https://github.com/AIM-Output/Univ-Us_AI.git ../univus-f9

# 1) 새 폴더·파일 복사
cp -r ../univus-f9/F9_Chat_agent ./F9_Chat_agent
cp ../univus-f9/univ_us_local/backend/app/chat.py   univ_us_local/backend/app/chat.py
cp ../univus-f9/univ_us_local/frontend/src/lib/chat.ts univ_us_local/frontend/src/lib/chat.ts

# 2) 기존 파일 12개 수정 — 기준 커밋(8243747)이면 그대로 들어간다. main 이 더 앞서 있으면 --3way 로 충돌 부분만 손본다
git apply --check ../univus-f9/patches/wemeet-f9-chat-integration.patch && git apply ../univus-f9/patches/wemeet-f9-chat-integration.patch
#   충돌 시: git apply --3way ../univus-f9/patches/wemeet-f9-chat-integration.patch

# 3) LLM 설정 — F9_Chat_agent/.env (예시 파일을 복사해 값만 채운다. .env 는 커밋 금지)
cp F9_Chat_agent/.env.example F9_Chat_agent/.env
#   F9_LLM_BASE_URL=http://127.0.0.1:8600/v1     ← gpu_server 주소
#   F9_LLM_API_KEY=<gpu_server 의 UNIVUS_API_KEY 와 같은 값>
#   F9_LLM_MODEL=univus-f9
#   (F9_LLM_* 가 없으면 공용 LLM_MAIN_BASE_URL · LLM_MAIN_API_KEY · LLM_MAIN_MODEL 을 대신 읽는다)

# 4) 확인
python -m pytest F9_Chat_agent/tests -q          # 가짜 LLM 으로 루프·미리보기·확인·거절 테스트 (fastapi · httpx · pytest 필요)
cd F9_Chat_agent && python -m chat status        # LLM 설정이 읽히는지
```

백엔드(`univ_us_local/backend`)를 띄우면 `/api/chat*` 가 열리고 `/api/status` 에 `chat` 칸(`available` · `llmConfigured` · `model`)이 생깁니다.
F9 폴더가 없거나 LLM 설정이 비어 있어도 나머지 화면은 그대로 동작합니다(`available=false`, `/api/chat` 은 503).

## 무엇이 바뀌나 (패치 12개 파일)

| 파일 | 내용 |
| --- | --- |
| `univ_us_local/backend/app/main.py` | `chat` 모듈 import · 라우터 include · `/api/status` 에 chat 칸 |
| `univ_us_local/backend/app/config.py` | `F9_AGENT_DIR`(기본 `PROJECT_ROOT/F9_Chat_agent`) |
| `univ_us_local/frontend/src/components/pages/ChatPage.tsx` | 대화 화면 — 말풍선 · 도구 결과 카드 · 쓰기 미리보기 [확인]/[취소] |
| `univ_us_local/frontend/src/lib/api.ts` · `types.ts` · `components/dashboard/FeatureTiles.tsx` | `/api/chat` 호출 · 타입 · 타일 활성화 |
| `C0_Platform_agent/osenv/appdata.py` · `desktop/cli.py` · `desktop/sidecar/univus-backend.spec` | 설치본 데이터 폴더(`F9_Chat_agent/data`) · 사이드카 번들에 F9 포함 |
| `.github/workflows/macos.yml` · `DEVELOPMENT.md` · `univ_us_local/README.md` | 빌드에 F9 폴더 포함 · 문서 |

## 동작 원리 (요약 — 자세한 건 [F9_Chat_agent/README.md](F9_Chat_agent/README.md))

- LLM 은 **도구 11개 중 하나를 고르고 인자를 채우는 일만** 합니다(`chat/tools.json`, 학습 데이터의 `tools/f9_tools.json` 과 같은 정의).
- 읽기 도구(일정·졸업요건·출결·과제·시험)는 각 기능이 계산한 값을 그대로 전달하고 서버가 문장·카드를 만듭니다 — 숫자를 다시 계산하지 않습니다.
- 쓰기(일정 등록·수정·삭제)는 **미리보기 카드**만 만들고(10분 유효), 사용자가 [확인]을 누르면 C1 캘린더에 씁니다. 학사일정·수업·시험·e클래스 마감은 바꾸지 않고 거절합니다.
- 표준 라이브러리 + fastapi 만 쓰므로 백엔드 `.venv` 에 추가 설치가 없습니다. 대화 기록은 `F9_Chat_agent/data/chat.db`(gitignore).
