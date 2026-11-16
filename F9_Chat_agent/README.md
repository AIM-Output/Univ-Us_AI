# F9_Chat_agent — 자연어 대화 (Function Calling, 유니버스 F9)

`다음 주 목요일 발표 리허설 저녁 8시` · `이번 주 마감 뭐 남았어?` · `e클래스 동기화해줘` 를 한 줄로 처리한다.
요구사항은 [요구사항정의서.md](../요구사항정의서.md) F9 절, 화면은 [Frontend-Route.md](../Frontend-Route.md) 14절.

- **LLM 은 도구를 고르기만 한다.** 모델은 AI팀이 pod 에 올린 학습 모델(OpenAI 호환 `chat/completions`, 도구 11개 — `chat/tools.json`).
- **읽기**(일정·졸업요건·출결·과제·시험)는 각 기능이 계산한 값을 그대로 옮겨 서버가 문장·목록 카드·근거 링크를 만든다 — 숫자를 다시 계산하지 않는다.
  `get_exams` 는 F5 `overview()` 를 부르므로 `/exams` 화면을 열 때와 같이 F5 DB 를 갱신한다(공지 동기화·기본 시험 채우기·지난 시험 닫기).
- **쓰기**(일정 등록·수정·삭제)는 미리보기 카드만 만든다(10분). 사용자가 [확인]을 누르면 서버가 C1 캘린더에 쓴다.
  e클래스 마감·학사일정·수업·시험은 바꾸지 않고 거절한다.
- **설치할 것이 없다** — 표준 라이브러리 + fastapi(라우터). 대화 기록은 `data/chat.db`(gitignore)에만 남는다.

```
F9_Chat_agent/
├── chat/
│   ├── config.py     경로 · .env 읽기 · LLM 슬롯(LLM_MAIN_*) · system 문구(학습 데이터와 같게) · 문구
│   ├── tools.json    도구 11개 (Univ-Us_Train tools/f9_tools.json 그대로 — x-kind 는 로더가 뗀다)
│   ├── tools.py      도구 목록 읽기 · 읽기 도구 실행 · Deps(백엔드가 넘겨주는 함수)
│   ├── llm.py        urllib 로 chat/completions — 실패를 503/502 용 예외로
│   ├── service.py    1턴 처리(2-step 포함) · 시간 검증 · 겹침 · 미리보기 · 확인/취소 · 템플릿 답변
│   ├── store.py      SQLite — messages · actions(대기 중인 쓰기)
│   ├── api.py        FastAPI 라우터 /api/chat* (univ_us_local/backend/app/chat.py 가 붙인다)
│   └── __main__.py   명령줄 (연결 확인용)
├── tests/            가짜 LLM 으로 루프 · 미리보기 · 확인 · 만료 · 거절
└── data/             (gitignore) chat.db · (설치본이면) .env
```

## .env

`.env.example` 을 `.env` 로 복사해 채운다. **키는 커밋하지 않는다**(`.env` 는 gitignore, 사이드카 번들에도 들어가지 않는다).

```
F9_LLM_BASE_URL=http://127.0.0.1:8600/v1        # 내 PC 터널 → pod
F9_LLM_API_KEY=<UNIVUS_API_KEY 와 같은 값>
F9_LLM_MODEL=univus-f9
F9_LLM_TIMEOUT=60                                # 초
# F9_LLM=off                                     # 끄기
```

`F9_LLM_*` 가 없으면 공용 슬롯 `LLM_MAIN_BASE_URL` · `LLM_MAIN_API_KEY` · `LLM_MAIN_MODEL` 을 대신 받는다.
찾는 순서: `DATA_DIR/.env` → `DATA_DIR` 의 부모 `/.env` → `F9_Chat_agent/.env` (앞의 파일이 이긴다). 환경변수가 있으면 그것이 먼저다.
`.env` 값은 F9 안에서만 쓰고 **환경변수로 내보내지 않는다** — 같은 프로세스의 F11·F12 와 그 수집기가 F9 모델로 요청하지 않게.
설치본은 `%LOCALAPPDATA%\kr.univus.desktop\data\F9_Chat_agent\data\.env` 에 둔다.
설정이 없으면 `/api/chat` 이 503 `분석 서비스가 설정되지 않았습니다`, 터널·서버가 죽어 있으면 502 를 준다.

## 명령

```
cd F9_Chat_agent
python -m pytest tests -q                 # 테스트 (fastapi · httpx · pytest)
python -m chat status                     # LLM 설정 확인
python -m chat "내일 일정 뭐 있어?"         # 한 턴 (백엔드 없이 — 읽기 값은 비어 있다)
```
