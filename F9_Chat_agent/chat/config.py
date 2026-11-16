"""F9 자연어 대화 — 설정 (경로 · LLM 슬롯 · 시간 · 문구).

요구사항: 요구사항정의서.md F9 절 / 화면: Frontend-Route.md 14절.

설치할 것이 없다
  - 전부 **표준 라이브러리**다(라우터만 fastapi). LLM 은 OpenAI 호환 chat/completions 를 urllib 로 부른다.
    이름은 F9_LLM_BASE_URL · F9_LLM_API_KEY · F9_LLM_MODEL 이 먼저이고, 없으면 공용 슬롯 LLM_MAIN_* 를 받는다.
    모델은 AI팀 학습 모델(pod)이다 — 코드에 모델 이름을 쓰지 않는다.

LLM 설정(.env) 찾는 순서 — 앞의 파일이 정한 키가 이긴다. 환경변수가 있으면 그것이 먼저다.
  1. DATA_DIR/.env          설치본: %LOCALAPPDATA%\\kr.univus.desktop\\data\\F9_Chat_agent\\data\\.env (번들에는 .env 가 빠진다)
  2. DATA_DIR.parent/.env
  3. ROOT/.env              저장소에서 돌릴 때 (F9_Chat_agent/.env — gitignore)
  .env 값은 **이 모듈 안에만** 둔다(os.environ 에 쓰지 않는다) — 같은 프로세스의 다른 기능·자식 수집기가 F9 모델을 쓰지 않게.
  F9_LLM=off 면 설정이 있어도 끈다. F9_LLM_TIMEOUT(초, 기본 60).

저장하는 것
  - data/chat.db 하나 — 대화 메시지 · 대기 중인 쓰기(PendingAction). 이 PC 밖으로 나가지 않는다(F9-R03·R42).
  - 다른 기능의 데이터는 쓰지 않는다. 확인된 쓰기만 백엔드가 넘겨준 C1 store 로 한다.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent             # F9_Chat_agent/
PROJECT_ROOT = ROOT.parent                               # 26 WE-Meet Project/


def _read_env(path: Path) -> dict[str, str]:
    """.env 의 KEY=VALUE 를 **이 모듈 안의 dict 로만** 읽는다 — os.environ 에는 쓰지 않는다.
    (환경변수에 넣으면 같은 프로세스의 F11·F12 config 와 그 자식 수집기가 F9 pod 모델로 LLM 요청을 보내게 된다.)"""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return out


DATA_DIR = Path(os.environ.get("F9_DATA_DIR") or ROOT / "data")
DB_PATH = DATA_DIR / "chat.db"

ENV_FILES = (DATA_DIR / ".env", DATA_DIR.parent / ".env", ROOT / ".env")
_FILE: dict[str, str] = {}
for _p in ENV_FILES:                                     # 앞의 파일이 정한 키가 이긴다
    for _k, _v in _read_env(_p).items():
        _FILE.setdefault(_k, _v)


def setting(*names: str, default: str = "") -> str:
    """이름 순서대로 — 각 이름은 환경변수(있으면, 빈 값 포함) → .env 값. 처음 나온 비어 있지 않은 값."""
    for n in names:
        v = os.environ[n] if n in os.environ else _FILE.get(n, "")
        if v:
            return v
    return default


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── LLM (OpenAI 호환 chat/completions — pod 의 학습 모델) ──
#   F9_LLM_* 가 먼저, 없으면 공용 슬롯 LLM_MAIN_* (F1·F11·F12·C4 와 같은 이름)
LLM_BASE_URL = setting("F9_LLM_BASE_URL", "LLM_MAIN_BASE_URL").rstrip("/")
LLM_API_KEY = setting("F9_LLM_API_KEY", "LLM_MAIN_API_KEY")
LLM_MODEL = setting("F9_LLM_MODEL", "LLM_MAIN_MODEL")
try:
    LLM_TIMEOUT = float(setting("F9_LLM_TIMEOUT") or 60)   # 도구 호출 1턴 3~8초, 텍스트 5~10초
except ValueError:
    LLM_TIMEOUT = 60.0
TEMPERATURE = 0
MAX_TOKENS = 256
MAX_LLM_CALLS = 3                      # 한 요청당 LLM 호출 최대 횟수 (2-step 포함)


def llm_configured() -> bool:
    if setting("F9_LLM").lower() in ("off", "0", "false", "no"):
        return False
    return bool(LLM_BASE_URL and LLM_API_KEY and LLM_MODEL)


# ── system 문구 — 학습 데이터(Univ-Us_Train src/prepare_f9.py system_message)와 **한 글자도 다르지 않게** ──
WEEKDAY_KO = "월화수목금토일"
SYSTEM_RULES = ("당신은 유니버스(Univ-Us) 앱의 일정 도우미입니다. "
                "일정 쓰기(등록·수정·삭제)는 미리보기만 만들고 실행은 사용자 확인 뒤 서버가 합니다. "
                "숫자는 직접 계산하지 않고 도구 결과에 있는 값만 말합니다. 모르면 모른다고 답합니다. "
                "e클래스 마감·학사일정·수업은 바꿀 수 없습니다.")

# ── 대화 ──
MAX_TEXT = 2000                        # 사용자 문장 최대 길이
PENDING_MINUTES = 10                   # 미리보기 유효 시간 (F9 6절 · Q3)
LIST_MAX = 10                          # 목록 카드 최대 줄 수
HISTORY_LIMIT = 100                    # GET messages 기본 개수
HISTORY_MAX = 500

# 쓰기 의도 표지 — 읽기 도구 결과를 돌려주고 한 번 더 부른다(2-step: list_events → update/delete)
WRITE_INTENT = ("바꿔", "변경", "수정", "옮겨", "미뤄", "당겨", "지워", "삭제", "취소", "빼 줘", "빼줘", "없애")

CATEGORIES = {"personal": "개인", "study": "학업", "team": "팀플", "etc": "기타"}   # C1 config.CATEGORIES 와 같은 키

KIND_LABEL = {"deadline": "마감", "academic": "학사일정", "class": "수업", "exam": "시험", "study": "학습", "user": "내 일정"}

# 내 일정이 아닌 id 접두 → 거절 문구 (F9-R15 · S08)
REFUSE_PREFIX = {
    "dl:": "e클래스에서 가져온 일정은 바꿀 수 없습니다",
    "ac:": "학사일정·수업은 바꿀 수 없습니다",
    "cl:": "학사일정·수업은 바꿀 수 없습니다",
    "ex:": "시험 일정은 바꿀 수 없습니다",
    "pb:": "학습 블록은 공강 배치 화면에서 바꿔 주세요",
}

# 이동할 수 있는 화면 (route → 이름). 이름은 "{이름} 화면으로 이동할게요" · "{이름} 열기" 에 쓴다.
ROUTES = {
    "/": "대시보드",
    "/chat": "대화",
    "/assignments": "과제 목록",
    "/academic": "학사일정",
    "/graduation": "졸업요건",
    "/attendance": "출결",
    "/courses": "강의자료",
    "/exams": "시험",
    "/scholarships": "장학",
    "/opportunities": "대외활동",
    "/briefing": "브리핑",
    "/eclass/posts": "e클래스 게시글",
    "/settings/profile": "프로필 설정",
    "/settings/sources": "수집 설정",
    "/settings/requirements": "졸업요건 설정",
    "/settings/availability": "공강 설정",
    "/settings/notifications": "알림 설정",
}

# list_assignments group 인자 → F7 group
ASSIGNMENT_GROUPS = {"now": "now", "this_week": "week", "week": "week", "later": "later"}
GROUP_LABEL = {"overdue": "놓친 마감", "now": "지금 해야 함", "week": "이번 주", "later": "나중에", "nodue": "마감 없음"}

# ── 문구 ──
MSG = {
    "again": "다시 말씀해 주세요",
    "bad_date": "다시 말씀해 주세요 — 날짜를 알아듣지 못했습니다",
    "no_target": "어느 일정인지 찾지 못했습니다. 날짜와 제목을 함께 말씀해 주세요",
    "past": "지난 날짜입니다",
    "repeat": "반복은 아직 지원하지 않아 1회로 넣습니다",
    "no_calendar": "캘린더를 쓸 수 없습니다",
    "not_configured": "분석 서비스가 설정되지 않았습니다",
    "unreachable": "분석 서비스에 연결할 수 없습니다 — 터널(8600)과 모델 서버를 확인하세요",
    "auth": "분석 서비스 인증에 실패했습니다 — 키를 확인하세요",
    "bad_response": "분석 서비스의 응답을 읽지 못했습니다",
    "expired": "만료되었습니다 — 다시 말씀해 주세요",
    "done_already": "이미 처리된 미리보기입니다",
    "gone": "대상 일정이 사라졌습니다",
    "no_action": "미리보기를 찾을 수 없습니다",
    "empty": "문장을 입력하세요",
    "too_long": f"문장이 너무 깁니다 ({MAX_TEXT}자까지)",
    "preview_reply": "확인을 누르면 실행합니다",
}
