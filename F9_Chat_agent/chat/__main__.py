"""명령줄 — LLM 연결 확인용 (F9_Chat_agent 에서 실행).

    python -m chat status              LLM 설정 · 모델 이름
    python -m chat "내일 일정 뭐 있어?"   한 턴 돌려 보기 (data/chat.db 에 남는다)

명령줄은 백엔드 없이 돈다 — 캘린더·다른 기능 값이 없으므로 읽기 도구는 '없음'으로, 쓰기는 미리보기까지만 나온다.
앱과 같은 값으로 보려면 앱의 /chat 화면을 쓴다.
"""
from __future__ import annotations

import json
import sys

from . import api, llm, service, store


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")         # Windows 콘솔(cp949)에 없는 글자(—)로 죽지 않게
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "status":
        print(json.dumps(api.status_summary(), ensure_ascii=False, indent=2))
        return 0
    try:
        with store.connect() as con:
            out = service.turn(con, api.configure(None), " ".join(argv))
    except (llm.LLMError, service.Invalid) as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
