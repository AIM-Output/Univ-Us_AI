#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""앱(WE-Meet F9_Chat_agent)이 모델에 보내는 것이 학습 데이터와 같은지 본다 — 가이드 10장 6번 "꼭 확인".

  python scripts/check_app_prompt.py <F9_Chat_agent/chat 폴더>
  # 예) 패치 브랜치에서 꺼내 보기:
  #   git show origin/wemeet/f9-chat-integration:F9_Chat_agent/chat/config.py  > /tmp/chat/config.py
  #   git show origin/wemeet/f9-chat-integration:F9_Chat_agent/chat/tools.json > /tmp/chat/tools.json
  #   python scripts/check_app_prompt.py /tmp/chat

한 글자라도 다르면 정확도가 떨어진다. 비교하는 것:
  1. system 문구 — 앱 `config.py` 의 WEEKDAY_KO · SYSTEM_RULES 로 만든 글 ↔ 생성기 `prepare_f9.system_message(now)`
     (요일 7개 · 자정 직후 · 한 자리 월·일을 포함한 여러 시각으로 비교)
  2. 도구 정의 — 앱 `tools.json` ↔ `tools/f9_tools.json`
앱의 system 글을 만드는 식(`service.system_prompt`)은 "오늘은 {날짜}({요일}) {시:분} 입니다. 주 시작은 월요일입니다.\\n{SYSTEM_RULES}" 로
가정한다 — 앱 코드가 바뀌면 이 파일의 APP_FORMAT 도 같이 고친다. 앱 코드를 실행하지는 않는다(config.py 의 상수만 읽는다).
다르면 종료 코드 1. 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare_f9  # noqa: E402

APP_FORMAT = "오늘은 {now:%Y-%m-%d}({wd}) {now:%H:%M} 입니다. 주 시작은 월요일입니다.\n{rules}"
NOWS = [dt.datetime(2026, 11, 23 + i, 9 + i, 5 * i) for i in range(7)] + [dt.datetime(2027, 1, 1, 0, 0), dt.datetime(2027, 3, 9, 23, 55)]


def constants(path: Path) -> dict:
    """config.py 를 실행하지 않고 맨 위의 상수 대입만 읽는다."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return out


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    chat = Path(sys.argv[1])
    bad = 0
    c = constants(chat / "config.py")
    if "SYSTEM_RULES" not in c or "WEEKDAY_KO" not in c:
        print("config.py 에서 SYSTEM_RULES · WEEKDAY_KO 를 찾지 못했습니다 — 앱 코드가 바뀌었는지 확인하세요")
        return 1
    diff = 0
    for now in NOWS:
        app = APP_FORMAT.format(now=now, wd=c["WEEKDAY_KO"][now.weekday()], rules=c["SYSTEM_RULES"])
        gen = prepare_f9.system_message(now)
        gen = gen["content"] if isinstance(gen, dict) else gen
        if app != gen:
            diff += 1
            if diff == 1:
                i = next((k for k, (a, b) in enumerate(zip(app, gen)) if a != b), min(len(app), len(gen)))
                print(f"system 문구가 다릅니다 (기준시각 {now:%Y-%m-%d %H:%M}) — {i}번째 글자부터")
                print("   앱    :", repr(app[max(0, i - 30):i + 40]))
                print("   생성기:", repr(gen[max(0, i - 30):i + 40]))
    print(f"1. system 문구: {'같음' if not diff else '다름'} (기준시각 {len(NOWS)}개 중 {len(NOWS) - diff}개 일치)")
    bad += bool(diff)

    a = json.loads((chat / "tools.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "tools" / "f9_tools.json").read_text(encoding="utf-8"))
    print(f"2. 도구 정의: {'같음' if a == b else '다름'}")
    if a != b:
        bad += 1
        la = a.get("tools", a) if isinstance(a, dict) else a
        lb = b.get("tools", b) if isinstance(b, dict) else b
        if isinstance(la, list) and isinstance(lb, list):
            na = {t["function"]["name"]: t for t in la if isinstance(t, dict) and "function" in t}
            nb = {t["function"]["name"]: t for t in lb if isinstance(t, dict) and "function" in t}
            for k in sorted(set(na) | set(nb)):
                if na.get(k) != nb.get(k):
                    print("   다른 도구:", k, "(앱에만 있음)" if k not in nb else "(저장소에만 있음)" if k not in na else "")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
