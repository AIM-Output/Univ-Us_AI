#!/usr/bin/env python
"""F9 모델에 말 걸기 — 서빙 중인 gpu_server 의 /v1/chat/completions 에 tools/f9_tools.json 11개를 붙여 보낸다 (표준 라이브러리만).

  내 PC :  .venv\\Scripts\\python.exe scripts\\f9_chat.py "내일 오후 3시 스터디 2시간"      터널 8600 경유 · 키는 scripts\\server.env
           .venv\\Scripts\\python.exe scripts\\f9_chat.py                                  인자 없으면 대화 모드 (빈 줄 또는 Ctrl+C 로 끝)
  pod   :  ~/.venv/univus-train/.venv/bin/python scripts/f9_chat.py "..."                 키는 config.local.env
  옵션  :  --today "2026-11-02 15:10"  (기본 지금)   --no-tools   --raw(응답 JSON 그대로)   --max-tokens 256   --url   --key

system 문구는 학습 데이터와 똑같이 src/prepare_f9.py 의 system_message() 로 만든다 (오늘 날짜·요일·시각 + 규칙).
한 줄 = 요청 하나(단일 턴). 모델이 도구를 고르면 그 도구 호출(백엔드가 미리보기 카드로 보여 줄 것)을 그대로 찍는다 — 실행은 하지 않는다.

  ■ 내일 오후 3시 스터디 2시간
    → 도구 create_event  {"title": "스터디", "start": "2026-11-03T15:00", "end": "2026-11-03T17:00", "allDay": false, "category": "study"}
    finish=tool_calls · prompt 2020 tok · 생성 76 tok · 서버 4.55s · 왕복 4.59s
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402
from prepare_f9 import system_message  # noqa: E402


def read_env_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"^{key}=(.*)$", line.strip())
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return ""


def load_key() -> str:
    k = os.environ.get("UNIVUS_API_KEY", "").strip()
    if k:
        return k
    for f in (ROOT / "config.local.env", ROOT / "scripts" / "server.env"):
        k = read_env_value(f, "UNIVUS_API_KEY")
        if k:
            return k
    return ""


def parse_today(s: str | None) -> dt.datetime:
    if not s:
        return dt.datetime.now().replace(second=0, microsecond=0)
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise SystemExit(f"--today 형식: 'YYYY-MM-DD HH:MM' 또는 'YYYY-MM-DD' (받은 값: {s!r})")


def ask(url: str, key: str, messages: list[dict], tools: list[dict] | None, max_tokens: int) -> tuple[dict, float]:
    body: dict = {"model": "univus-f9", "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", method="POST",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8")), time.time() - t0


def show(d: dict, wall: float, raw: bool) -> None:
    if raw:
        print(json.dumps(d, ensure_ascii=False, indent=1))
        return
    ch = d["choices"][0]
    msg, fin, u = ch["message"], ch["finish_reason"], d.get("usage", {})
    for tc in msg.get("tool_calls") or []:
        f = tc["function"]
        print(f"  → 도구 {f['name']}  {f['arguments']}")
    if msg.get("content"):
        print(f"  → 본문: {msg['content']}")
    elif not msg.get("tool_calls"):
        print("  → 본문: (빈 응답)")
    lat = d.get("univus", {}).get("latency_s")
    print(f"  finish={fin} · prompt {u.get('prompt_tokens')} tok · 생성 {u.get('completion_tokens')} tok · 서버 {lat}s · 왕복 {wall:.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="사용자 문장. 비우면 대화 모드")
    ap.add_argument("--url", default=os.environ.get("GPU_SERVER_URL", "http://127.0.0.1:8600"))
    ap.add_argument("--key", default="", help="비우면 UNIVUS_API_KEY 환경변수 → config.local.env → scripts/server.env")
    ap.add_argument("--today", default=None, help="system 의 '오늘' — 'YYYY-MM-DD HH:MM'. 기본 지금")
    ap.add_argument("--no-tools", action="store_true", help="도구 없이 보냄 (C4·F1·F11·F12 가 쓰는 경로)")
    ap.add_argument("--raw", action="store_true", help="응답 JSON 그대로 출력")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    key = args.key or load_key()
    now = parse_today(args.today)
    tools, _ = common.load_tools(ROOT / "tools" / "f9_tools.json")
    if args.no_tools:
        tools = None
    system = system_message(now)

    def one(text: str) -> None:
        print(f"■ {text}")
        try:
            d, wall = ask(args.url, key, [{"role": "system", "content": system}, {"role": "user", "content": text}], tools, args.max_tokens)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            hint = " — 인증 실패: 키 확인" if e.code == 401 else (" — 모델이 serving 상태가 아님: POST /model/load" if e.code == 503 else "")
            print(f"  HTTP {e.code}{hint}\n  {body}")
            return
        except Exception as e:  # noqa: BLE001
            print(f"  연결 실패: {e}\n  터널(ssh -L 8600)과 pod gpu_server 가 켜져 있는지 확인")
            return
        show(d, wall, args.raw)

    print(f"서버 {args.url} · 도구 {len(tools) if tools else 0}개 · 오늘 {system.splitlines()[0]}")
    if args.text:
        one(" ".join(args.text))
        return
    print("문장을 입력하세요 (빈 줄 또는 Ctrl+C 로 끝)")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        one(text)


if __name__ == "__main__":
    main()
