"""LLM 호출 — OpenAI 호환 POST {LLM_MAIN_BASE_URL}/chat/completions (tools 포함, 스트리밍 없음).

표준 라이브러리(urllib)만 쓴다. 응답은 OpenAI 형식:
    choices[0].message.{content?, tool_calls?[].function.{name, arguments(JSON 문자열)}} · finish_reason tool_calls|stop
실패는 도메인 예외로 바꾼다 — api.py 가 HTTP 상태로 옮긴다.
    NotConfigured → 503 · Unreachable(연결·타임아웃·5xx) → 502 · AuthFailed(401·403) → 502 · BadResponse → 502
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from . import config as C


class LLMError(Exception):
    """LLM 쪽 실패 — str(e) 가 사용자에게 보일 한글 문장이다."""


class NotConfigured(LLMError):
    pass


class Unreachable(LLMError):
    pass


class AuthFailed(LLMError):
    pass


class BadResponse(LLMError):
    pass


def complete(messages: list[dict], tools: list[dict]) -> dict:
    """한 번 부르고 choices[0].message 를 돌려준다 ({content?, tool_calls?})."""
    if not C.llm_configured():
        raise NotConfigured(C.MSG["not_configured"])
    body: dict[str, Any] = {"model": C.LLM_MODEL, "messages": messages, "temperature": C.TEMPERATURE,
                            "max_tokens": C.MAX_TOKENS, "stream": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{C.LLM_BASE_URL}/chat/completions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json",
                                                         "Authorization": f"Bearer {C.LLM_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=C.LLM_TIMEOUT) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise AuthFailed(C.MSG["auth"]) from e
        if e.code >= 500:
            raise Unreachable(C.MSG["unreachable"]) from e
        raise BadResponse(f"{C.MSG['bad_response']} (HTTP {e.code})") from e
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as e:
        raise Unreachable(C.MSG["unreachable"]) from e
    try:
        data = json.loads(raw)
        msg = data["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise BadResponse(C.MSG["bad_response"]) from e
    if not isinstance(msg, dict):
        raise BadResponse(C.MSG["bad_response"])
    return msg
