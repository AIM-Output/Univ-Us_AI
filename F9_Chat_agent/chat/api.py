"""대시보드(univ_us_local)에 붙이는 F9 API — FastAPI APIRouter. (Frontend-Route 14-6)

    GET    /api/chat/status                       {available, llmConfigured, model?} (/api/status 의 chat 칸과 같다)
    GET    /api/chat/messages?before=&limit=100   {messages: ChatMsg[]} 오래된 → 최신 (만료된 미리보기는 state=expired)
    DELETE /api/chat/messages                     204 — 메시지 · 대기 동작 전부 삭제
    POST   /api/chat                              {text} → {reply, messages(이번 턴에 새로 생긴 것)}
    POST   /api/chat/actions/{id}/confirm         {title?, start?, end?, allDay?, category?} → {messages: [preview(confirmed), system]}
    POST   /api/chat/actions/{id}/cancel          → {messages: [preview(canceled)]}

**쓰기 도구는 /api/chat 에서 실행되지 않는다.** 미리보기만 만들고, 실제 변경은 confirm 에서만 일어난다(F9-R40).
핸들러는 동기 def 다 — LLM 호출(urllib)이 막히는 동안 이벤트 루프를 붙잡지 않게 스레드풀에서 돈다.

의존성: fastapi + 표준 라이브러리. 캘린더 · 다른 기능 값 · C1 쓰기는 백엔드가 Deps 로 넘겨준다.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Union

from fastapi import APIRouter, Body, HTTPException, Query, Response

from . import config as C
from . import llm, service, store
from .tools import Deps

__all__ = ["Deps", "build_router", "configure", "status_summary"]

_deps = Deps()


def configure(deps: Union[Deps, dict, None]) -> Deps:
    """백엔드가 넘겨준 의존을 기억한다 (dict 도 받는다 — Deps 의 칸 이름)."""
    global _deps
    if deps is None:
        _deps = Deps()
    elif isinstance(deps, Deps):
        _deps = deps
    else:
        _deps = Deps(**deps)
    return _deps


def status_summary() -> dict:
    """/api/status 의 chat 칸 — 화면이 입력창을 잠글지 정한다(F9-S09)."""
    out: dict[str, Any] = {"available": True, "llmConfigured": C.llm_configured()}
    if C.LLM_MODEL:
        out["model"] = C.LLM_MODEL
    return out


def _errors(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except service.Invalid as e:
        raise HTTPException(422, str(e))
    except service.NotFound as e:
        raise HTTPException(404, str(e))
    except service.Conflict as e:
        raise HTTPException(409, str(e))
    except service.Unavailable as e:
        raise HTTPException(503, str(e))
    except llm.NotConfigured as e:
        raise HTTPException(503, str(e))
    except llm.LLMError as e:
        raise HTTPException(502, str(e))


def build_router(deps: Union[Deps, dict, None] = None) -> APIRouter:
    if deps is not None:
        configure(deps)
    r = APIRouter(tags=["F9 대화"])

    @r.get("/api/chat/status")
    def get_status() -> dict:
        return status_summary()

    @r.get("/api/chat/messages")
    def get_messages(before: Optional[str] = None,
                     limit: int = Query(C.HISTORY_LIMIT, ge=1, le=C.HISTORY_MAX)) -> dict:
        with store.connect() as con:
            return service.messages(con, _deps, before, limit)

    @r.delete("/api/chat/messages", status_code=204)
    def delete_messages() -> Response:
        with store.connect() as con:
            service.clear(con)
        return Response(status_code=204)

    @r.post("/api/chat")
    def post_chat(body: Any = Body(default=None)) -> dict:
        text = body.get("text") if isinstance(body, dict) else None
        if text is not None and not isinstance(text, str):
            raise HTTPException(422, "text 는 문자열입니다")

        def run():
            with store.connect() as con:
                return service.turn(con, _deps, text)
        return _errors(run)

    @r.post("/api/chat/actions/{action_id}/confirm")
    def post_confirm(action_id: str, body: Any = Body(default=None)) -> dict:
        def run():
            with store.connect() as con:
                return service.confirm(con, _deps, action_id, body if isinstance(body, dict) else {})
        return _errors(run)

    @r.post("/api/chat/actions/{action_id}/cancel")
    def post_cancel(action_id: str) -> dict:
        def run():
            with store.connect() as con:
                return service.cancel(con, _deps, action_id)
        return _errors(run)

    return r
