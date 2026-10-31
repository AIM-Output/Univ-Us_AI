# -*- coding: utf-8 -*-
"""학습(train.py)·평가(eval.py)·데이터 생성(prepare_f9.py)이 같이 쓰는 도우미.

핵심은 render_chat() 하나다 — 학습 텍스트와 평가 프롬프트가 **같은 함수**로 만들어져야
모델 비교가 성립한다(개발계획서 7.3 '동일 프롬프트'). 모델 이름은 이 파일에 없다.

도구 호출 표기는 모델과 무관하게 한 가지로 통일한다:
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Windows 콘솔(cp949)에서 한국어 출력이 깨지지 않게 — 이 모듈을 import 하는 모든 스크립트에 적용된다
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 파이프·구버전이면 그대로 둔다
        pass

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"

# 폴백 모드에서 system 메시지에 붙이는 안내 (한국어)
FALLBACK_TOOL_INSTRUCTION = (
    "아래 도구 중 하나가 필요하면 다음 형식으로만 답하세요:\n"
    f'{TOOL_CALL_OPEN}{{"name": "<도구 이름>", "arguments": {{<인자 객체>}}}}{TOOL_CALL_CLOSE}\n'
    "도구가 필요 없으면 한국어로 짧게 답하세요. 도구 결과에 없는 숫자는 말하지 않습니다.\n\n"
    "# 도구\n"
)

_ISO_DT = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}):(\d{2})(?::\d{2})?)?$")
_THINK = re.compile(r"<think>.*?(?:</think>\s*|\Z)", re.DOTALL)     # 닫히지 않은 <think> 꼬리도 지운다
_TOOL_CALL = re.compile(re.escape(TOOL_CALL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_CLOSE), re.DOTALL)


# ── 파일 ────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: str | Path, limit: Optional[int] = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_tools(path: str | Path) -> tuple[list[dict], dict[str, str]]:
    """tools/f9_tools.json → (모델에 넘길 도구 목록, 이름→x-kind).  x-kind 는 떼어 낸다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data["tools"] if isinstance(data, dict) else data
    tools, kinds = [], {}
    for t in raw:
        t = dict(t)
        kinds[t["function"]["name"]] = t.pop("x-kind", "read")
        tools.append(t)
    return tools, kinds


def required_args(tools: list[dict]) -> dict[str, list[str]]:
    return {t["function"]["name"]: list((t["function"].get("parameters") or {}).get("required") or [])
            for t in tools}


# ── 도구 호출 표기 ───────────────────────────────────────────

def tool_call_text(name: str, arguments: Any) -> str:
    """"name" 을 먼저, 그다음 "arguments"(키 정렬) — Qwen 네이티브 tool-call 순서와 같다."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    if isinstance(arguments, dict):
        arguments = json.loads(json.dumps(arguments, ensure_ascii=False, sort_keys=True))   # 키 정렬(중첩 포함)
    body = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}"


def flatten_tool_calls(messages: list[dict]) -> list[dict]:
    """OpenAI 모양의 assistant.tool_calls 를 content 문자열(<tool_call>…)로 바꾼다. 다른 메시지는 그대로."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            parts = [tool_call_text(tc["function"]["name"], tc["function"].get("arguments", {}))
                     for tc in m["tool_calls"]]
            content = (m.get("content") or "").strip()
            text = ("\n".join(parts) if not content else content + "\n" + "\n".join(parts))
            out.append({"role": "assistant", "content": text})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def inject_tools_system(messages: list[dict], tools: list[dict]) -> list[dict]:
    """폴백 모드 — 도구 목록을 system 에 JSON 으로 넣고, tool 역할 메시지는 user 의 <tool_response> 로 바꾼다."""
    tool_lines = "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools)
    system_extra = FALLBACK_TOOL_INSTRUCTION + tool_lines
    out: list[dict] = []
    has_system = bool(messages) and messages[0].get("role") == "system"
    if has_system:
        out.append({"role": "system", "content": (messages[0].get("content") or "") + "\n\n" + system_extra})
        rest = messages[1:]
    else:
        out.append({"role": "system", "content": system_extra})
        rest = messages
    for m in rest:
        if m.get("role") == "tool":
            out.append({"role": "user", "content": f"{TOOL_RESPONSE_OPEN}{m.get('content') or ''}{TOOL_RESPONSE_CLOSE}"})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    # 연속된 같은 역할(user, user)을 템플릿이 거부하는 경우가 있어 합친다
    merged: list[dict] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(m)
    return merged


def manual_render(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """chat template 이 아예 없는 토크나이저용 최소 형식."""
    parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)


def _template_kwargs() -> dict:
    # Qwen3 계열: 생각 블록을 끈다. 다른 템플릿은 모르는 변수라 무시한다.
    return {"enable_thinking": False}


def detect_render_mode(tok: Any, tools: list[dict]) -> str:
    """native(템플릿이 tools 를 받음) · fallback(system 주입) · manual(템플릿 없음). 토크나이저에 캐시."""
    cached = getattr(tok, "_univus_render_mode", None)
    if cached:
        return cached
    probe = [{"role": "system", "content": "probe"}, {"role": "user", "content": "probe"}]
    first_name = tools[0]["function"]["name"] if tools else ""
    mode = "manual"
    if getattr(tok, "chat_template", None):
        try:
            text = tok.apply_chat_template(probe, tools=tools, tokenize=False, add_generation_prompt=True,
                                           **_template_kwargs())
            mode = "native" if (first_name and first_name in text) else "fallback"
        except Exception:  # noqa: BLE001 — tools 를 모르는 템플릿
            try:
                tok.apply_chat_template(inject_tools_system(probe, tools), tokenize=False,
                                        add_generation_prompt=True, **_template_kwargs())
                mode = "fallback"
            except Exception:  # noqa: BLE001
                mode = "manual"
    try:
        tok._univus_render_mode = mode
    except Exception:  # noqa: BLE001
        pass
    return mode


def render_chat(tok: Any, messages: list[dict], tools: list[dict], add_generation_prompt: bool = False) -> str:
    """메시지 → 학습/평가 텍스트. 모드는 detect_render_mode() 로 한 번 정해 일관되게 쓴다."""
    msgs = flatten_tool_calls(messages)
    mode = detect_render_mode(tok, tools)
    if mode == "native":
        return tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                       add_generation_prompt=add_generation_prompt, **_template_kwargs())
    injected = inject_tools_system(msgs, tools)
    if mode == "fallback":
        return tok.apply_chat_template(injected, tokenize=False,
                                       add_generation_prompt=add_generation_prompt, **_template_kwargs())
    return manual_render(injected, add_generation_prompt)


# ── 파싱·채점 ────────────────────────────────────────────────

def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def _first_json_object(text: str) -> Optional[dict]:
    """문자열 안에서 첫 번째 균형 잡힌 JSON 객체를 찾는다 (따옴표 안의 중괄호는 무시)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def parse_tool_call(text: str) -> Optional[dict]:
    """생성 결과 → {"name":…, "arguments":{…}} 또는 None(도구 호출 없음)."""
    text = strip_think(text or "")
    m = _TOOL_CALL.search(text)
    obj = _first_json_object(m.group(1)) if m else None
    if obj is None and (TOOL_CALL_OPEN in text or '"name"' in text):
        obj = _first_json_object(text)
    if not obj or "name" not in obj:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return {"name": str(obj["name"]), "arguments": args if isinstance(args, dict) else {}}


def split_prompt_and_target(messages: list[dict]) -> tuple[list[dict], Optional[dict]]:
    """첫 assistant 턴 앞까지를 프롬프트로, 그 assistant 턴을 정답으로."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            return messages[:i], m
    return messages, None


def expected_call(msg: Optional[dict]) -> Optional[dict]:
    if not msg:
        return None
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]["function"]
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        return {"name": tc["name"], "arguments": args}
    return parse_tool_call(msg.get("content") or "")


def normalize_value(v: Any) -> Any:
    """채점용 정규화 — 날짜는 ISO 표준형, 문자열은 strip, 리스트는 정렬, 정수형 실수는 int."""
    if isinstance(v, str):
        s = v.strip()
        m = _ISO_DT.match(s)
        if m:
            return f"{m.group(1)}T{m.group(2)}:{m.group(3)}" if m.group(2) else m.group(1)
        return s
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return sorted((normalize_value(x) for x in v), key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    if isinstance(v, dict):
        return {str(k): normalize_value(x) for k, x in v.items()}
    return v


def lib_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


if __name__ == "__main__":
    # 자가 점검 — 네트워크·모델 없이 파서와 정규화만
    sample = '<think>\n\n</think>\n\n<tool_call>{"arguments": {"from": "2026-10-27", "to": "2026-10-27"}, "name": "list_events"}</tool_call>'
    assert parse_tool_call(sample) == {"name": "list_events", "arguments": {"from": "2026-10-27", "to": "2026-10-27"}}
    assert parse_tool_call('{"name":"get_graduation","arguments":{}}') == {"name": "get_graduation", "arguments": {}}
    assert parse_tool_call("졸업까지 32학점 남았습니다.") is None
    assert normalize_value("2026-10-27 19:00:00") == "2026-10-27T19:00"
    assert normalize_value({"kinds": ["user", "deadline"]}) == {"kinds": ["deadline", "user"]}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                                                  "function": {"name": "navigate", "arguments": '{"route": "/academic"}'}}]},
            {"role": "tool", "content": '{"ok": true}'}, {"role": "assistant", "content": "열었습니다."}]
    flat = flatten_tool_calls(msgs)
    assert flat[2]["content"] == tool_call_text("navigate", {"route": "/academic"})
    assert tool_call_text("list_events", {"to": "b", "from": "a"}) == \
        '<tool_call>{"name": "list_events", "arguments": {"from": "a", "to": "b"}}</tool_call>'
    inj = inject_tools_system(flat, [{"type": "function", "function": {"name": "navigate", "parameters": {}}}])
    assert inj[0]["role"] == "system" and "navigate" in inj[0]["content"]
    assert [m["role"] for m in inj] == ["system", "user", "assistant", "user", "assistant"]
    print("common.py 자가 점검 OK")
