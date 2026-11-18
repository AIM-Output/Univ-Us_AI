# -*- coding: utf-8 -*-
"""data_src/ 의 TSV(사람이 쓴 실제 문장 · 말투 틀)를 읽는 공통 도우미. 표준 라이브러리만 쓴다.

make_real.py · prepare_f9_v3.py · scripts/peek_*.py 가 같이 쓴다. 오류 메시지는 `파일:줄: 오류: 이유` 한 줄로 낸다 —
처음 쓰는 사람이 어느 줄을 고치면 되는지 바로 알 수 있게.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# 파일 이름(사람) → id 접두. 여기 없는 파일 이름(예시 파일 등)은 접두를 검사하지 않는다.
OWNERS = {"haram": "hr", "jiuk": "ju", "subin": "sb"}
WD = "월화수목금토일"
D1 = dt.timedelta(days=1)


@dataclass
class Line:
    path: Path
    no: int                 # 파일 안의 줄 번호 (1부터)
    cells: list[str]

    @property
    def where(self) -> str:
        return f"{rel(self.path)}:{self.no}"


class Report:
    """오류(고쳐야 통과)와 경고(확인만 하면 됨)를 모은다. 출력은 파일·줄 순서."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self._all: list[tuple[str, int, int, str]] = []

    def _add(self, where: str, level: int, text: str) -> None:
        m = re.match(r"^(.*):(\d+)$", where)
        self._all.append((m.group(1), int(m.group(2)), level, text) if m else (where, 0, level, text))

    def error(self, where: str, msg: str) -> None:
        text = f"{where}: 오류: {msg}"
        self.errors.append(text)
        self._add(where, 0, text)

    def warn(self, where: str, msg: str) -> None:
        text = f"{where}: 경고: {msg}"
        self.warnings.append(text)
        self._add(where, 1, text)

    def dump(self) -> None:
        for _f, _n, _lv, text in sorted(self._all, key=lambda x: x[:3]):
            print(text)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def unquote_cell(s: str) -> str:
    """스프레드시트가 칸을 "…" 로 감싸고 안쪽 따옴표를 "" 로 겹쳐 내보낸 경우 원래대로 돌린다."""
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"' and '""' in s[1:-1]:
        return s[1:-1].replace('""', '"')
    return s


def read_tsv(path: Path, header: list[str], rep: Report, min_cols: int) -> list[Line]:
    """머리글 한 줄 + 데이터 줄. 빈 줄과 # 으로 시작하는 줄은 건너뛴다. 모자란 뒤 칸은 빈 문자열로 채운다."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        rep.error(rel(path), "UTF-8 로 읽을 수 없습니다 — 파일을 UTF-8 로 저장해 주세요")
        return []
    out: list[Line] = []
    seen_header = False
    for no, raw in enumerate(text.split("\n"), 1):
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cells = [unquote_cell(c.strip()) for c in line.split("\t")]
        where = f"{rel(path)}:{no}"
        if not seen_header:
            seen_header = True
            if len(cells) < min_cols or cells != header[:len(cells)]:
                rep.error(where, "첫 줄(머리글)은 탭으로 나눈 `" + " · ".join(header) + "` 이어야 합니다")
            continue
        if "\t" not in line:
            rep.error(where, "칸 사이에 탭이 없습니다 — 구글 시트에 쓰고 행을 복사해 붙이면 탭이 그대로 들어갑니다")
            continue
        while len(cells) > min_cols and cells[-1] == "":       # 줄 끝에 남은 빈 칸은 무시
            cells.pop()
        if len(cells) < min_cols or len(cells) > len(header):
            rep.error(where, f"칸이 {len(cells)}개입니다 — 탭으로 나눈 {min_cols}~{len(header)}칸이어야 합니다 ({' · '.join(header)})")
            continue
        cells += [""] * (len(header) - len(cells))
        out.append(Line(path, no, cells))
    if not seen_header:
        rep.error(rel(path), "빈 파일입니다 — 첫 줄에 머리글이 있어야 합니다: " + " · ".join(header))
    return out


def list_sources(folder: Path, given: list[str]) -> list[Path]:
    """직접 준 파일이 있으면 그것만, 없으면 폴더의 *.tsv (이름이 _ 로 시작하는 파일 제외)를 이름순으로."""
    if given:
        return [Path(g) for g in given]
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.tsv") if not p.name.startswith("_"))


def owner_prefix(path: Path) -> str | None:
    return OWNERS.get(path.stem)


def real_users(folder: Path) -> set[str]:
    """data_src/real/*.tsv 의 문장 칸만 느슨하게 읽는다 — 생성 데이터가 시험 문장과 겹치지 않게 막는 데 쓴다."""
    users: set[str] = set()
    for p in list_sources(folder, []):
        try:
            lines = p.read_text(encoding="utf-8-sig").split("\n")
        except (OSError, UnicodeDecodeError):
            continue
        for i, raw in enumerate(l for l in lines if l.strip() and not l.lstrip().startswith("#")):
            cells = [unquote_cell(c.strip()) for c in raw.rstrip("\r").split("\t")]
            if i > 0 and len(cells) >= 3 and cells[2]:
                users.add(cells[2])
    return users


# ── 사람이 읽기 쉬운 정답 표기 — 날짜 옆에 요일을 붙인다 ─────────────

def md(d: dt.date) -> str:
    return f"{d.month}/{d.day}({WD[d.weekday()]})"


def _when(s: str) -> str:
    """'2026-11-26T20:00' → '11/26(목) 20:00' · '2026-11-26' → '11/26(목)'."""
    if "T" in s:
        t = dt.datetime.strptime(s, "%Y-%m-%dT%H:%M")
        return f"{md(t.date())} {t:%H:%M}"
    return md(dt.date.fromisoformat(s))


def describe_call(name: str, args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return f"{name} {args}"
    a = args or {}
    try:
        if name == "list_events":
            da, db = dt.date.fromisoformat(a["from"]), dt.date.fromisoformat(a["to"])
            span = md(da) if da == db else f"{md(da)}~{md(db)}"
            return f"list_events {span}" + (f" · {', '.join(a['kinds'])}" if a.get("kinds") else "")
        if name == "create_event":
            if a["allDay"]:
                s, e = dt.date.fromisoformat(a["start"]), dt.date.fromisoformat(a["end"]) - D1
                when = f"{md(s)} 종일" if s == e else f"{md(s)}~{md(e)} 종일"
            else:
                s = dt.datetime.strptime(a["start"], "%Y-%m-%dT%H:%M")
                e = dt.datetime.strptime(a["end"], "%Y-%m-%dT%H:%M")
                when = f"{md(s.date())} {s:%H:%M}~" + (f"{e:%H:%M}" if e.date() == s.date() else f"{md(e.date())} {e:%H:%M}")
            rep_s = f" · 매주 {a['repeat'].split(':')[1]}" if a.get("repeat") else ""
            return f"create_event {when} · {a['title']} · {a['category']}{rep_s}"
        if name == "update_event":
            p = a["patch"]
            parts = [f"{k}={_when(v) if k in ('start', 'end') else v}" for k, v in p.items()]
            return f"update_event id {a['id']} · " + ", ".join(parts)
        if name == "delete_event":
            return f"delete_event id {a['id']}"
    except (KeyError, ValueError, TypeError, AttributeError):
        pass
    return f"{name} {json.dumps(a, ensure_ascii=False)}"


_SYS_NOW = re.compile(r"^오늘은 (\d{4}-\d{2}-\d{2})\([월화수목금토일]\) (\d{2}:\d{2}) 입니다\.")


def row_now(row: dict) -> str:
    """행의 system 메시지에서 기준시각을 뽑아 '11/18(수) 14:00' 로."""
    msgs = row.get("messages") or []
    m = _SYS_NOW.match(msgs[0].get("content") or "") if msgs and msgs[0].get("role") == "system" else None
    if not m:
        return ""
    return f"{md(dt.date.fromisoformat(m.group(1)))} {m.group(2)}"


def row_user(row: dict) -> str:
    return next((m.get("content") or "" for m in row.get("messages") or [] if m.get("role") == "user"), "")


def row_answer(row: dict) -> str:
    """assistant 턴들을 ' → ' 로 이어 한 줄로. 도구 결과(tool) 턴은 건너뛴다."""
    parts = []
    for m in row.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):
            fn = m["tool_calls"][0]["function"]
            parts.append(describe_call(fn["name"], fn.get("arguments", {})))
        else:
            parts.append("글: " + (m.get("content") or ""))
    return " → ".join(parts)


def cell(s: Any) -> str:
    """마크다운 표 칸에 넣을 수 있게."""
    return str(s).replace("|", "\\|").replace("\n", " ")
