#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""사람이 쓴 실제 문장(data_src/real/*.tsv) → 시험 데이터 real_dev.jsonl · real_test.jsonl.

  python src/make_real.py            # 검사 → data/f9/real_dev.jsonl · real_test.jsonl
  python src/make_real.py --check    # 검사만 (오류가 있으면 종료 코드 1)
  python src/make_real.py --show     # 검수용 표 — 날짜 옆에 요일이 붙는다
  python src/make_real.py --check data_src/examples/real_example.tsv     # 파일을 직접 지정

TSV 칸: id · 기준시각 · 문장 · 정답 · 메모   (docs/F9_학습_가이드.md 7-5)
  정답 = `거절` 또는 `도구이름 {JSON 인자}`. 수정·삭제는 첫 단계 list_events 만 적는다.

· 행 형식은 prepare_f9 와 같다(system_message(기준시각) + user + assistant) — eval.py 가 그대로 채점한다.
· 학습에는 넣지 않는다. 시험 전용이다.
· dev/test 는 id 의 sha1 로 나눈다 — 행을 더해도 이미 있는 행의 소속은 바뀌지 않는다.
· 오류는 고쳐야 통과하고, 경고는 확인만 하면 된다(--strict 를 주면 경고도 실패).
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_jsonl, load_tools, write_jsonl  # noqa: E402
from datasrc import (D1, ROOT, WD, Line, Report, cell, describe_call, list_sources, md, owner_prefix,  # noqa: E402
                     read_tsv, rel)
import prepare_f9  # noqa: E402
import prepare_f9_v2 as v2  # noqa: E402  — 날짜·시각을 문장에서 다시 읽는 검산기와 norm_key 를 빌려 쓴다

HEADER = ["id", "기준시각", "문장", "정답", "메모"]
STD_NOWS = ("2026-11-16 09:00", "2026-11-18 14:00", "2026-11-20 18:00", "2026-11-22 22:00")
REFUSE = "거절"
# 거절 행의 assistant 내용 — 채점은 예측에 거절 표현이 있는지만 본다(eval.score_turn). 이 글은 실패 사례 표에 보이는 설명이다.
EXPECT_REFUSAL = "(거절) 도구를 부르지 않고 '…수 없습니다' 처럼 답해야 합니다."
SECOND_STEP = {"update_event", "delete_event"}
WD_CODE = prepare_f9.WD_CODE
_ID =re.compile(r"^([a-z]{2,3})-(\d{3})$")
_ANSWER = re.compile(r"^([A-Za-z_]+)\s*(\{.*\})$", re.S)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_WRITE_VERB = re.compile(r"옮겨|바꿔|변경|수정|고쳐|미뤄|당겨|늦춰|취소|지워|삭제|빼|없애|날려")
_RANGE_MARK = re.compile(r"부터|까지|~|에서")
_PHONE = re.compile(r"01\d[-\s]?\d{3,4}[-\s]?\d{4}")
_STUDENT_NO = re.compile(r"(?<!\d)(?:19|20)\d{6}(?!\d)")
_DURS = sorted(v2.DURS, key=lambda x: -len(x[0]))          # 긴 표현부터 찾는다 ('1시간 반' 이 '1시간' 보다 먼저)


@dataclass
class Item:
    line: Line
    id: str
    now: dt.datetime
    user: str
    tool: Optional[str]          # None = 거절
    args: Optional[dict]
    memo: str
    seen_in_train: bool = False


def hm(t: tuple[int, int]) -> str:
    return f"{t[0]:02d}:{t[1]:02d}"


# ── 정답 칸 읽기 ─────────────────────────────────────────────

def parse_answer(raw: str) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """(도구 이름 | None(거절), 인자, 오류 메시지)."""
    s = raw.strip()
    if s == REFUSE:
        return None, None, None
    if not s:
        return None, None, "정답 칸이 비어 있습니다 — `거절` 또는 `도구이름 {JSON 인자}` 를 적어 주세요"
    m = _ANSWER.match(s)
    if not m:
        return None, None, ("정답은 `거절` 또는 `도구이름 {JSON 인자}` 꼴이어야 합니다 "
                            "(예: get_graduation {} · navigate {\"route\": \"/graduation\"})")
    name, body = m.group(1), m.group(2)
    try:
        args = json.loads(body)
    except ValueError as e:
        hint = ""
        if re.search("[“”‘’]", body):
            hint = " — 둥근 따옴표(“ ”)가 있습니다. 곧은 큰따옴표(\")로 바꿔 주세요"
        elif "'" in body:
            hint = " — 작은따옴표(')가 있습니다. 큰따옴표(\")를 써야 합니다"
        elif re.search(r"\b(True|False)\b", body):
            hint = " — true / false 는 소문자로 씁니다"
        elif re.search(r",\s*[}\]]", body):
            hint = " — 마지막 값 뒤에 쉼표가 남아 있습니다"
        return name, None, f"정답의 JSON 을 읽을 수 없습니다 ({e.msg}, {e.pos}번째 글자){hint}"
    if not isinstance(args, dict):
        return name, None, "인자는 { } 로 감싼 객체여야 합니다"
    return name, args, None


# ── 도구 정의(스키마) 검사 ────────────────────────────────────

def check_schema(name: str, args: dict, schema: dict) -> list[str]:
    errs: list[str] = []
    props = schema.get("properties") or {}
    for r in schema.get("required") or []:
        if r not in args:
            errs.append(f"{name} 에 필수 인자 \"{r}\" 가 없습니다")
    for k, v in args.items():
        if k not in props:
            errs.append(f"\"{k}\" 는 {name} 에 없는 인자입니다 (쓸 수 있는 인자: {', '.join(props) or '없음'})")
            continue
        p = props[k]
        t = p.get("type")
        if "enum" in p and v not in p["enum"]:
            errs.append(f"\"{k}\" 값 {json.dumps(v, ensure_ascii=False)} 은(는) 쓸 수 없습니다 (가능: {' · '.join(map(str, p['enum']))})")
        elif t == "array":
            allowed = (p.get("items") or {}).get("enum")
            if not isinstance(v, list) or not v:
                errs.append(f"\"{k}\" 값은 하나 이상 든 목록이어야 합니다 (예: [\"deadline\"]). 필요 없으면 \"{k}\" 자체를 빼세요")
            else:
                bad = [x for x in v if allowed and x not in allowed]
                if bad:
                    errs.append(f"\"{k}\" 의 {json.dumps(bad, ensure_ascii=False)} 은(는) 쓸 수 없습니다 (가능: {' · '.join(allowed)})")
                if len(set(map(str, v))) != len(v):
                    errs.append(f"\"{k}\" 에 같은 값이 두 번 들어 있습니다")
        elif t == "boolean" and not isinstance(v, bool):
            errs.append(f"\"{k}\" 값은 따옴표 없이 true 또는 false 로 씁니다")
        elif t == "string" and (not isinstance(v, str) or not v.strip()):
            errs.append(f"\"{k}\" 값이 비어 있거나 글자가 아닙니다")
        elif t == "object" and (not isinstance(v, dict) or not v):
            errs.append(f"\"{k}\" 값은 비어 있지 않은 객체여야 합니다")
    return errs


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _duration_in(user: str) -> Optional[int]:
    for phrase, mins in _DURS:
        if phrase in user:
            return mins
    return None


# ── 도구별 검사 — (오류, 경고) ────────────────────────────────

def check_list_events(it: Item) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    a = it.args or {}
    f, t = a.get("from"), a.get("to")
    for k, v in (("from", f), ("to", t)):
        if isinstance(v, str) and not _DATE.match(v):
            errs.append(f"\"{k}\" 는 YYYY-MM-DD 꼴이어야 합니다 (지금: {v})")
    if errs or not isinstance(f, str) or not isinstance(t, str):
        return errs, warns
    try:
        da, db = dt.date.fromisoformat(f), dt.date.fromisoformat(t)
    except ValueError:
        return [f"없는 날짜입니다: {f} ~ {t}"], warns
    if da > db:
        errs.append(f"\"from\"({f}) 이 \"to\"({t}) 보다 늦습니다")
        return errs, warns
    pr = v2.parse_range(it.user, it.now.date())
    if pr and pr != (da, db):
        warns.append(f"문장에서 읽은 기간은 {md(pr[0])}~{md(pr[1])} 인데 정답은 {md(da)}~{md(db)} 입니다")
    if "kinds" not in a and re.search(r"마감|데드라인", it.user) and not _WRITE_VERB.search(it.user):
        warns.append("문장에 '마감'이 있는데 kinds 가 없습니다 — 마감 조회면 \"kinds\": [\"deadline\"]")
    return errs, warns


def check_create_event(it: Item) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    a = it.args or {}
    now, today = it.now, it.now.date()
    start, end, all_day = a.get("start"), a.get("end"), a.get("allDay")
    if not (isinstance(start, str) and isinstance(end, str) and isinstance(all_day, bool)):
        return errs, warns                                  # 스키마 검사가 이미 알려 준다
    pat = _DATE if all_day else _DATETIME
    shape = "YYYY-MM-DD (종일이라 시각 없이)" if all_day else "YYYY-MM-DDTHH:MM (시각 일정이라 T 와 시:분까지, 초는 없이)"
    for k, v in (("start", start), ("end", end)):
        if not pat.match(v):
            errs.append(f"\"{k}\" 는 {shape} 꼴이어야 합니다 (지금: {v})")
    if errs:
        return errs, warns
    try:
        s = dt.datetime.strptime(start, "%Y-%m-%d" if all_day else "%Y-%m-%dT%H:%M")
        e = dt.datetime.strptime(end, "%Y-%m-%d" if all_day else "%Y-%m-%dT%H:%M")
    except ValueError:
        return [f"없는 날짜·시각입니다: {start} ~ {end}"], warns
    if e <= s:
        errs.append("\"end\" 가 \"start\" 와 같거나 앞섭니다" + (" — 종일 일정의 end 는 마지막 날의 다음 날입니다 (11/20 하루 → end 2026-11-21)" if all_day else ""))
        return errs, warns
    if s <= now:
        errs.append(f"start 가 기준시각({now:%Y-%m-%d %H:%M})보다 뒤여야 합니다 — 지난 시각·오늘 종일에는 등록하지 않습니다")
    title = a.get("title")
    if isinstance(title, str) and _squash(title) not in _squash(it.user):
        warns.append(f"제목 \"{title}\" 이(가) 문장에 그대로 없습니다 — 제목은 문장 속 일정 이름 그대로 씁니다")

    repeat = a.get("repeat")
    if repeat is not None:
        m = re.fullmatch(r"weekly:(MO|TU|WE|TH|FR|SA|SU)", str(repeat))
        if not m:
            errs.append("\"repeat\" 는 weekly:MO 처럼 씁니다 (MO TU WE TH FR SA SU)")
            return errs, warns
        if WD_CODE[s.weekday()] != m.group(1):
            errs.append(f"repeat 은 {m.group(1)} 인데 start({md(s.date())})는 {WD_CODE[s.weekday()]} 입니다 — 요일을 맞춰 주세요")
        if not (today < s.date() <= today + 7 * D1):
            errs.append(f"반복 일정의 start 는 내일부터 7일 안의 첫 그 요일이어야 합니다 (지금: {md(s.date())})")
        wm = re.search(r"([월화수목금토일])요일", it.user)
        if wm and WD.index(wm.group(1)) != s.weekday():
            warns.append(f"문장은 {wm.group(1)}요일인데 start 는 {md(s.date())} 입니다")
    else:
        ds = v2.parse_dates(it.user, today)
        if ds and ds[0] != s.date():
            warns.append(f"문장에서 읽은 날짜는 {md(ds[0])} 인데 start 는 {md(s.date())} 입니다")

    tms = v2.parse_times(it.user)
    if all_day:
        if tms:
            warns.append(f"문장에 시각({hm(tms[0])})이 있는데 종일(allDay: true)로 적었습니다")
        if (e - s).days != 1 and not _RANGE_MARK.search(it.user):
            warns.append(f"하루짜리 종일 일정이면 end 는 start 다음 날입니다 (지금: {start} ~ {end})")
    else:
        if e.date() != s.date():
            warns.append("start 와 end 의 날짜가 다릅니다 — 시각 일정은 같은 날 안에서 끝나야 합니다")
        if tms and tms[0] != (s.hour, s.minute):
            warns.append(f"문장에서 읽은 시각은 {hm(tms[0])} 인데 start 는 {s:%H:%M} 입니다")
        if len(tms) >= 2:
            if tms[1] != (e.hour, e.minute):
                warns.append(f"문장에서 읽은 끝 시각은 {hm(tms[1])} 인데 end 는 {e:%H:%M} 입니다")
        else:
            want = _duration_in(it.user) or 60
            got = int((e - s).total_seconds() // 60)
            if got != want:
                why = "문장에서 읽은 길이" if _duration_in(it.user) else "길이를 말하지 않으면"
                warns.append(f"{why} {want}분인데 정답은 {got}분입니다")
    return errs, warns


def check_item(it: Item, schemas: dict[str, dict]) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    if v2.bare_hour_hits(it.user):
        errs.append("오전·오후 없는 1~12시가 있습니다 — 아침·오전·점심·낮·오후·저녁·밤을 붙이거나 13~23시로 써 주세요")
    if _PHONE.search(it.user) or _STUDENT_NO.search(it.user):
        errs.append("전화번호·학번처럼 보이는 숫자가 있습니다 — 공개 저장소라 개인정보는 쓰지 않습니다")
    if len(it.user) > 80:
        warns.append(f"문장이 {len(it.user)}자로 깁니다 — 한 문장에 부탁 하나만 써 주세요")
    if it.tool is None:
        return errs, warns
    if it.tool in SECOND_STEP:
        errs.append(f"{it.tool} 은(는) 적지 않습니다 — 수정·삭제는 첫 단계인 list_events {{\"from\": 그날, \"to\": 그날}} 만 적습니다")
        return errs, warns
    if it.tool not in schemas:
        errs.append(f"\"{it.tool}\" 은(는) 없는 도구입니다 (도구: {' · '.join(schemas)})")
        return errs, warns
    errs += check_schema(it.tool, it.args or {}, schemas[it.tool])
    a = it.args or {}
    if it.tool == "list_events":
        e2, w2 = check_list_events(it)
        errs += e2
        warns += w2
    elif it.tool == "create_event":
        e2, w2 = check_create_event(it)
        errs += e2
        warns += w2
    elif it.tool == "get_exams":
        if a.get("upcoming") is False:
            errs.append("\"upcoming\": false 는 쓰지 않습니다 — 다가오는 시험이면 true, 아니면 {} 로 비웁니다")
    elif it.tool == "get_attendance":
        c = a.get("course")
        if isinstance(c, str) and _squash(c) not in _squash(it.user):
            warns.append(f"과목 \"{c}\" 이(가) 문장에 그대로 없습니다 — 과목 이름은 문장에 쓴 그대로 적습니다")
    return errs, warns


# ── TSV → Item ───────────────────────────────────────────────

def load_items(paths: list[Path], schemas: dict[str, dict], rep: Report) -> list[Item]:
    items: list[Item] = []
    seen_id: dict[str, str] = {}
    seen_user: dict[str, tuple[str, dt.datetime]] = {}
    for path in paths:
        if not path.exists():
            rep.error(rel(path), "파일이 없습니다")
            continue
        prefix = owner_prefix(path)
        for ln in read_tsv(path, HEADER, rep, min_cols=4):
            rid, now_s, user, answer, memo = ln.cells
            bad = False
            m = _ID.match(rid)
            if not m:
                rep.error(ln.where, f"id \"{rid}\" 는 `이니셜-번호 3자리` 꼴이어야 합니다 (예: sb-001)")
                bad = True
            elif prefix and m.group(1) != prefix:
                rep.error(ln.where, f"이 파일의 id 는 {prefix}- 로 시작해야 합니다 (지금: {rid})")
                bad = True
            elif rid in seen_id:
                rep.error(ln.where, f"id {rid} 은(는) 이미 있습니다 ({seen_id[rid]}) — 한 번 쓴 번호는 다시 쓰지 않습니다")
                bad = True
            else:
                seen_id[rid] = ln.where
            try:
                now = dt.datetime.strptime(now_s, "%Y-%m-%d %H:%M")
            except ValueError:
                rep.error(ln.where, f"기준시각 \"{now_s}\" 는 `YYYY-MM-DD HH:MM` 꼴이어야 합니다 (예: 2026-11-18 14:00). "
                                    "구글 시트라면 칸 서식을 '일반 텍스트'로 바꿔 주세요")
                now, bad = None, True
            else:
                if now_s not in STD_NOWS:
                    rep.warn(ln.where, f"기준시각 {now_s} 는 정해 둔 네 개가 아닙니다 ({' · '.join(STD_NOWS)})")
            if not user:
                rep.error(ln.where, "문장 칸이 비어 있습니다")
                bad = True
            tool, args, err = parse_answer(answer)
            if err:
                rep.error(ln.where, err)
                bad = True
            if bad or now is None:
                continue
            it = Item(ln, rid, now, user, tool, args, memo)
            errs, warns = check_item(it, schemas)
            for e in errs:
                rep.error(ln.where, e)
            for w in warns:
                rep.warn(ln.where, w)
            if user in seen_user:
                w0, n0 = seen_user[user]
                if n0 == now:
                    rep.error(ln.where, f"같은 문장·같은 기준시각이 이미 있습니다 ({w0})")
                    errs.append("dup")
                else:
                    rep.warn(ln.where, f"같은 문장이 이미 있습니다 ({w0}) — 되도록 다른 말로 써 주세요")
            else:
                seen_user[user] = (ln.where, now)
            if memo.strip() == "?":
                rep.warn(ln.where, "메모에 ? 가 있습니다 — 검수할 때 정답을 같이 확인해 주세요")
            if not errs:
                items.append(it)
    return items


def check_overlap(items: list[Item], data_dir: Path, rep: Report) -> None:
    """학습 데이터와 겹치는지 — 똑같은 문장은 시험 가치가 떨어져 경고하고, meta.seen_in_train 으로 표시해 둔다."""
    path = data_dir / "train.jsonl"
    if not path.exists():
        print(f"(알림) {rel(path)} 이 없어 학습 데이터와의 겹침 검사를 건너뜁니다 — prepare_f9.py 를 먼저 돌리면 검사합니다)")
        return
    users = {m["content"] for r in load_jsonl(path) for m in r["messages"] if m["role"] == "user"}
    norm = {v2.norm_key(u) for u in users}
    for it in items:
        if it.user in users:
            it.seen_in_train = True
            rep.warn(it.line.where, "학습 데이터에 똑같은 문장이 있습니다 — 내 말투로 조금 바꾸면 더 좋은 시험 문제가 됩니다")
        elif v2.norm_key(it.user) in norm:
            it.seen_in_train = True
            rep.warn(it.line.where, "학습 데이터에 거의 같은 문장이 있습니다(어미·띄어쓰기만 다름)")


# ── 내보내기 ─────────────────────────────────────────────────

def split_of(rid: str) -> str:
    return "dev" if int(hashlib.sha1(rid.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "test"


def to_row(it: Item, tools: list[dict]) -> dict:
    msgs: list[dict] = [{"role": "system", "content": prepare_f9.system_message(it.now)},
                        {"role": "user", "content": it.user}]
    if it.tool is None:
        msgs.append({"role": "assistant", "content": EXPECT_REFUSAL})
        tool, kind = "none", "refusal"
    else:
        msgs.append({"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": it.tool, "arguments": json.dumps(it.args, ensure_ascii=False)}}]})
        tool, kind = it.tool, "single"
    meta = {"tool": tool, "template": f"real_{it.id.split('-')[0]}", "kind": kind, "source": "real"}
    if it.seen_in_train:
        meta["seen_in_train"] = True
    return {"id": it.id, "messages": msgs, "tools": tools, "meta": meta}


def describe(it: Item) -> str:
    """검수용 한 줄 — 날짜 옆에 요일을 붙인다."""
    return "거절" if it.tool is None else describe_call(it.tool, it.args or {})


def show(items: list[Item]) -> None:
    print("| id | 기준시각 | 문장 | 정답 | 메모 |")
    print("| --- | --- | --- | --- | --- |")
    for it in items:
        print(f"| {it.id} | {it.now:%m/%d}({WD[it.now.weekday()]}) {it.now:%H:%M} | {cell(it.user)} | {cell(describe(it))} | {cell(it.memo)} |")


def summary(items: list[Item]) -> None:
    by: dict[str, Counter] = {}
    for it in items:
        by.setdefault(it.id.split("-")[0], Counter())[it.tool or REFUSE] += 1
    names = sorted({k for c in by.values() for k in c})
    if not names:
        return
    print("\n| 누가 | " + " | ".join(names) + " | 합계 |")
    print("| --- | " + " | ".join("---:" for _ in names) + " | ---: |")
    for who in sorted(by):
        print(f"| {who} | " + " | ".join(str(by[who].get(n, 0)) for n in names) + f" | {sum(by[who].values())} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="검사할 TSV. 비우면 --src 폴더의 *.tsv 전부")
    ap.add_argument("--src", default=str(ROOT / "data_src" / "real"))
    ap.add_argument("--out", default=str(ROOT / "data" / "f9"))
    ap.add_argument("--tools", default=str(prepare_f9.TOOLS_PATH))
    ap.add_argument("--check", action="store_true", help="검사만 하고 파일을 만들지 않는다")
    ap.add_argument("--show", action="store_true", help="검수용 표를 출력한다 (파일을 만들지 않는다)")
    ap.add_argument("--strict", action="store_true", help="경고가 있어도 실패로 친다")
    args = ap.parse_args()

    tools, _ = load_tools(args.tools)
    schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
    rep = Report()
    paths = list_sources(Path(args.src), args.files)
    if not paths:
        print(f"TSV 가 없습니다: {args.src}")
        return 1
    items = load_items(paths, schemas, rep)
    out = Path(args.out)
    check_overlap(items, out, rep)

    if args.show:
        show(items)
    rep.dump()
    summary(items)
    n_err, n_warn = len(rep.errors), len(rep.warnings)
    print(f"\n파일 {len(paths)}개 · 통과한 문장 {len(items)}개 · 오류 {n_err} · 경고 {n_warn}")
    if n_err or (args.strict and n_warn):
        print("오류를 고친 뒤 다시 실행해 주세요." if n_err else "--strict: 경고를 확인해 주세요.")
        return 1
    if args.check or args.show:
        return 0
    if not items:
        print("문장이 아직 없어 파일을 만들지 않았습니다.")
        return 0
    rows = {"dev": [], "test": []}
    for it in items:
        rows[split_of(it.id)].append(to_row(it, tools))
    for name in ("dev", "test"):
        p = out / f"real_{name}.jsonl"
        write_jsonl(p, rows[name])
        print(f"{rel(p)}  {len(rows[name])}행  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
