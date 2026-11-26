#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""날짜 규칙 — 모델이 낸 도구 호출의 날짜를, 사용자 문장에서 코드로 계산한 날짜로 덮어쓴다 (모델은 그대로).

  python scripts/date_rule.py gold  data/f9/train.jsonl                                   # 정답에 규칙을 적용 — 정답을 바꾸면 규칙이 틀린 것
  python scripts/date_rule.py apply outputs/f9-r1/eval_adapter.json data/f9/eval.jsonl    # 모델 답에 적용해 다시 채점 (GPU 없이)
  python scripts/date_rule.py apply <eval 결과.json> <rows.jsonl> --label 이름 --hide     # --hide: 문장·답을 출력하지 않고 개수만

왜: 모델(f9-r1)은 정답 날짜가 그 달의 말일·1일이면 3번에 1번꼴로 틀린다(docs/eval/date_probe). 날짜 계산은 답이 하나로
정해진 일이라 코드가 한다. 요구사항정의서 F9 5장도 "LLM 이 준 날짜 문자열은 백엔드가 다시 검증한다" 고 정해 두었다.
이 파일은 그 규칙의 **검증용 원형**이다 — 앱(F9_Chat_agent)에 넣을 때는 apply_rule 과 같은 일을 하게 옮긴다.

규칙 (문장에서 날짜를 읽는 것은 src/prepare_f9_v2.py 의 parse_dates · parse_range 를 그대로 쓴다)
  · create_event  문장에 날짜 표현이 딱 1개일 때만 → start 를 그 날짜로, end 는 같은 간격만큼 옮긴다(시각은 모델 답 그대로)
  · list_events   날짜 표현이 있으면 → from = to = 첫 날짜 (날짜를 옮기는 수정은 첫 날짜가 지금 일정이 있는 날)
                  날짜 표현이 없고 '이번 주 · 다음 주 · 다다음 주 · 주말 · 이번 달 · 다음 달' 이 있으면 → 그 범위
  · update_event  날짜 표현이 2개(…을 …로 옮겨)일 때만 → patch.start 를 둘째 날짜로, patch.end 는 같은 간격만큼
  · 손대지 않는 것: 반복 일정('매주' · '…마다'), 날짜 표현이 2개 이상인 등록(…부터 …까지), 날짜 표현이 없는 문장,
                  그 밖의 도구, 글로 답한 턴
real_test 파일에는 자동으로 --hide 가 걸린다(가이드 3-7 — 최종 시험은 점수만 본다).
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare_f9_v2 as v2  # noqa: E402
from common import load_jsonl, parse_tool_call  # noqa: E402
from eval import score_turn  # noqa: E402  — 채점 함수만 쓴다(모델은 eval.main 안에서만 불러온다)

D1 = dt.timedelta(days=1)
WDN = "월화수목금토일"
_NOW = re.compile(r"오늘은 (\d{4}-\d{2}-\d{2})\([월화수목금토일]\) (\d{2}:\d{2}) 입니다")
_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})(.*)$", re.S)


def today_of(messages) -> dt.date:
    for m in messages:
        if m["role"] == "system":
            hit = _NOW.search(m["content"])
            if hit:
                return dt.date.fromisoformat(hit.group(1))
    raise ValueError("기준시각 줄이 없다")


def _split(s):
    """'2026-10-31T10:00' → (date 또는 None, 날짜 문자열, 뒤에 붙은 시각 문자열). 날짜 꼴이 아니면 None."""
    if not isinstance(s, str):
        return None
    m = _DATE.match(s)
    if not m:
        return None
    try:
        d = dt.date.fromisoformat(m.group(1))
    except ValueError:
        d = None                                   # 달력에 없는 날짜 (예: 11-31)
    return d, m.group(1), m.group(2)


def _move_pair(obj, k1, k2, d1, d2):
    """obj[k1] 의 날짜를 d1 로. obj[k2] 는 d2 가 있으면 d2, 없으면 k1 과 같은 간격만큼 옮긴다."""
    a, b = _split(obj.get(k1)), _split(obj.get(k2))
    if a is None:
        return False
    new1 = d1.isoformat() + a[2]
    if b is None:
        obj[k1] = new1
        return True
    if d2 is not None:
        new2 = d2.isoformat() + b[2]
    elif a[0] is not None and b[0] is not None:
        new2 = (b[0] + (d1 - a[0])).isoformat() + b[2]
    elif a[1] == b[1]:                              # 둘 다 같은 (없는) 날짜 문자열 → 같은 날로
        new2 = d1.isoformat() + b[2]
    else:
        return False
    obj[k1], obj[k2] = new1, new2
    return True


def apply_rule(user: str, today: dt.date, call: dict) -> tuple[dict, str]:
    """(고친 호출, 한 일). 한 일: 'none'(규칙 대상 아님) · 'same'(규칙 답 = 모델 답) · 'changed' · 'skip:<이유>'."""
    name, args = call.get("name"), call.get("arguments")
    if name not in ("create_event", "list_events", "update_event") or not isinstance(args, dict):
        return call, "none"
    if "매주" in user or "마다" in user:
        return call, "skip:반복"
    new = json.loads(json.dumps(args))
    try:
        ds = v2.parse_dates(user, today)
    except ValueError:
        return call, "skip:읽기 실패"
    ok = False
    if name == "create_event":
        if len(ds) == 1:
            ok = _move_pair(new, "start", "end", ds[0], None)
        else:
            return call, ("none" if not ds else "skip:날짜 2개 이상")
    elif name == "list_events":
        if ds:
            ok = _move_pair(new, "from", "to", ds[0], ds[0])
        else:
            r = v2.parse_range(user, today)
            if r is None:
                return call, "none"
            ok = _move_pair(new, "from", "to", r[0], r[1])
    elif name == "update_event":
        patch = new.get("patch")
        if len(ds) != 2 or not isinstance(patch, dict) or "start" not in patch:
            return call, "none"
        ok = _move_pair(patch, "start", "end", ds[1], None)
    if not ok:
        return call, "skip:모델 답 꼴"
    return {"name": name, "arguments": new}, ("same" if new == args else "changed")


def turns_of(rows):
    """(턴 id, 오늘, 사용자 문장, 정답 호출 또는 None, 틀 이름, 출처) — eval.py 와 같은 순서."""
    for r in rows:
        msgs, t = r["messages"], 0
        today = today_of(msgs)
        meta = r.get("meta") or {}
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            user = next((p["content"] for p in reversed(msgs[:i]) if p["role"] == "user"), "")
            call = None
            if m.get("tool_calls"):
                fn = m["tool_calls"][0]["function"]
                a = fn["arguments"]
                call = {"name": fn["name"], "arguments": json.loads(a) if isinstance(a, str) else a}
            yield f"{r.get('id')}#t{t}", today, user, call, meta.get("template"), meta.get("source")
            t += 1


def rescored(s: dict, user: str, today: dt.date) -> tuple[bool, str, str]:
    """평가 결과의 한 턴에 규칙을 적용한다 → (규칙 뒤 정답 여부, 한 일, 규칙 뒤 답 글)."""
    before = bool(s["exact_ok"] or s["text_ok"])
    pred = parse_tool_call(s["pred_text"])
    if pred is None:
        return before, "none", s["pred_text"]
    new, what = apply_rule(user, today, pred)
    if what != "changed":
        return before, what, s["pred_text"]
    text = "<tool_call>" + json.dumps(new, ensure_ascii=False) + "</tool_call>"
    exp = s["expected"] if isinstance(s["expected"], dict) else None
    sc = score_turn(exp, text, refusal=(s["turn_type"] == "refusal"))
    return bool(sc["exact_ok"] or sc["text_ok"]), what, text


def is_test_file(path) -> bool:
    return Path(path).name.startswith("real_test")


def cmd_gold(path: str, label: str, hide: bool) -> None:
    rows = load_jsonl(path)
    st = collections.Counter()
    bad = collections.defaultdict(list)
    for tid, today, user, call, tpl, src in turns_of(rows):
        if call is None:
            continue
        new, what = apply_rule(user, today, call)
        st[(call["name"], what)] += 1
        if what == "changed":
            bad[(src, (tpl or "?").split("|")[0])].append((tid, today, user, call["arguments"], new["arguments"]))
    print(f"== 정답에 규칙 적용 — {label} · {len(rows)}행")
    for n in sorted({k[0] for k in st}):
        tot = sum(v for k, v in st.items() if k[0] == n)
        print(f"  {n:16s} 턴 {tot:5d} · " + " · ".join(f"{k[1]} {v}" for k, v in sorted(st.items()) if k[0] == n))
    fired = sum(v for k, v in st.items() if k[1] in ("same", "changed"))
    chg = sum(v for k, v in st.items() if k[1] == "changed")
    dated = sum(v for k, v in st.items() if k[0] in ("create_event", "list_events", "update_event"))
    print(f"  날짜가 들어가는 도구 턴 {dated} · 규칙이 날짜를 정한 턴 {fired} · 그중 정답과 같음 {fired - chg} · 정답을 바꿈 {chg}")
    for (src, tpl), items in sorted(bad.items(), key=lambda x: -len(x[1])):
        print(f"  [정답을 바꿈] {src} · {tpl} × {len(items)}")
        if hide:
            continue
        for tid, today, user, a, b in items[:2]:
            print(f"      {tid} 오늘 {today}({WDN[today.weekday()]}) | {user}")
            print(f"        정답 {json.dumps(a, ensure_ascii=False)[:170]}")
            print(f"        규칙 {json.dumps(b, ensure_ascii=False)[:170]}")


def cmd_apply(res_path: str, rows_path: str, label: str, hide: bool) -> None:
    res = json.loads(Path(res_path).read_text(encoding="utf-8"))["results"]
    meta = {tid: (today, user) for tid, today, user, _c, _t, _s in turns_of(load_jsonl(rows_path))}
    assert [s["id"] for s in res] == list(meta), "결과와 시험 파일의 턴이 다르다"
    st = collections.Counter()
    lists = collections.defaultdict(list)
    for s in res:
        today, user = meta[s["id"]]
        assert user == s["user"]
        exp = s["expected"] if isinstance(s["expected"], dict) else None
        before = score_turn(exp, s["pred_text"], refusal=(s["turn_type"] == "refusal"))
        b_ok = bool(before["exact_ok"] or before["text_ok"])
        assert b_ok == bool(s["exact_ok"] or s["text_ok"]), f"다시 채점한 값이 저장된 값과 다르다: {s['id']}"
        a_ok, what, new_text = rescored(s, user, today)
        st["n"] += 1
        st["before"] += b_ok
        st["after"] += a_ok
        st["rule:" + what.split(":")[0]] += 1
        if what == "changed":
            key = ("고쳐짐" if a_ok else "바꿨지만 여전히 틀림") if not b_ok else ("그대로 맞음" if a_ok else "맞던 것을 틀리게 함")
            st[key] += 1
            lists[key].append((s, new_text))
        elif not b_ok:
            st["규칙이 안 건드린 오답"] += 1
            lists["규칙이 안 건드린 오답"].append((s, what))
    n = st["n"]
    print(f"== 모델 답에 규칙 적용 — {label} · 턴 {n}")
    print(f"  맞은 턴: 규칙 전 {st['before']}/{n} ({100*st['before']/n:.1f}%) → 규칙 후 {st['after']}/{n} ({100*st['after']/n:.1f}%)")
    print(f"  규칙이 날짜를 정한 턴 {st['rule:same'] + st['rule:changed']} (모델 답과 같음 {st['rule:same']} · 바꿈 {st['rule:changed']})"
          f" · 대상 아님 {st['rule:none']} · 건너뜀 {st['rule:skip']}")
    print(f"  바꾼 {st['rule:changed']}턴: 고쳐짐 {st['고쳐짐']} · 맞던 것을 틀리게 함 {st['맞던 것을 틀리게 함']}"
          f" · 바꿨지만 여전히 틀림 {st['바꿨지만 여전히 틀림']} · 그대로 맞음 {st['그대로 맞음']}")
    print(f"  규칙이 안 건드린 오답 {st['규칙이 안 건드린 오답']}")
    if hide:
        return

    def one(x):
        return re.sub(r"\s+", " ", json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x))[:160]

    for key in ("맞던 것을 틀리게 함", "바꿨지만 여전히 틀림"):
        for s, new_text in lists[key][:8]:
            print(f"  [{key}] {s['id']} `{s.get('template')}` | {s['user']}")
            print(f"      정답 {one(s['expected'])}")
            print(f"      모델 {one(s['pred_text'])}")
            print(f"      규칙 {one(new_text)}")
    c = collections.Counter((s.get("template") or "?").split("|")[0] + " · " + what for s, what in lists["규칙이 안 건드린 오답"])
    for k, v in c.most_common():
        print(f"  [규칙이 안 건드린 오답] {k} × {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gold", help="정답에 규칙을 적용한다 — 정답을 바꾸면 규칙이 틀린 것")
    g.add_argument("rows")
    a = sub.add_parser("apply", help="모델 답(eval.py 의 .json)에 규칙을 적용해 다시 채점한다")
    a.add_argument("result")
    a.add_argument("rows")
    for p in (g, a):
        p.add_argument("--label", default="")
        p.add_argument("--hide", action="store_true", help="문장·답을 출력하지 않고 개수만")
    args = ap.parse_args()
    hide = args.hide or is_test_file(args.rows)
    if args.mode == "gold":
        cmd_gold(args.rows, args.label or args.rows, hide)
    else:
        cmd_apply(args.result, args.rows, args.label or args.result, hide)
    return 0


if __name__ == "__main__":
    sys.exit(main())
