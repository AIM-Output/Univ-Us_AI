#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F9 평가 — 같은 eval.jsonl 을 어댑터 없이/있게 채점해 비교표를 만든다 (개발계획서 7.2 3자 비교).

  python src/eval.py --model_id <HF id> --data data/f9/eval.jsonl --out outputs/run1/eval_base.md --label "학습 전 Instruct"
  python src/eval.py --model_id <HF id> --adapter outputs/run1/adapter --out outputs/run1/eval_adapter.md --label "학습 후"

**모든 assistant 턴**을 채점한다(teacher forcing — 그 턴 앞까지의 정답 대화를 그대로 프롬프트로 준다).
두 단계 행(도구 호출 → 결과 → 답변)은 턴 2개, update/delete 행은 list_events 턴 + update/delete 턴.

지표 (턴 단위)
  · 도구 선택 정확도   도구 턴에서 도구 이름이 맞는가
  · 인자 일치          정규화(날짜 ISO·공백·정렬) 후 인자 전체가 같은가
  · 필드 정확도        기대 인자 필드별 일치율
  · 거절 정확도        첫 턴이 텍스트인 행(거절·범위 밖)에서 도구를 안 부르고 거절 표현('…수 없습니다' 등, common.REFUSAL_MARKERS)이 있는가
  · 텍스트 턴 정확도   텍스트 턴 전체(거절 + 도구 결과 뒤 답변)에서 도구를 안 부르고 빈 답이 아닌가
  · 턴 정확도          (인자까지 맞은 도구 턴 + 맞은 텍스트 턴) / 전체 턴
  · 지연               턴당 생성 시간
출력: <out>.md (요약표·턴 종류·도구별·실패 10건) · <out>.json · <out 디렉터리>/summary_row_<label>.md (비교표 한 줄)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TOOL_CALL_CLOSE, detect_render_mode, ensure_dir, expected_call, is_refusal_text,  # noqa: E402
                    lib_versions, load_jsonl, load_tools, normalize_value, parse_tool_call, render_chat, strip_think)

ROOT = Path(__file__).resolve().parents[1]
HEADER = ["| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
STOP_STRINGS = [TOOL_CALL_CLOSE, "<|im_end|>"]     # Base 모델(eos=<|endoftext|>)이 템플릿의 <|im_end|> 뒤로 계속 쓰지 않게


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--adapter", default="", help="LoRA 어댑터 폴더. 비우면 베이스만")
    ap.add_argument("--data", default=str(ROOT / "data" / "f9" / "eval.jsonl"))
    ap.add_argument("--tools", default=str(ROOT / "tools" / "f9_tools.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "run1" / "eval.md"))
    ap.add_argument("--label", default="", help="비교표 행 이름 (예: 학습 전 Instruct / 학습 후)")
    ap.add_argument("--limit", type=int, default=0, help="행 수 제한. 0=전부")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--load_4bit", type=int, default=-1, help="-1=CUDA 면 1, 아니면 0. 학습과 같은 값을 줄 것")
    ap.add_argument("--compute_dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    return ap.parse_args()


def load_model(args, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cuda = torch.cuda.is_available()
    if args.load_4bit == -1:
        args.load_4bit = 1 if cuda else 0
    if args.compute_dtype == "auto":
        args.compute_dtype = "fp32" if not cuda else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    if not cuda:
        args.load_4bit, args.compute_dtype = 0, "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.compute_dtype]
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype) if args.load_4bit else None
    kw = dict(quantization_config=quant, trust_remote_code=True, device_map="auto" if cuda else None)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return tok, model, cuda


def make_stopper(torch, tok, prompt_len: int, stops: list[str]):
    """</tool_call> · <|im_end|> · eos 문자열이 나오면 멈춘다 — 도구 호출은 짧아서 시간을 많이 아낀다."""
    from transformers import StoppingCriteria
    stops = [s for s in dict.fromkeys(stops + ([tok.eos_token] if tok.eos_token else [])) if s]

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            tail = tok.decode(input_ids[0, prompt_len:][-24:], skip_special_tokens=False)
            done = any(s in tail for s in stops)
            return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)
    return _Stop()


def generate(torch, tok, model, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    from transformers import StoppingCriteriaList
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    stop = make_stopper(torch, tok, enc.input_ids.shape[1], STOP_STRINGS)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([stop]))
    text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    for s in STOP_STRINGS:          # skip_special_tokens 가 못 지우는 문자열 토큰(폴백 모드) 정리
        if s != TOOL_CALL_CLOSE:
            text = text.replace(s, "")
    return text, time.time() - t0


def score_turn(expected: dict | None, pred_text: str, refusal: bool = False) -> dict:
    """도구 턴: 이름·인자 채점. 텍스트 턴: 도구를 안 부르고 빈 답이 아니면 맞음.
    거절 턴(refusal=True): 거기에 더해 거절 표현(is_refusal_text)이 있어야 맞음 — "도와드리겠다"고 답하면 틀림."""
    pred = parse_tool_call(pred_text)
    r = {"expected_tool": expected["name"] if expected else None, "pred_tool": pred["name"] if pred else None,
         "name_ok": False, "exact_ok": False, "field_total": 0, "field_ok": 0, "text_ok": None}
    if expected is None:
        text = strip_think(pred_text).strip()
        r["text_ok"] = pred is None and bool(text) and (is_refusal_text(text) if refusal else True)
        return r
    if pred is None:
        r["field_total"] = len(expected["arguments"])
        return r
    r["name_ok"] = pred["name"] == expected["name"]
    ea, pa = normalize_value(expected["arguments"]), normalize_value(pred["arguments"])
    r["field_total"] = len(ea)
    r["field_ok"] = sum(1 for k, v in ea.items() if k in pa and pa[k] == v)
    r["exact_ok"] = r["name_ok"] and ea == pa
    return r


def _pct(a: float, b: float) -> float:
    return (100.0 * a / b) if b else float("nan")


def _f(x: float) -> str:
    return "—" if x != x else f"{x:.1f}%"


def main() -> None:
    args = parse_args()
    import torch
    tools, kinds = load_tools(args.tools)
    rows = load_jsonl(args.data, limit=args.limit or None)
    tok, model, cuda = load_model(args, torch)
    mode = detect_render_mode(tok, tools)
    label = args.label or ("학습 후" if args.adapter else "학습 전")
    n_turns_total = sum(1 for r in rows for m in r["messages"] if m.get("role") == "assistant")
    print(f"평가 {len(rows)}행 · assistant 턴 {n_turns_total}개 · {args.model_id} {'+ ' + args.adapter if args.adapter else ''}"
          f" · 모드 {mode} · {'cuda' if cuda else 'cpu'} 4bit={bool(args.load_4bit)} {args.compute_dtype}")

    results, lat, done = [], [], 0
    sec = 0.0
    for ri, row in enumerate(rows, 1):
        msgs, tl = row["messages"], row.get("tools") or tools
        turn = 0
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            expected = expected_call(m)
            prompt = render_chat(tok, msgs[:i], tl, add_generation_prompt=True)
            text, sec = generate(torch, tok, model, prompt, args.max_new_tokens)
            lat.append(sec)
            prev_tool_turns = sum(1 for p in msgs[:i] if p.get("role") == "assistant" and p.get("tool_calls"))
            s = score_turn(expected, text, refusal=(expected is None and prev_tool_turns == 0))
            s.update({"id": f"{row.get('id')}#t{turn}", "turn": turn,
                      "turn_type": "tool" if expected else ("refusal" if prev_tool_turns == 0 else "answer"),
                      "user": next((p["content"] for p in reversed(msgs[:i]) if p["role"] == "user"), ""),
                      "expected": expected or (m.get("content") or ""), "pred_text": text.strip(), "sec": round(sec, 2),
                      "template": (row.get("meta") or {}).get("template")})
            results.append(s)
            turn += 1
            done += 1
        if ri % 10 == 0 or ri == len(rows):
            print(f"  행 {ri}/{len(rows)} · 턴 {done}/{n_turns_total} · 마지막 {sec:.1f}s")

    tool_t = [r for r in results if r["turn_type"] == "tool"]
    text_t = [r for r in results if r["turn_type"] != "tool"]
    ref_t = [r for r in results if r["turn_type"] == "refusal"]
    ans_t = [r for r in results if r["turn_type"] == "answer"]
    summary = {
        "label": label, "model_id": args.model_id, "adapter": args.adapter, "render_mode": mode,
        "n_rows": len(rows), "n_turns": len(results), "n_tool_turns": len(tool_t),
        "n_refusal_turns": len(ref_t), "n_answer_turns": len(ans_t),
        "tool_name_acc": _pct(sum(r["name_ok"] for r in tool_t), len(tool_t)),
        "arg_exact_acc": _pct(sum(r["exact_ok"] for r in tool_t), len(tool_t)),
        "field_acc": _pct(sum(r["field_ok"] for r in tool_t), sum(r["field_total"] for r in tool_t)),
        "refusal_acc": _pct(sum(bool(r["text_ok"]) for r in ref_t), len(ref_t)),
        "text_turn_acc": _pct(sum(bool(r["text_ok"]) for r in text_t), len(text_t)),
        "turn_acc": _pct(sum(r["exact_ok"] for r in tool_t) + sum(bool(r["text_ok"]) for r in text_t), len(results)),
        "latency_mean_s": sum(lat) / len(lat) if lat else 0.0,
        "latency_p95_s": sorted(lat)[max(0, int(len(lat) * 0.95) - 1)] if lat else 0.0,
        "versions": lib_versions(), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    per_tool: dict[str, dict] = {}
    for r in tool_t:
        d = per_tool.setdefault(r["expected_tool"], {"n": 0, "name_ok": 0, "exact_ok": 0, "f_ok": 0, "f_tot": 0})
        d["n"] += 1; d["name_ok"] += r["name_ok"]; d["exact_ok"] += r["exact_ok"]
        d["f_ok"] += r["field_ok"]; d["f_tot"] += r["field_total"]

    row_md = (f"| {label} | {summary['n_turns']} | {_f(summary['tool_name_acc'])} | {_f(summary['arg_exact_acc'])} | "
              f"{_f(summary['field_acc'])} | {_f(summary['refusal_acc'])} | {_f(summary['text_turn_acc'])} | "
              f"{_f(summary['turn_acc'])} | {summary['latency_mean_s']:.2f} |")
    lines = [f"# F9 평가 — {label}", "",
             f"> 모델 `{args.model_id}`" + (f" + 어댑터 `{args.adapter}`" if args.adapter else "") +
             f" · 템플릿 모드 {mode} · {summary['finished_at']} · 데이터 `{args.data}` ({len(rows)}행 · {len(results)}턴)", "",
             *HEADER, row_md, "",
             "## 턴 종류", "", "| 종류 | 턴 수 | 정확도 |", "| --- | ---: | ---: |",
             f"| 도구 호출 | {len(tool_t)} | 인자까지 {_f(summary['arg_exact_acc'])} |",
             f"| 거절·범위 밖 (첫 턴 텍스트) | {len(ref_t)} | {_f(summary['refusal_acc'])} |",
             f"| 도구 결과 뒤 답변 | {len(ans_t)} | {_f(_pct(sum(bool(r['text_ok']) for r in ans_t), len(ans_t)))} |", "",
             "## 도구별", "", "| 도구 | 종류 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, d in sorted(per_tool.items()):
        lines.append(f"| {name} | {kinds.get(name, '?')} | {d['n']} | {_f(_pct(d['name_ok'], d['n']))} | "
                     f"{_f(_pct(d['exact_ok'], d['n']))} | {_f(_pct(d['f_ok'], d['f_tot']))} |")
    missing = sorted(set(kinds) - set(per_tool))
    if missing:
        lines.append(f"| (평가셋에 없음) {', '.join(missing)} | — | — | — | — | — |")
    fails = [r for r in results if not (r["exact_ok"] or r["text_ok"])]
    lines += ["", f"## 실패 사례 (최대 10건 / 전체 실패 {len(fails)}턴)", ""]
    for r in fails[:10]:
        exp = json.dumps(r["expected"], ensure_ascii=False) if isinstance(r["expected"], dict) else str(r["expected"])
        pred_one = re.sub(r"\s+", " ", r["pred_text"])[:200]
        lines += [f"- **{r['id']}** `{r['template']}` [{r['turn_type']}] — 사용자: {r['user']}",
                  f"  - 기대: `{exp[:200]}`",
                  f"  - 예측: `{pred_one}`"]
    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "per_tool": per_tool, "results": results},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    safe = re.sub(r"[^\w가-힣]+", "_", label)
    (out.parent / f"summary_row_{safe}.md").write_text(row_md + "\n", encoding="utf-8")
    print("\n" + "\n".join(HEADER + [row_md]))
    print(f"\n저장 → {out} · {out.with_suffix('.json')} · summary_row_{safe}.md")


if __name__ == "__main__":
    main()
