#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F9 평가 — 같은 eval.jsonl 을 어댑터 없이/있게 채점해 비교표를 만든다 (개발계획서 7.2 3자 비교).

  python src/eval.py --model_id <HF id> --data data/f9/eval.jsonl --out outputs/run1/eval_base.md --label "학습 전 Instruct"
  python src/eval.py --model_id <HF id> --adapter outputs/run1/adapter --out outputs/run1/eval_adapter.md --label "학습 후"

지표
  · 도구 선택 정확도   첫 assistant 턴의 도구 이름이 맞는가
  · 인자 일치          정규화(날짜 ISO·공백·정렬) 후 인자 전체가 같은가
  · 필드 정확도        기대 인자 필드별 일치율
  · 거절 정확도        도구를 부르면 안 되는 질문에서 도구를 안 불렀는가
  · 턴 정확도          위 둘을 합친 전체 정답률
  · 지연               샘플당 생성 시간
출력: <out>.md (요약표·도구별·실패 10건) · <out>.json · <out 디렉터리>/summary_row_<label>.md (비교표 한 줄)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (detect_render_mode, ensure_dir, expected_call, lib_versions, load_jsonl, load_tools,  # noqa: E402
                    normalize_value, parse_tool_call, render_chat, split_prompt_and_target, strip_think,
                    TOOL_CALL_CLOSE)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--adapter", default="", help="LoRA 어댑터 폴더. 비우면 베이스만")
    ap.add_argument("--data", default=str(ROOT / "data" / "f9" / "eval.jsonl"))
    ap.add_argument("--tools", default=str(ROOT / "tools" / "f9_tools.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "run1" / "eval.md"))
    ap.add_argument("--label", default="", help="비교표 행 이름 (예: 학습 전 Instruct / 학습 후)")
    ap.add_argument("--limit", type=int, default=0, help="0=전부")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--load_4bit", type=int, default=-1, help="-1=CUDA 면 1, 아니면 0")
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


class _StopOnText:
    """</tool_call> 이 나오면 생성을 멈춘다 — 도구 호출은 짧아서 시간을 많이 아낀다."""

    def __init__(self, torch, tok, prompt_len: int, stops: list[str]):
        from transformers import StoppingCriteria
        self.torch, self.tok, self.prompt_len, self.stops = torch, tok, prompt_len, stops

        class _C(StoppingCriteria):
            def __call__(_self, input_ids, scores, **kwargs):
                tail = tok.decode(input_ids[0, prompt_len:][-24:], skip_special_tokens=False)
                done = any(s in tail for s in stops)
                return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)
        self.criteria = _C()


def generate(torch, tok, model, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    from transformers import StoppingCriteriaList
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    stop = _StopOnText(torch, tok, enc.input_ids.shape[1], [TOOL_CALL_CLOSE])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([stop.criteria]))
    text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return text, time.time() - t0


def score_row(expected: dict | None, pred_text: str) -> dict:
    pred = parse_tool_call(pred_text)
    r = {"expected_tool": expected["name"] if expected else None, "pred_tool": pred["name"] if pred else None,
         "name_ok": False, "exact_ok": False, "field_total": 0, "field_ok": 0, "refusal_ok": None}
    if expected is None:
        r["refusal_ok"] = pred is None and bool(strip_think(pred_text).strip())
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


def main() -> None:
    args = parse_args()
    import torch
    tools, kinds = load_tools(args.tools)
    rows = load_jsonl(args.data, limit=args.limit or None)
    tok, model, cuda = load_model(args, torch)
    mode = detect_render_mode(tok, tools)
    label = args.label or ("학습 후" if args.adapter else "학습 전")
    print(f"평가 {len(rows)}건 · {args.model_id} {'+ ' + args.adapter if args.adapter else ''} · 모드 {mode} · "
          f"{'cuda' if cuda else 'cpu'} 4bit={bool(args.load_4bit)} {args.compute_dtype}")

    results, lat = [], []
    for i, row in enumerate(rows, 1):
        prompt_msgs, target = split_prompt_and_target(row["messages"])
        expected = expected_call(target)
        prompt = render_chat(tok, prompt_msgs, row.get("tools") or tools, add_generation_prompt=True)
        text, sec = generate(torch, tok, model, prompt, args.max_new_tokens)
        lat.append(sec)
        s = score_row(expected, text)
        s.update({"id": row.get("id"), "user": next((m["content"] for m in prompt_msgs if m["role"] == "user"), ""),
                  "expected": expected or (target or {}).get("content"), "pred_text": text.strip(), "sec": round(sec, 2),
                  "template": (row.get("meta") or {}).get("template")})
        results.append(s)
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  마지막 {sec:.1f}s")

    tool_rows = [r for r in results if r["expected_tool"]]
    ref_rows = [r for r in results if r["expected_tool"] is None]
    pct = lambda a, b: (100.0 * a / b) if b else float("nan")  # noqa: E731
    summary = {
        "label": label, "model_id": args.model_id, "adapter": args.adapter, "render_mode": mode, "n": len(results),
        "n_tool": len(tool_rows), "n_refusal": len(ref_rows),
        "tool_name_acc": pct(sum(r["name_ok"] for r in tool_rows), len(tool_rows)),
        "arg_exact_acc": pct(sum(r["exact_ok"] for r in tool_rows), len(tool_rows)),
        "field_acc": pct(sum(r["field_ok"] for r in tool_rows), sum(r["field_total"] for r in tool_rows)),
        "refusal_acc": pct(sum(bool(r["refusal_ok"]) for r in ref_rows), len(ref_rows)),
        "turn_acc": pct(sum(r["exact_ok"] for r in tool_rows) + sum(bool(r["refusal_ok"]) for r in ref_rows), len(results)),
        "latency_mean_s": sum(lat) / len(lat) if lat else 0.0,
        "latency_p95_s": sorted(lat)[int(len(lat) * 0.95) - 1] if len(lat) >= 2 else (lat[0] if lat else 0.0),
        "versions": lib_versions(), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    per_tool = {}
    for r in tool_rows:
        d = per_tool.setdefault(r["expected_tool"], {"n": 0, "name_ok": 0, "exact_ok": 0, "f_ok": 0, "f_tot": 0})
        d["n"] += 1; d["name_ok"] += r["name_ok"]; d["exact_ok"] += r["exact_ok"]
        d["f_ok"] += r["field_ok"]; d["f_tot"] += r["field_total"]

    def f(x):
        return "—" if x != x else f"{x:.1f}%"
    row_md = (f"| {label} | {summary['n']} | {f(summary['tool_name_acc'])} | {f(summary['arg_exact_acc'])} | "
              f"{f(summary['field_acc'])} | {f(summary['refusal_acc'])} | {f(summary['turn_acc'])} | "
              f"{summary['latency_mean_s']:.2f} |")
    header = ["| 구분 | n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 턴 정확도 | 평균 지연(s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines = [f"# F9 평가 — {label}", "",
             f"> 모델 `{args.model_id}`" + (f" + 어댑터 `{args.adapter}`" if args.adapter else "") +
             f" · 템플릿 모드 {mode} · {summary['finished_at']} · 데이터 `{args.data}` ({summary['n']}건)", "",
             *header, row_md, "",
             "## 도구별", "", "| 도구 | 종류 | n | 도구 선택 | 인자 일치 | 필드 정확도 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, d in sorted(per_tool.items()):
        lines.append(f"| {name} | {kinds.get(name, '?')} | {d['n']} | {f(pct(d['name_ok'], d['n']))} | "
                     f"{f(pct(d['exact_ok'], d['n']))} | {f(pct(d['f_ok'], d['f_tot']))} |")
    if ref_rows:
        lines.append(f"| (도구 없음·거절) | — | {len(ref_rows)} | — | — | 거절 {f(summary['refusal_acc'])} |")
    fails = [r for r in results if not (r["exact_ok"] or r["refusal_ok"])][:10]
    lines += ["", f"## 실패 사례 (최대 10건 / 전체 실패 {sum(1 for r in results if not (r['exact_ok'] or r['refusal_ok']))}건)", ""]
    for r in fails:
        exp = json.dumps(r["expected"], ensure_ascii=False) if isinstance(r["expected"], dict) else str(r["expected"])
        pred_one = re.sub(r"\s+", " ", r["pred_text"])[:200]
        lines += [f"- **{r['id']}** `{r['template']}` — 사용자: {r['user']}",
                  f"  - 기대: `{exp[:200]}`",
                  f"  - 예측: `{pred_one}`"]
    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "per_tool": per_tool, "results": results},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    safe = re.sub(r"[^\w가-힣]+", "_", label)
    (out.parent / f"summary_row_{safe}.md").write_text(row_md + "\n", encoding="utf-8")
    print("\n" + "\n".join(header + [row_md]))
    print(f"\n저장 → {out} · {out.with_suffix('.json')} · summary_row_{safe}.md")


if __name__ == "__main__":
    main()
