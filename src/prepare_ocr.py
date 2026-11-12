#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SynthDoG-ko parquet 점검 — 학습 전에 스키마·길이·이미지 크기를 보고 MAX_LEN·IMAGE_H/W 가 맞는지 확인한다.

  python src/prepare_ocr.py            # 또는  bash train_ocr.sh --prepare
출력: 스키마·행 수 · train 표본(기본 2,000행)과 validation 전체의 text_sequence 문자 길이·토큰 길이
      (토크나이저는 MODEL_ID, <s_synthdog> + 본문 + </s>) p50/p95/p99/max · MAX_LEN 초과 비율
      · 이미지 크기 분포(표본 50장) · 표본 이미지 3장 → $OCR_OUT_DIR/samples/*.png
결과: $OCR_OUT_DIR/data_stats.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_common import TASK_TOKEN, decode_image, gt_text, load_cfg, load_split  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    c = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=c.get("MODEL_ID", ""), help="토크나이저를 읽을 HF id 또는 경로")
    ap.add_argument("--data_dir", default=c.get("OCR_DATA_DIR", ""))
    ap.add_argument("--out_dir", default=c.get("OCR_OUT_DIR", "outputs/donut-ocr"))
    ap.add_argument("--max_len", type=int, default=int(c.get("MAX_LEN", 768)))
    ap.add_argument("--train_sample", type=int, default=2000, help="train 앞 몇 행으로 길이를 볼지")
    ap.add_argument("--image_sample", type=int, default=50)
    return ap.parse_args(argv)


def pct(vals: list[int]) -> dict:
    s = sorted(vals)
    if not s:
        return {}
    q = lambda p: s[min(len(s) - 1, int(len(s) * p))]  # noqa: E731
    return {"n": len(s), "p50": q(0.50), "p95": q(0.95), "p99": q(0.99), "max": s[-1]}


def main(argv=None) -> None:
    args = parse_args(argv)
    from transformers import AutoTokenizer
    out = Path(args.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    lines = ["# SynthDoG-ko 데이터 점검", "", f"- data_dir: `{args.data_dir}`", f"- tokenizer: `{args.model}` · MAX_LEN {args.max_len}", ""]

    tr_full = load_split(args.data_dir, "train", 0)
    va = load_split(args.data_dir, "validation", 0)
    lines += ["## 스키마·행 수", "", f"- train {len(tr_full):,}행 · validation {len(va):,}행",
              f"- features: `{tr_full.features}`", ""]
    print(f"train {len(tr_full):,}행 · validation {len(va):,}행\nfeatures {tr_full.features}")
    tr = tr_full.select(range(min(args.train_sample, len(tr_full))))

    lines += ["## 길이 (토큰 = task 토큰 + 본문 + </s>)", "", "| 분할 | 행 | 문자 p50/p95/p99/max | 토큰 p50/p95/p99/max | MAX_LEN 초과 |",
              "| --- | --- | --- | --- | --- |"]
    for name, d in (("train 표본", tr), ("validation", va)):
        texts = [gt_text(g) for g in d["ground_truth"]]
        chars = [len(t) for t in texts]
        toks = [len(tok(TASK_TOKEN + t + tok.eos_token, add_special_tokens=False).input_ids) for t in texts]
        c, k = pct(chars), pct(toks)
        over = sum(1 for n in toks if n > args.max_len) / max(1, len(toks))
        row = (f"| {name} | {len(texts):,} | {c['p50']}/{c['p95']}/{c['p99']}/{c['max']} | "
               f"{k['p50']}/{k['p95']}/{k['p99']}/{k['max']} | {over * 100:.2f}% |")
        lines.append(row)
        print(row)
        if name == "train 표본":
            ex = texts[0]

    sizes = []
    for i in range(min(args.image_sample, len(tr))):
        img = decode_image(tr[i])
        sizes.append(img.size)
        if i < 3:
            img.save(out / "samples" / f"train_{i}.png")
    ws, hs = sorted(w for w, _ in sizes), sorted(h for _, h in sizes)
    portrait = sum(1 for w, h in sizes if h >= w)
    size_line = (f"- 너비 min/p50/max {ws[0]}/{ws[len(ws) // 2]}/{ws[-1]} · "
                 f"높이 min/p50/max {hs[0]}/{hs[len(hs) // 2]}/{hs[-1]} · 세로형 {portrait}장")
    lines += ["", f"## 이미지 크기 (train 표본 {len(sizes)}장)", "", size_line,
              f"- 표본 3장: `{out / 'samples'}`", "", "## 첫 행 text_sequence", "", f"> {ex[:300]}", ""]
    print(size_line)
    p = out / "data_stats.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"→ {p}")


if __name__ == "__main__":
    main()
