#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Donut OCR 채점 — validation 앞 rows 행을 읽혀 CER · 1-NED · 정확 일치 · 문자 가중 CER · 지연을 잰다.

  python src/eval_ocr.py --model <MODEL_ID 또는 경로> --rows 500 --tag base|finetuned
  python src/eval_ocr.py --compare          # eval_base.json + eval_finetuned.json → compare.md
  bash eval_ocr.sh                          # base → finetuned → compare 전부

생성: decoder_input_ids=[[<s_synthdog>]] · greedy(num_beams=1) · max_length=MAX_LEN · <unk> 금지 · bf16(GPU)
채점: 특수 토큰 제거 → normalize(양끝 공백 제거·연속 공백 1개) 뒤
  CER = 편집거리/max(1,len(ref)) 의 평균 · 1-NED = 1 - 편집거리/max(len(ref),len(hyp)) 의 평균
  정확 일치율 · 문자 가중 CER = Σ편집거리/Σlen(ref) · 평균 지연 = 배치 시간/배치 크기 (s/장)
산출물: $OCR_OUT_DIR/eval_<tag>.json (행마다 ref·hyp·cer·sec) · eval_<tag>.md (요약 + 나쁜 10건 + 무작위 5건)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_common import (clean_text, decode_image, gt_text, lib_versions, load_cfg, load_model,  # noqa: E402
                        load_split, pixel_values, score_pair, summarize)

METRICS = [("cer_mean", "CER 평균 ↓"), ("ned_sim_mean", "1-NED 평균 ↑"), ("exact_rate", "정확 일치율 ↑"),
           ("cer_weighted", "문자 가중 CER ↓"), ("sec_mean", "평균 지연 s/장 ↓")]


def parse_args(argv=None) -> argparse.Namespace:
    c = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=c.get("MODEL_ID", ""), help="HF id(학습 전) 또는 $OCR_OUT_DIR/model(학습 후)")
    ap.add_argument("--tag", default="base", help="base | finetuned → eval_<tag>.json/md")
    ap.add_argument("--rows", type=int, default=int(c.get("EVAL_ROWS", 500)), help="validation 앞 몇 행")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--data_dir", default=c.get("OCR_DATA_DIR", ""))
    ap.add_argument("--out_dir", default=c.get("OCR_OUT_DIR", "outputs/donut-ocr"))
    ap.add_argument("--image_h", type=int, default=int(c.get("IMAGE_H", 1280)))
    ap.add_argument("--image_w", type=int, default=int(c.get("IMAGE_W", 960)))
    ap.add_argument("--max_len", type=int, default=int(c.get("MAX_LEN", 768)))
    ap.add_argument("--compare", action="store_true", help="eval_base.json + eval_finetuned.json → compare.md 만")
    a = ap.parse_args(argv)
    if not a.compare and not a.model:
        ap.error("--model 이 비어 있습니다 (config.env 의 MODEL_ID)")
    return a


def generate_batch(model, processor, images, task_id: int, max_len: int, image_h: int, image_w: int) -> list[str]:
    """이미지 묶음 → 정리된 문자열 목록."""
    import torch
    tok = processor.tokenizer
    dtype = next(model.parameters()).dtype
    pv = pixel_values(processor, images, image_h, image_w).to(model.device, dtype=dtype)
    start = torch.full((len(images), 1), task_id, dtype=torch.long, device=model.device)
    with torch.inference_mode():
        out = model.generate(pv, decoder_input_ids=start, max_length=max_len, num_beams=1, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             bad_words_ids=[[tok.unk_token_id]], use_cache=True, return_dict_in_generate=True)
    seqs = out.sequences if hasattr(out, "sequences") else out
    return [clean_text(t) for t in tok.batch_decode(seqs, skip_special_tokens=True)]


def fmt(key: str, v) -> str:
    if v is None:
        return "-"
    if key == "sec_mean":
        return f"{v:.3f}"
    return f"{v * 100:.2f}%"


def write_md(path: Path, res: dict) -> None:
    s, rows = res["summary"], res["rows"]
    lines = [f"# OCR 평가 — {res['tag']}", "",
             f"- 모델: `{res['model']}`", f"- 데이터: validation 앞 {s['n']}행 · image {res['image_size'][0]}x{res['image_size'][1]} · "
             f"max_len {res['max_len']} · greedy · {res['dtype']} · {res['device']}", "",
             "| 지표 | 값 |", "| --- | --- |"]
    lines += [f"| {name} | {fmt(k, s.get(k))} |" for k, name in METRICS]

    def block(title: str, items: list[dict]) -> None:
        lines.extend(["", f"## {title}", ""])
        for r in items:
            lines.append(f"**#{r['i']}** · CER {r['cer'] * 100:.1f}% · {r['sec']:.2f}s")
            lines.append(f"- ref: `{r['ref']}`")
            lines.append(f"- hyp: `{r['hyp']}`")
            lines.append("")

    block("가장 나쁜 10건 (CER 순)", sorted(rows, key=lambda r: -r["cer"])[:10])
    block("무작위 5건 (seed 0)", random.Random(0).sample(rows, min(5, len(rows))))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def compare(out_dir: Path) -> Path:
    """eval_base.json + eval_finetuned.json → compare.md (학습 전/후 한 표 + 개선 폭)."""
    b = json.loads((out_dir / "eval_base.json").read_text(encoding="utf-8"))
    f = json.loads((out_dir / "eval_finetuned.json").read_text(encoding="utf-8"))
    sb, sf = b["summary"], f["summary"]
    lines = ["# Donut 한국어 OCR — 학습 전/후 (SynthDoG-ko validation)", "",
             f"- 학습 전: `{b['model']}` · 학습 후: `{f['model']}`",
             f"- 행 수 {sb['n']} / {sf['n']} · image {f['image_size'][0]}x{f['image_size'][1]} · max_len {f['max_len']} · greedy", "",
             "| 지표 | 학습 전 | 학습 후 | 변화 |", "| --- | --- | --- | --- |"]
    for k, name in METRICS:
        vb, vf = sb.get(k), sf.get(k)
        if vb is None or vf is None:
            d = "-"
        elif k == "sec_mean":
            d = f"{vf - vb:+.3f}s"
        else:
            d = f"{(vf - vb) * 100:+.2f}%p"
        lines.append(f"| {name} | {fmt(k, vb)} | {fmt(k, vf)} | {d} |")
    if sb["n"] != sf["n"]:
        lines += ["", f"⚠️ 두 평가의 행 수가 다르다 ({sb['n']} vs {sf['n']}) — 같은 --rows 로 다시 돌릴 것"]
    p = out_dir / "compare.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.compare:
        p = compare(out_dir)
        print(p.read_text(encoding="utf-8"))
        print(f"→ {p}")
        return

    import torch
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    dev_name = torch.cuda.get_device_name(0) if cuda else "cpu"
    model, processor, task_id = load_model(args.model, args.image_h, args.image_w, args.max_len, dtype)
    model.to("cuda" if cuda else "cpu").eval()
    ds = load_split(args.data_dir, "validation", args.rows)
    print(f"평가 [{args.tag}] {args.model} · {len(ds)}행 · batch {args.batch} · {dtype} · {dev_name} · task id {task_id}", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    n_batches = (len(ds) + args.batch - 1) // args.batch
    try:
        for bi in range(n_batches):
            chunk = ds.select(range(bi * args.batch, min(len(ds), (bi + 1) * args.batch)))
            recs = [chunk[j] for j in range(len(chunk))]
            images = [decode_image(r) for r in recs]
            if cuda:
                torch.cuda.synchronize()
            tb = time.time()
            hyps = generate_batch(model, processor, images, task_id, args.max_len, args.image_h, args.image_w)
            if cuda:
                torch.cuda.synchronize()
            sec = (time.time() - tb) / len(recs)
            for j, (r, h) in enumerate(zip(recs, hyps)):
                sp = score_pair(gt_text(r), h)
                sp.update(i=bi * args.batch + j, sec=sec)
                rows.append(sp)
            if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
                el = time.time() - t0
                s = summarize(rows)
                gib = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
                print(f"  {len(rows)}/{len(ds)} · CER {s['cer_mean'] * 100:.2f}% · {el / len(rows):.2f}s/장 "
                      f"· ETA {(len(ds) - len(rows)) * el / len(rows) / 60:.1f}min · GPU {gib:.2f} GiB", flush=True)
    except torch.cuda.OutOfMemoryError:
        sys.exit(f"!!! 평가 OOM (batch {args.batch}) — 배치를 줄여 다시: bash eval_ocr.sh {args.tag} --batch 2  (그래도면 --batch 1)")

    res = {"tag": args.tag, "model": args.model, "image_size": [args.image_h, args.image_w], "max_len": args.max_len,
           "dtype": str(dtype).replace("torch.", ""), "device": dev_name, "batch": args.batch,
           "summary": summarize(rows), "versions": lib_versions(), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "rows": rows}
    jp, mp = out_dir / f"eval_{args.tag}.json", out_dir / f"eval_{args.tag}.md"
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(mp, res)
    s = res["summary"]
    print(f"\n══ [{args.tag}] " + " · ".join(f"{name} {fmt(k, s.get(k))}" for k, name in METRICS))
    print(f"→ {jp}\n→ {mp}")


if __name__ == "__main__":
    main()
