# -*- coding: utf-8 -*-
"""Donut OCR(prepare_ocr · train_ocr · eval_ocr)이 같이 쓰는 도우미. 모델 이름은 여기 없다 — config.env 의 MODEL_ID.

  · load_cfg()         config.env → config.local.env → 셸 env 순으로 덮어쓴 dict (train.sh 와 같은 규칙)
  · load_split()       {data_dir}/{split}-*.parquet → datasets.Dataset (열: image={bytes,path} · ground_truth=JSON 문자열)
  · gt_text()          ground_truth JSON → text_sequence
  · decode_image()     image 열 → PIL RGB
  · load_model()       VisionEncoderDecoderModel + DonutProcessor 를 image_size·task 토큰까지 맞춰서
  · normalize/cer/ned_sim/score_pair  채점 (editdistance 로 직접 — evaluate·jiwer 없음)
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

# Windows 콘솔(cp949)에서 한국어 출력이 깨지지 않게
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 파이프·구버전이면 그대로 둔다
        pass

ROOT = Path(__file__).resolve().parents[1]
TASK_TOKEN = "<s_synthdog>"
# 디코드 뒤에도 남을 수 있는 표식 — skip_special_tokens 만 믿지 않고 직접 지운다
STRIP_TOKENS = (TASK_TOKEN, "</s_synthdog>", "<s>", "</s>", "<pad>", "<unk>")


# ── 설정 ────────────────────────────────────────────────────

def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = re.split(r"\s+#", v, maxsplit=1)[0].strip().strip('"').strip("'")   # 줄 끝 주석 제거 (bash source 와 같게)
        out[k.strip()] = os.path.expandvars(v)
    return out


def load_cfg(root: Path = ROOT) -> dict[str, str]:
    """config.env → config.local.env(있으면) → 셸 env. 뒤의 것이 이긴다."""
    cfg = _read_env_file(root / "config.env")
    cfg.update(_read_env_file(root / "config.local.env"))
    for k in list(cfg) + [k for k in os.environ if k.startswith(("OCR_", "HF_HUB_"))]:
        if k in os.environ:
            cfg[k] = os.environ[k]
    for k in ("MODEL_ID", "IMAGE_H", "IMAGE_W", "MAX_LEN", "BATCH", "GRAD_ACCUM", "LR", "WARMUP_STEPS",
              "MAX_TRAIN_ROWS", "MAX_STEPS", "TIME_BUDGET_MIN", "SAVE_STEPS", "EVAL_ROWS"):
        if k in os.environ:
            cfg[k] = os.environ[k]
    return cfg


def lib_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in ("torch", "transformers", "datasets", "tokenizers", "sentencepiece", "pillow", "editdistance", "accelerate"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


# ── 데이터 ──────────────────────────────────────────────────

def load_split(data_dir: str | Path, split: str, limit: int = 0):
    """`{split}-*.parquet` 를 모아 한 Dataset 으로. limit>0 이면 앞에서 limit 행.
    limit 이 있으면 그만큼을 채우는 앞쪽 파일만 읽는다(Arrow 캐시 변환 시간·용량 절약 — 캐시는 HF_DATASETS_CACHE)."""
    from datasets import load_dataset
    files = sorted(glob(os.path.join(str(data_dir), f"{split}-*.parquet")))
    if not files:
        sys.exit(f"[중단] {data_dir}/{split}-*.parquet 가 없습니다 — config.env 의 OCR_DATA_DIR 확인")
    if limit and limit > 0:
        import pyarrow.parquet as pq
        need, n = [], 0
        for f in files:
            need.append(f)
            n += pq.ParquetFile(f).metadata.num_rows
            if n >= limit:
                break
        files = need
    ds = load_dataset("parquet", data_files=files, split="train")
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def gt_text(row) -> str:
    """ground_truth = '{"gt_parse": {"text_sequence": "..."}}'. 이미 dict 이거나 JSON 이 아니어도 안전하게."""
    gt = row["ground_truth"] if isinstance(row, dict) else row
    if isinstance(gt, (bytes, bytearray)):
        gt = gt.decode("utf-8")
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except ValueError:
            return gt
    if isinstance(gt, dict):
        parse = gt.get("gt_parse", gt)
        if isinstance(parse, dict):
            return str(parse.get("text_sequence", ""))
        return str(parse)
    return str(gt)


def decode_image(row):
    """image 열: 서버 parquet 은 {"bytes": ..., "path": ...} dict. Image feature 로 이미 풀렸거나 bytes 여도 받는다."""
    from PIL import Image
    img = row["image"] if isinstance(row, dict) and "image" in row else row
    if isinstance(img, dict):
        if img.get("bytes") is not None:
            img = img["bytes"]
        else:
            return Image.open(img["path"]).convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img)).convert("RGB")
    return img.convert("RGB")


# ── 모델 ────────────────────────────────────────────────────

def configure_model(model, processor, image_h: int, image_w: int, max_len: int) -> int:
    """image_size·task 토큰·pad/eos 를 모델·프로세서·generation_config 에 맞춘다. task 토큰 id 를 돌려준다."""
    tok = processor.tokenizer
    task_id = tok.convert_tokens_to_ids(TASK_TOKEN)
    if task_id is None or task_id == tok.unk_token_id:          # 사전학습 체크포인트에는 있다. 없으면 추가하고 임베딩을 늘린다
        tok.add_tokens([TASK_TOKEN], special_tokens=True)
        task_id = tok.convert_tokens_to_ids(TASK_TOKEN)
    if len(tok) > model.config.decoder.vocab_size:
        model.decoder.resize_token_embeddings(len(tok))
    processor.image_processor.size = {"height": image_h, "width": image_w}
    model.config.encoder.image_size = [image_h, image_w]
    model.config.decoder_start_token_id = task_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    gc = model.generation_config
    gc.decoder_start_token_id = task_id
    gc.pad_token_id = tok.pad_token_id
    gc.eos_token_id = tok.eos_token_id
    gc.max_length = max_len
    return task_id


def load_model(model_id: str, image_h: int, image_w: int, max_len: int, dtype):
    """(model, processor, task_id). image_size 는 모델을 만들기 전에 config 에도 넣어 둔다."""
    from transformers import DonutProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
    config = VisionEncoderDecoderConfig.from_pretrained(model_id)
    config.encoder.image_size = [image_h, image_w]
    model = VisionEncoderDecoderModel.from_pretrained(model_id, config=config, dtype=dtype)
    processor = DonutProcessor.from_pretrained(model_id)
    task_id = configure_model(model, processor, image_h, image_w, max_len)
    return model, processor, task_id


def pixel_values(processor, images, image_h: int, image_w: int):
    """[B,3,H,W] float32. size 를 매번 명시해 저장된 프로세서 설정과 무관하게 같은 해상도로."""
    return processor.image_processor(images, size={"height": image_h, "width": image_w},
                                     return_tensors="pt").pixel_values


def clean_text(text: str) -> str:
    for t in STRIP_TOKENS:
        text = text.replace(t, " ")
    return normalize(text)


# ── 채점 ────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """양끝 공백 제거 + 연속 공백 1개. SynthDoG 의 글자 사이 공백('미 늄 매 트를')은 그대로 둔다."""
    return re.sub(r"\s+", " ", s or "").strip()


def cer(ref: str, hyp: str) -> float:
    import editdistance
    return editdistance.eval(ref, hyp) / max(1, len(ref))


def ned_sim(ref: str, hyp: str) -> float:
    """1 - NED = 1 - 편집거리/max(len). 둘 다 비면 1."""
    import editdistance
    n = max(len(ref), len(hyp))
    return 1.0 if n == 0 else 1.0 - editdistance.eval(ref, hyp) / n


def score_pair(ref: str, hyp: str) -> dict:
    """normalize 뒤 채점. 가중 CER 합산용 edits·ref_len 도 같이."""
    import editdistance
    ref, hyp = normalize(ref), normalize(hyp)
    e = editdistance.eval(ref, hyp)
    return {"ref": ref, "hyp": hyp, "edits": int(e), "ref_len": len(ref),
            "cer": e / max(1, len(ref)), "ned_sim": ned_sim(ref, hyp), "exact": ref == hyp}


def summarize(rows: list[dict]) -> dict:
    """행별 score_pair(+sec) → 요약. CER 평균 · 1-NED 평균 · 정확 일치율 · 문자 가중 CER · 평균 지연."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    tot_ref = sum(r["ref_len"] for r in rows)
    return {
        "n": n,
        "cer_mean": sum(r["cer"] for r in rows) / n,
        "ned_sim_mean": sum(r["ned_sim"] for r in rows) / n,
        "exact_rate": sum(1 for r in rows if r["exact"]) / n,
        "cer_weighted": sum(r["edits"] for r in rows) / max(1, tot_ref),
        "sec_mean": sum(r.get("sec", 0.0) for r in rows) / n,
    }
