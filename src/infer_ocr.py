#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""학습된 Donut 한국어 OCR 추론 — WE-Meet 백엔드가 그대로 가져다 쓰는 파일 하나 (표준 라이브러리 + torch · transformers · Pillow).

WE-Meet `osenv.ocr`(OS 내장 OCR)과 같은 모양의 모듈 함수:
    available(refresh=False)    {"ok", "engine", "lang", "korean", "reason", "device", "model_dir"} — 처음 부를 때 모델을 한 번 올린다
    image_texts([이미지, …])     이미지마다 글자(못 읽으면 None). 이미지 = bytes · 파일 경로(str/Path) · PIL.Image
    image_text(이미지)           한 장
    pdf_text(bytes, pages=3)    스캔 PDF 앞 몇 쪽 — pypdfium2 가 깔려 있을 때만(없으면 None, 이유는 pdf_support())

모델 폴더(train_ocr.py 가 저장한 bf16 safetensors + processor + tokenizer):
    env OCR_MODEL_DIR → 없으면 <레포>/outputs/donut-ocr-ko/model → <레포>/results/donut-ocr-ko/model.
    OCR_MODEL_DIR 을 지정했으면 그 폴더만 본다. 못 찾으면 available()["ok"] 가 False, 함수는 None.
UNIVUS_OCR=off 면 끈다(osenv.ocr 과 같다). GPU 가 있으면 cuda·bf16, 없으면 cpu·fp32.

생성은 eval_ocr.py(검증 500장 CER 5.19%)와 똑같다: decoder_input_ids=[[<s_synthdog>]] · greedy(num_beams=1) ·
max_length = 저장된 generation_config(768) · <unk> 금지 · 특수 토큰 제거 → normalize(양끝 공백 제거·연속 공백 1개).
해상도는 저장된 config 의 encoder.image_size(1280x960)를 그대로 쓴다.

    python src/infer_ocr.py [--model DIR] [--device cuda|cpu] [--batch N] [--json] 이미지 …
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "donut-ocr-ko"
TASK_TOKEN = "<s_synthdog>"
# 디코드 뒤에도 남을 수 있는 표식 — skip_special_tokens 만 믿지 않고 직접 지운다 (ocr_common.STRIP_TOKENS 와 같다)
STRIP_TOKENS = (TASK_TOKEN, "</s_synthdog>", "<s>", "</s>", "<pad>", "<unk>")
MAX_BYTES = 15 * 1024 * 1024              # osenv.ocr 과 같은 한도 — 넘는 bytes 는 None
PDF_DPI = 150
ENV_DIR = "OCR_MODEL_DIR"
DEFAULT_DIRS = (ROOT / "outputs" / "donut-ocr-ko" / "model", ROOT / "results" / "donut-ocr-ko" / "model")

ImageLike = Union[bytes, bytearray, memoryview, str, Path, Any]   # Any = PIL.Image.Image
log = logging.getLogger(__name__)


def normalize(s: str) -> str:
    """양끝 공백 제거 + 연속 공백 1개 (ocr_common.normalize 와 같다)."""
    return re.sub(r"\s+", " ", s or "").strip()


def clean_text(text: str) -> str:
    for t in STRIP_TOKENS:
        text = text.replace(t, " ")
    return normalize(text)


def to_image(x: ImageLike):
    """bytes · 경로 · PIL → PIL RGB. 못 열면 None (예외를 밖으로 내지 않는다)."""
    from PIL import Image
    try:
        if isinstance(x, (bytes, bytearray, memoryview)):
            data = bytes(x)
            if not data or len(data) > MAX_BYTES:
                return None
            img = Image.open(io.BytesIO(data))
        elif isinstance(x, (str, Path)):
            img = Image.open(x)
        elif isinstance(x, Image.Image):
            img = x
        else:
            return None
        img.load()                                     # 잘린 파일은 여기서 실패한다
        return img.convert("RGB")
    except Exception:                                  # noqa: BLE001 — 깨진 이미지·없는 파일·폭탄 이미지
        return None


class DonutOCR:
    """모델을 한 번 올려 두고 이미지 → 글자. 스레드 여러 개에서 불러도 생성은 한 번에 하나씩."""

    def __init__(self, model_dir: Union[str, Path], device: Optional[str] = None, dtype=None,
                 max_len: Optional[int] = None, batch: int = 4):
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        self.model_dir = Path(model_dir).resolve()
        if not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"{self.model_dir} 에 config.json 이 없습니다 (학습된 모델 폴더가 아님)")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.batch = max(1, int(batch))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_dir), dtype=dtype).to(self.device).eval()
        self.processor = DonutProcessor.from_pretrained(str(self.model_dir))
        tok = self.processor.tokenizer
        task_id = tok.convert_tokens_to_ids(TASK_TOKEN)
        if task_id is None or task_id == tok.unk_token_id:
            task_id = self.model.config.decoder_start_token_id
        if task_id is None:
            raise ValueError(f"{TASK_TOKEN} 토큰도 decoder_start_token_id 도 없습니다")
        self.task_id = int(task_id)
        size = self.model.config.encoder.image_size
        self.image_h, self.image_w = (int(size[0]), int(size[1])) if isinstance(size, (list, tuple)) else (int(size), int(size))
        gc_len = getattr(self.model.generation_config, "max_length", None)
        self.max_len = int(max_len or gc_len or getattr(self.model.config.decoder, "max_length", None) or 768)
        n_pos = getattr(self.model.config.decoder, "max_position_embeddings", None)
        if n_pos:                                      # 위치 임베딩을 넘으면 index 오류(cuda 에선 device assert) — 학습 모델은 768 < 1536
            self.max_len = min(self.max_len, int(n_pos))
        self._lock = threading.Lock()

    def available(self) -> dict:
        return {"ok": True, "engine": ENGINE, "lang": "ko", "korean": True, "device": str(self.device),
                "model_dir": str(self.model_dir), "reason": ""}

    def _generate(self, images: list) -> list[str]:
        """eval_ocr.generate_batch 와 같은 호출."""
        import torch
        tok = self.processor.tokenizer
        pv = self.processor.image_processor(images, size={"height": self.image_h, "width": self.image_w},
                                            return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)
        start = torch.full((len(images), 1), self.task_id, dtype=torch.long, device=self.device)
        with self._lock, torch.inference_mode():
            out = self.model.generate(pv, decoder_input_ids=start, max_length=self.max_len, num_beams=1, do_sample=False,
                                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                      bad_words_ids=[[tok.unk_token_id]], use_cache=True, return_dict_in_generate=True)
        seqs = out.sequences if hasattr(out, "sequences") else out
        return [clean_text(t) for t in tok.batch_decode(seqs, skip_special_tokens=True)]

    def image_texts(self, images: list[ImageLike], batch: Optional[int] = None) -> list[Optional[str]]:
        """이미지마다 글자. 못 연 이미지·생성 실패는 그 자리만 None — 한 장 때문에 예외를 내지 않는다."""
        import torch
        out: list[Optional[str]] = [None] * len(images)
        todo = [(i, img) for i, img in enumerate(to_image(x) for x in images) if img is not None]
        bs = max(1, int(batch or self.batch))
        for k in range(0, len(todo), bs):
            chunk = todo[k:k + bs]
            try:
                texts = self._generate([img for _, img in chunk])
            except Exception as e:                     # noqa: BLE001 — OOM 등. 한 장씩 다시
                log.warning("OCR batch of %d failed (%s) — retrying one by one", len(chunk), type(e).__name__)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                texts = []
                for _, img in chunk:
                    try:
                        texts.append(self._generate([img])[0])
                    except Exception as e1:            # noqa: BLE001
                        log.warning("OCR failed for one image: %s: %s", type(e1).__name__, e1)
                        texts.append(None)
            for (i, _), t in zip(chunk, texts):
                out[i] = t
        return out

    def image_text(self, x: ImageLike) -> Optional[str]:
        return self.image_texts([x])[0]


# ── 모듈 함수 (osenv.ocr 과 같은 모양) ───────────────────────────

_engine: Optional[DonutOCR] = None
_avail: Optional[dict] = None
_load_lock = threading.Lock()


def _off() -> bool:
    return os.environ.get("UNIVUS_OCR", "").strip().lower() in ("off", "0", "false", "no")


def find_model_dir() -> tuple[Optional[Path], list[Path]]:
    """(찾은 폴더 또는 None, 찾아본 곳). OCR_MODEL_DIR 이 있으면 그것만."""
    env = os.environ.get(ENV_DIR, "").strip()
    cands = [Path(env).expanduser()] if env else list(DEFAULT_DIRS)
    for p in cands:
        if (p / "config.json").is_file():
            return p, cands
    return None, cands


def available(refresh: bool = False) -> dict:
    """처음(또는 refresh=True) 부를 때 모델을 올린다. 같은 폴더면 다시 올리지 않는다."""
    global _engine, _avail
    if _avail is not None and not refresh:
        return _avail
    with _load_lock:
        if _avail is not None and not refresh:
            return _avail
        res = {"ok": False, "engine": ENGINE, "lang": "ko", "korean": False, "device": None, "model_dir": None, "reason": ""}
        d, cands = (None, []) if _off() else find_model_dir()
        if _off():
            res["reason"] = "UNIVUS_OCR=off"
            _engine = None
        elif d is None:
            res["reason"] = (f"OCR 모델 폴더가 없습니다 — env {ENV_DIR} 에 학습된 모델 폴더(config.json 이 있는 곳)를 지정하세요 "
                             f"(찾아본 곳: {', '.join(str(p) for p in cands)})")
            _engine = None
        else:
            try:
                if _engine is None or _engine.model_dir != d.resolve():
                    _engine = None
                    _engine = DonutOCR(d)
                res = _engine.available()
            except Exception as e:                     # noqa: BLE001 — 깨진 가중치·torch 없음 등
                _engine = None
                res["model_dir"] = str(d)
                res["reason"] = f"OCR 모델을 올리지 못했습니다 ({d}): {type(e).__name__}: {e}"
        _avail = res
        return res


def image_texts(images: list[ImageLike]) -> list[Optional[str]]:
    if not images:
        return []
    if not available()["ok"] or _engine is None:
        return [None] * len(images)
    return _engine.image_texts(images)


def image_text(x: ImageLike) -> Optional[str]:
    return image_texts([x])[0]


def pdf_support() -> tuple[bool, str]:
    """(pdf_text 를 쓸 수 있나, 못 쓰면 이유). 쪽을 그리는 데 pypdfium2 가 필요하다(requirements 에는 없다 — 선택)."""
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        return False, "pypdfium2 가 없어 PDF 쪽을 그릴 수 없습니다 — pip install pypdfium2"
    return True, ""


def pdf_text(data: bytes, pages: int = 3) -> Optional[str]:
    """스캔 PDF 앞 pages 쪽을 ~150dpi 로 그려 OCR, 쪽 사이는 빈 줄. 못 읽으면 None."""
    if not data or len(data) > MAX_BYTES:
        return None
    ok, why = pdf_support()
    if not ok:
        log.warning("pdf_text: %s", why)
        return None
    if not available()["ok"] or _engine is None:
        return None
    import pypdfium2 as pdfium
    try:
        doc = pdfium.PdfDocument(bytes(data))
    except Exception:                                  # noqa: BLE001 — 깨진·암호 PDF
        return None
    try:
        imgs = []
        for i in range(min(max(0, pages), len(doc))):
            page = doc[i]
            try:
                imgs.append(page.render(scale=PDF_DPI / 72).to_pil().convert("RGB"))
            finally:
                page.close()
    except Exception:                                  # noqa: BLE001
        return None
    finally:
        doc.close()
    texts = [t for t in _engine.image_texts(imgs) if t]
    return "\n\n".join(texts) if texts else None


# ── CLI ───────────────────────────────────────────────────────

def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                              # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="학습된 Donut 한국어 OCR — 이미지마다 글자와 s/장")
    ap.add_argument("images", nargs="+", help="이미지 파일")
    ap.add_argument("--model", default=None, help=f"모델 폴더 (기본: env {ENV_DIR} → outputs/… → results/donut-ocr-ko/model)")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--json", action="store_true", help="JSON 목록 [{path, text, sec}] 만 stdout 으로")
    a = ap.parse_args(argv)

    import torch
    d = Path(a.model) if a.model else find_model_dir()[0]
    if d is None:
        print(f"모델 폴더가 없습니다 — --model 또는 env {ENV_DIR}", file=sys.stderr)
        return 2
    t0 = time.time()
    eng = DonutOCR(d, device=a.device, batch=a.batch)
    cuda = eng.device.type == "cuda"
    print(f"[load] {eng.model_dir} · {eng.device} · {str(eng.dtype).replace('torch.', '')} · "
          f"{eng.image_h}x{eng.image_w} · max_len {eng.max_len} · {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for k in range(0, len(a.images), eng.batch):
        paths = a.images[k:k + eng.batch]
        if cuda:
            torch.cuda.synchronize()
        tb = time.time()
        texts = eng.image_texts(paths)
        if cuda:
            torch.cuda.synchronize()
        sec = (time.time() - tb) / len(paths)
        for p, t in zip(paths, texts):
            rows.append({"path": p, "text": t, "sec": round(sec, 3)})
            if not a.json:
                print(f"── {p} · {sec:.2f}s · {len(t) if t is not None else '-'}자\n{t if t is not None else '(읽지 못함)'}",
                      flush=True)
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    if cuda:
        print(f"[gpu] peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated · "
              f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
