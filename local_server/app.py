#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""로컬(내 PC) 데이터 서버 — GPU 서버가 학습 데이터를 받아 가고, 학습 결과를 올려 두는 곳.

  python local_server/app.py --port 8601            # 127.0.0.1:8601
  UNIVUS_API_KEY=... python local_server/app.py     # 키가 있으면 모든 요청에 Authorization: Bearer <키> 필요

GPU 서버(pod)는 바깥으로 나올 수 없으므로 이 서버는 **SSH 역터널(-R)** 로만 pod 에 보인다
(scripts/tunnel.ps1). 그래서 127.0.0.1 에만 묶고, 키가 없어도 동작은 하되 켜 두는 것을 권장한다.

엔드포인트
  GET  /health                       상태 · 데이터 폴더 · 파일 목록
  GET  /data/manifest                data/f9 의 파일 이름·크기·sha256 (GPU 서버가 받은 뒤 검증한다)
  GET  /data/<name>                  train.jsonl · eval.jsonl · README.md 만 (경로 조작 거부)
  POST /results/<run>/<path>         GPU 서버가 올리는 결과 — results/<run>/<path> 에 저장 (파일당 100MB)
  GET  /results                      results/ 트리
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402,F401 — stdout UTF-8 재설정만 빌려 쓴다

ROOT = Path(__file__).resolve().parents[1]
DATA_WHITELIST = ("train.jsonl", "eval.jsonl", "README.md")
MAX_UPLOAD = 100 * 1024 * 1024          # 파일당 100MB — 어댑터(수십 MB)·평가 md/json 이 전부다
RUN_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
CFG = {"api_key": "", "data_dir": ROOT / "data" / "f9", "results_dir": ROOT / "results", "started": time.time()}


# ── 공통 ─────────────────────────────────────────────────────

def _authorized() -> bool:
    key = CFG["api_key"]
    if not key:
        return True
    auth = request.headers.get("Authorization", "")
    return auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip(), key)


@app.before_request
def _auth_gate():
    if not _authorized():
        return jsonify({"error": "unauthorized", "detail": "Authorization: Bearer <UNIVUS_API_KEY> 가 필요합니다"}), 401
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(rel: str) -> Path:
    """업로드 경로 검사 — 절대경로·'..'·빈 조각 거부. 하위 폴더(adapter/…)는 허용."""
    parts = [p for p in rel.replace("\\", "/").split("/")]
    if not parts or any(p in ("", ".", "..") for p in parts) or rel.startswith("/") or ":" in rel:
        abort(400, description="잘못된 파일 경로")
    return Path(*parts)


def _safe_run(run: str) -> str:
    if not run or len(run) > 80 or any(c not in RUN_CHARS for c in run) or run in (".", ".."):
        abort(400, description="run 이름은 영문·숫자·._- 만")
    return run


# ── 엔드포인트 ───────────────────────────────────────────────

@app.get("/health")
def health():
    d = CFG["data_dir"]
    return jsonify({"ok": True, "data_dir": str(d), "results_dir": str(CFG["results_dir"]),
                    "files": sorted(p.name for p in d.iterdir() if p.is_file()) if d.exists() else [],
                    "auth": bool(CFG["api_key"]), "uptime_s": round(time.time() - CFG["started"])})


@app.get("/data/manifest")
def manifest():
    d = CFG["data_dir"]
    files = []
    for name in DATA_WHITELIST:
        p = d / name
        if p.is_file():
            files.append({"name": name, "size": p.stat().st_size, "sha256": _sha256(p),
                          "mtime": int(p.stat().st_mtime)})
    return jsonify({"data_dir": str(d), "files": files})


@app.get("/data/<name>")
def data_file(name: str):
    if name not in DATA_WHITELIST:          # 화이트리스트 밖(…/, ..)은 전부 404 — 경로 조작 차단
        abort(404, description="허용된 파일이 아닙니다 (train.jsonl · eval.jsonl · README.md)")
    p = CFG["data_dir"] / name
    if not p.is_file():
        abort(404, description=f"{name} 없음 — 먼저 python src/prepare_f9.py")
    resp = send_file(p, as_attachment=True, download_name=name, mimetype="application/octet-stream")
    resp.headers["X-SHA256"] = _sha256(p)
    return resp


@app.post("/results/<run>/<path:filename>")
def upload_result(run: str, filename: str):
    run = _safe_run(run)
    rel = _safe_rel(filename)
    dest = (CFG["results_dir"] / run / rel).resolve()
    base = CFG["results_dir"].resolve()
    if base not in dest.parents:
        abort(400, description="결과 폴더 밖으로 나갈 수 없습니다")
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    size = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        while True:
            chunk = request.stream.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close(); tmp.unlink(missing_ok=True)
                abort(413, description="파일당 100MB 까지")
            f.write(chunk); h.update(chunk)
    tmp.replace(dest)
    return jsonify({"saved": str(dest.relative_to(base)).replace("\\", "/"), "bytes": size, "sha256": h.hexdigest()})


@app.get("/results")
def list_results():
    base = CFG["results_dir"]
    tree: dict[str, list[dict]] = {}
    if base.exists():
        for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            tree[run_dir.name] = [{"path": str(p.relative_to(run_dir)).replace("\\", "/"), "size": p.stat().st_size}
                                  for p in sorted(run_dir.rglob("*")) if p.is_file()]
    return jsonify({"results_dir": str(base), "runs": tree})


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(413)
def _err(e):
    return jsonify({"error": getattr(e, "name", "error"), "detail": getattr(e, "description", str(e))}), e.code


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="역터널로만 노출되므로 127.0.0.1 고정 권장")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LOCAL_PORT", "8601")))
    ap.add_argument("--data-dir", default=os.environ.get("LOCAL_DATA_DIR", str(ROOT / "data" / "f9")))
    ap.add_argument("--results-dir", default=os.environ.get("LOCAL_RESULTS_DIR", str(ROOT / "results")))
    ap.add_argument("--pidfile", default="", help="내 PID 를 적어 둘 파일 (종료 스크립트용)")
    args = ap.parse_args()
    CFG["api_key"] = os.environ.get("UNIVUS_API_KEY", "").strip()
    CFG["data_dir"] = Path(args.data_dir).resolve()
    CFG["results_dir"] = Path(args.results_dir).resolve()
    CFG["results_dir"].mkdir(parents=True, exist_ok=True)
    if args.pidfile:
        Path(args.pidfile).write_text(str(os.getpid()), encoding="utf-8")
    print(f"local_server  http://{args.host}:{args.port}  data={CFG['data_dir']}  results={CFG['results_dir']}  "
          f"auth={'on' if CFG['api_key'] else 'OFF (UNIVUS_API_KEY 비어 있음)'}")
    if not (CFG["data_dir"] / "train.jsonl").exists():
        print("  ⚠️ train.jsonl 이 없습니다 — python src/prepare_f9.py --n 1500 --seed 42 로 먼저 만드세요")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
