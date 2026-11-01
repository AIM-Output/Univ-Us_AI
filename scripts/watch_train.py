#!/usr/bin/env python
"""학습 진행 게이지 — gpu_server 의 /train/status 를 주기적으로 읽어 한 화면으로 보여 준다 (표준 라이브러리만).

  내 PC :  .venv\\Scripts\\python.exe scripts\\watch_train.py              터널 8600 경유 · 키는 scripts\\server.env
  pod   :  ~/.venv/univus-train/.venv/bin/python scripts/watch_train.py     키는 config.local.env
  옵션  :  --interval 30   --once   --ascii   --url http://127.0.0.1:8600

화면
  qwen3-4b-f9 · training/train          2026-11-02 03:10:12 · 경과 37m
  받기 ✓ → 학습 ● → 평가 ○ → 업로드 ○
  [███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  9.5%   step 56 / 588
  loss 0.812 · epoch 0.29 / 3 · 39.3 s/step
  남은 ≈ 5:48:20 → 학습 종료 예상 08:58 (+ 평가 ≈ 1h)
  loss 추세 (5 step 마다): 1.53 1.21 0.98 0.91 0.81
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAR_RE = re.compile(r"(\d+)/(\d+) \[(\d+:\d+(?::\d+)?)<([\d:?]+),\s*([\d.]+)s/it\]")
LOSS_RE = re.compile(r"\{'loss': '?([\d.]+)'?.*?'epoch': '?([\d.]+)'?")
PHASES = [("download", "받기"), ("train", "학습"), ("eval", "평가"), ("upload", "업로드")]


def read_env_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"^{key}=(.*)$", line.strip())
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return ""


def load_key() -> str:
    k = os.environ.get("UNIVUS_API_KEY", "").strip()
    if k:
        return k
    for f in (ROOT / "config.local.env", ROOT / "scripts" / "server.env"):
        k = read_env_value(f, "UNIVUS_API_KEY")
        if k:
            return k
    return ""


def fetch(url: str, key: str, path: str, timeout: float = 15) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_tail(lines: list[str]) -> tuple:
    """log_tail → (step, total, s/step, 남은 시간 문자열, [(epoch, loss), …]). tqdm 은 \\r 로 덧쓰므로 한 줄에 여러 번 나올 수 있다."""
    step = total = None
    sec_per = None
    remaining = None
    losses: list[tuple[float, float]] = []
    for ln in lines:
        for m in BAR_RE.finditer(ln):
            step, total = int(m.group(1)), int(m.group(2))
            remaining, sec_per = m.group(4), float(m.group(5))
        for m in LOSS_RE.finditer(ln):
            losses.append((float(m.group(2)), float(m.group(1))))
    return step, total, sec_per, remaining, losses


def hms_to_sec(s: str) -> int | None:
    if not s or "?" in s:
        return None
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


def fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    h, m = sec // 3600, (sec % 3600) // 60
    return f"{h}h {m:02d}m" if h else f"{m}m"


def bar(frac: float, width: int, ascii_only: bool) -> str:
    frac = min(1.0, max(0.0, frac))
    n = int(round(frac * width))
    full, empty = ("#", ".") if ascii_only else ("█", "░")
    return full * n + empty * (width - n)


def phase_strip(d: dict, ascii_only: bool) -> str:
    done, cur, todo = ("v", "*", "o") if ascii_only else ("✓", "●", "○")
    phase, state = d.get("phase"), d.get("state")
    finished = state == "idle" and d.get("finished_at")
    keys = [k for k, _ in PHASES]
    idx = keys.index(phase) if phase in keys else (len(keys) if finished else -1)
    out = []
    for i, (_, label) in enumerate(PHASES):
        mark = done if i < idx else (cur if i == idx else todo)
        out.append(f"{label} {mark}")
    return " → ".join(out)


def render(d: dict, hist: dict[float, float], epochs: int, ascii_only: bool, width: int = 40) -> str:
    now = dt.datetime.now()
    run = d.get("run_name") or "-"
    state, phase = d.get("state"), d.get("phase")
    started = d.get("started_at")
    elapsed = f" · 경과 {fmt_dur(time.time() - started)}" if started else ""
    lines = [f" {run} · {state}/{phase or '-'}      {now:%Y-%m-%d %H:%M:%S}{elapsed}", " " + phase_strip(d, ascii_only), ""]

    step, total, sec_per, remaining, losses = parse_tail(d.get("log_tail") or [])
    for ep, loss in losses:
        hist[ep] = loss
    prog = d.get("progress") or {}

    if phase == "train" and step is not None and total:
        frac = step / total
        lines.append(f" [{bar(frac, width, ascii_only)}] {frac * 100:5.1f}%   step {step} / {total}")
        loss_s = f"loss {prog['loss']:.3f}" if "loss" in prog else "loss (5 step 뒤 표시)"
        ep_s = f"epoch {prog['epoch']:.2f} / {epochs}" if "epoch" in prog else f"epoch - / {epochs}"
        sp_s = f"{sec_per:.1f} s/step" if sec_per else ""
        lines.append(f" {loss_s} · {ep_s} · {sp_s}")
        rem = hms_to_sec(remaining or "")
        if rem is not None:
            eta = now + dt.timedelta(seconds=rem)
            lines.append(f" 남은 ≈ {remaining} → 학습 종료 예상 {eta:%H:%M} (+ 평가 ≈ 1h)")
    elif phase == "download":
        lines.append(" 내 PC 로컬 서버에서 train/eval 데이터 받는 중 (sha256 검증)")
    elif phase == "eval":
        lines.append(f" [{bar(1.0, width, ascii_only)}] 100.0%   학습 끝 → 평가 중 (학습 전 / 학습 후 2회 채점, ≈ 1h)")
    elif phase == "upload":
        lines.append(f" [{bar(1.0, width, ascii_only)}] 100.0%   결과를 내 PC results/{run}/ 로 올리는 중")
    elif state == "idle" and d.get("finished_at"):
        ok = d.get("ok")
        up = d.get("upload") or {}
        if ok is True:
            lines.append(f" [{bar(1.0, width, ascii_only)}] 완료 ✓  업로드 {up.get('n', 0)}개 → 내 PC results/{run}/  ·  checkpoint 정리 {len(d.get('cleaned') or [])}개")
        elif ok == "partial":
            lines.append(f" 완료(부분) — 업로드 실패 {up.get('n_failed', 0)}개. 어댑터는 pod outputs/{run}/ 에 남아 있음")
        else:
            lines.append(f" 실패 ✗  {d.get('error')}")
        if d.get("finished_at") and started:
            lines.append(f" 총 소요 {fmt_dur(d['finished_at'] - started)}")
    elif state == "idle":
        lines.append(" 학습 없음 (idle). 시작: POST /train/start {\"run_name\":\"…\"}")
    else:
        lines.append(f" {state}/{phase}")

    if d.get("error") and not (state == "idle" and d.get("finished_at")):
        lines.append(f" ⚠ {d['error']}")
    if hist:
        recent = [f"{hist[k]:.3f}" if hist[k] < 0.1 else f"{hist[k]:.2f}" for k in sorted(hist)[-10:]]   # 0.1 미만은 소수 3자리
        lines.append("")
        lines.append(" loss 추세 (5 step 마다): " + " ".join(recent))
    if phase in ("eval", "upload"):
        tail = [ln.strip()[-110:] for ln in (d.get("log_tail") or [])[-3:] if ln.strip()]
        lines += [""] + [f"   {t}" for t in tail]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("GPU_SERVER_URL", "http://127.0.0.1:8600"))
    ap.add_argument("--key", default="", help="비우면 UNIVUS_API_KEY 환경변수 → config.local.env → scripts/server.env")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--once", action="store_true", help="한 번만 출력하고 끝")
    ap.add_argument("--ascii", action="store_true", help="블록 문자 대신 #/. (글꼴이 깨질 때)")
    args = ap.parse_args()
    key = args.key or load_key()
    epochs = int(read_env_value(ROOT / "config.env", "EPOCHS") or 3)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        os.system("")          # 콘솔의 ANSI(VT) 처리 켜기
    hist: dict[float, float] = {}
    while True:
        try:
            d = fetch(args.url, key, "/train/status")
            screen = render(d, hist, epochs, args.ascii)
        except urllib.error.HTTPError as e:
            screen = f" HTTP {e.code} — {'인증 실패: 키 확인' if e.code == 401 else e.reason}"
        except Exception as e:  # noqa: BLE001
            screen = f" 연결 실패: {e}\n 터널(ssh -L 8600)과 pod gpu_server 가 켜져 있는지 확인"
        if not args.once:
            print("\x1b[2J\x1b[H", end="")
        print(screen)
        if args.once:
            break
        print(f"\n ({args.interval:.0f}초마다 갱신 · Ctrl+C 로 종료)")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
