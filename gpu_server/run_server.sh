#!/usr/bin/env bash
# gpu_server/app.py 를 pod 에서 켜고 끈다.
#   bash gpu_server/run_server.sh start | stop | status | restart | log
# pidfile 은 SCRATCH_DIR(overlay) — pod 가 재시작되면 프로세스와 함께 사라지므로 묵은 pidfile 이 남지 않는다.
# 서버는 setsid 로 자기 프로세스 그룹의 리더가 되고, stop 은 그룹 전체(학습 자식 포함)를 끝낸다.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f config.env ] && { set -a; source ./config.env; set +a; }
[ -f config.local.env ] && { set -a; source ./config.local.env; set +a; }     # UNIVUS_API_KEY · LOCAL_SERVER_URL 은 여기(gitignore)
PERSIST_DIR="${PERSIST_DIR:-$HOME/univus-train-out}"
SCRATCH_DIR="${SCRATCH_DIR:-$HOME/.venv/univus-train}"
PY="$SCRATCH_DIR/.venv/bin/python"
HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${SERVER_PORT:-8600}"
PID_FILE="$SCRATCH_DIR/gpu_server.pid"
LOG="$PERSIST_DIR/logs/server.log"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH_DIR/hf_cache}" PYTHONUTF8=1 PERSIST_DIR SCRATCH_DIR

pid_of() { [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null; }
is_ours() {   # pid 가 살아 있고 /proc/<pid>/cmdline 에 gpu_server/app.py 가 있을 때만 우리 서버로 본다
  local p="$1"; [ -n "$p" ] && [ -d "/proc/$p" ] && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "gpu_server/app.py"
}
alive() { is_ours "$(pid_of)"; }

case "${1:-status}" in
  start)
    if alive; then echo "이미 실행 중 (pid $(pid_of)) — http://$HOST:$PORT/health"; exit 0; fi
    rm -f "$PID_FILE"
    [ -x "$PY" ] || { echo "venv 없음 ($PY) — pod 재시작 후라면 정상:  bash gpu_server/setup_server.sh"; exit 1; }
    if [ -z "${UNIVUS_API_KEY:-}" ] && [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ]; then
      echo "거부: UNIVUS_API_KEY 없이 $HOST 로 열 수 없습니다 — config.local.env 에 UNIVUS_API_KEY=... 를 넣거나 SERVER_HOST=127.0.0.1"; exit 1
    fi
    [ -n "${UNIVUS_API_KEY:-}" ] || echo "ℹ️ UNIVUS_API_KEY 가 비어 있음 — 루프백(ssh 터널) 전용으로 뜹니다"
    mkdir -p "$PERSIST_DIR/logs" "$SCRATCH_DIR"
    setsid nohup "$PY" gpu_server/app.py --host "$HOST" --port "$PORT" --pidfile "$PID_FILE" >> "$LOG" 2>&1 < /dev/null &
    for _ in 1 2 3 4 5 6 7 8 9 10; do sleep 1; alive && break; done
    if alive; then echo "시작됨 pid $(pid_of) · 로그 $LOG · 상태:  bash gpu_server/run_server.sh status"; else echo "시작 실패 — tail $LOG"; tail -20 "$LOG"; exit 1; fi ;;
  stop)
    p="$(pid_of)"
    if ! is_ours "$p"; then echo "실행 중 아님"; rm -f "$PID_FILE"; exit 0; fi
    kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null        # 프로세스 그룹(학습 자식 포함)
    for _ in $(seq 1 30); do is_ours "$p" || break; sleep 1; done
    if is_ours "$p"; then kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null; sleep 1; fi
    if is_ours "$p"; then echo "종료 실패 (pid $p)"; exit 1; fi
    rm -f "$PID_FILE"; echo "중지됨" ;;
  restart) "$0" stop; "$0" start ;;
  status)
    if alive; then echo "실행 중 pid $(pid_of) · $HOST:$PORT"; curl -s -H "Authorization: Bearer ${UNIVUS_API_KEY:-}" "http://127.0.0.1:$PORT/health" | head -c 600; echo; else echo "실행 중 아님"; fi ;;
  log) tail -n "${2:-50}" -f "$LOG" ;;
  *) echo "사용법: $0 start|stop|status|restart|log"; exit 1 ;;
esac
