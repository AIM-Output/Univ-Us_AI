#!/usr/bin/env bash
# gpu_server/app.py 를 pod 에서 켜고 끈다 (pidfile: $PERSIST_DIR/gpu_server.pid, 로그: $PERSIST_DIR/logs/server.log)
#   bash gpu_server/run_server.sh start | stop | status | restart | log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f config.env ] && { set -a; source ./config.env; set +a; }
[ -f config.local.env ] && { set -a; source ./config.local.env; set +a; }     # UNIVUS_API_KEY · LOCAL_SERVER_URL 은 여기(gitignore)
PERSIST_DIR="${PERSIST_DIR:-$HOME/univus-train-out}"
SCRATCH_DIR="${SCRATCH_DIR:-$HOME/.venv/univus-train}"
PY="$SCRATCH_DIR/.venv/bin/python"
PORT="${SERVER_PORT:-8600}"
PID_FILE="$PERSIST_DIR/gpu_server.pid"
LOG="$PERSIST_DIR/logs/server.log"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH_DIR/hf_cache}" PYTHONUTF8=1 PERSIST_DIR SCRATCH_DIR

alive() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if alive; then echo "이미 실행 중 (pid $(cat "$PID_FILE")) — http://127.0.0.1:$PORT/health"; exit 0; fi
    [ -x "$PY" ] || { echo "venv 없음 ($PY) — pod 재시작 후라면 정상:  bash gpu_server/setup_server.sh"; exit 1; }
    [ -n "${UNIVUS_API_KEY:-}" ] || echo "⚠️ UNIVUS_API_KEY 가 비어 있어 인증 없이 뜹니다 — config.local.env 에 넣으세요"
    mkdir -p "$PERSIST_DIR/logs"
    nohup "$PY" gpu_server/app.py --port "$PORT" --pidfile "$PID_FILE" >> "$LOG" 2>&1 < /dev/null &
    sleep 2
    if alive; then echo "시작됨 pid $(cat "$PID_FILE") · 로그 $LOG · 상태 curl http://127.0.0.1:$PORT/health"; else echo "시작 실패 — tail $LOG"; tail -20 "$LOG"; exit 1; fi ;;
  stop)
    if alive; then kill "$(cat "$PID_FILE")" && sleep 1 && echo "중지됨"; rm -f "$PID_FILE"; else echo "실행 중 아님"; rm -f "$PID_FILE"; fi ;;
  restart) "$0" stop; "$0" start ;;
  status)
    if alive; then echo "실행 중 pid $(cat "$PID_FILE") · 포트 $PORT"; curl -s -H "Authorization: Bearer ${UNIVUS_API_KEY:-}" "http://127.0.0.1:$PORT/health" | head -c 600; echo; else echo "실행 중 아님"; fi ;;
  log) tail -n "${2:-50}" -f "$LOG" ;;
  *) echo "사용법: $0 start|stop|status|restart|log"; exit 1 ;;
esac
