"""F9 대화 저장소 — SQLite data/chat.db (표준 라이브러리만).

    messages(id, at, role, body, seq)   body = 화면에 내려주는 ChatMsg 그대로(JSON). seq 순서가 대화 순서다.
    actions(id, tool, args, target_id, state, expires_at, created_at, message_id)
                                        대기 중인 쓰기(PendingAction). 미리보기 메시지 id == action id.

메시지 본문에는 memo·학번·이름 같은 개인 필드를 넣지 않는다 (service 가 만드는 모양만).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from . import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id    TEXT PRIMARY KEY,
    at    TEXT NOT NULL,
    role  TEXT NOT NULL,
    body  TEXT NOT NULL,
    seq   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_seq ON messages(seq);
CREATE TABLE IF NOT EXISTS actions (
    id          TEXT PRIMARY KEY,
    tool        TEXT NOT NULL,
    args        TEXT NOT NULL,
    target_id   TEXT,
    state       TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    message_id  TEXT NOT NULL
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    C.ensure_dirs()
    con = sqlite3.connect(C.DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------- messages

def has_message(con: sqlite3.Connection, mid: str) -> bool:
    return con.execute("SELECT 1 FROM messages WHERE id = ?", (mid,)).fetchone() is not None


def add_message(con: sqlite3.Connection, msg: dict) -> dict:
    seq = int(con.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM messages").fetchone()[0])
    con.execute("INSERT INTO messages(id, at, role, body, seq) VALUES (?, ?, ?, ?, ?)",
                (msg["id"], msg["at"], msg["role"], json.dumps(msg, ensure_ascii=False), seq))
    return msg


def get_message(con: sqlite3.Connection, mid: str) -> Optional[dict]:
    r = con.execute("SELECT body FROM messages WHERE id = ?", (mid,)).fetchone()
    return json.loads(r["body"]) if r else None


def put_message(con: sqlite3.Connection, msg: dict) -> dict:
    """같은 id 의 본문을 바꾼다 (미리보기의 state·값). 순서(seq)는 그대로."""
    con.execute("UPDATE messages SET body = ? WHERE id = ?", (json.dumps(msg, ensure_ascii=False), msg["id"]))
    return msg


def list_messages(con: sqlite3.Connection, before: Optional[str] = None, limit: int = C.HISTORY_LIMIT) -> list[dict]:
    """오래된 → 최신. before 가 있으면 그 메시지보다 앞의 것 중 최근 limit 개."""
    if before:
        r = con.execute("SELECT seq FROM messages WHERE id = ?", (before,)).fetchone()
        if not r:
            return []
        rows = con.execute("SELECT body FROM messages WHERE seq < ? ORDER BY seq DESC LIMIT ?", (r["seq"], limit)).fetchall()
    else:
        rows = con.execute("SELECT body FROM messages ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
    return [json.loads(x["body"]) for x in reversed(rows)]


def clear(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM messages")
    con.execute("DELETE FROM actions")


# ---------------------------------------------------------------- actions

def add_action(con: sqlite3.Connection, a: dict) -> dict:
    con.execute("INSERT INTO actions(id, tool, args, target_id, state, expires_at, created_at, message_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (a["id"], a["tool"], json.dumps(a["args"], ensure_ascii=False), a.get("targetId"), a["state"],
                 a["expiresAt"], a["createdAt"], a["messageId"]))
    return a


def get_action(con: sqlite3.Connection, aid: str) -> Optional[dict]:
    r = con.execute("SELECT * FROM actions WHERE id = ?", (aid,)).fetchone()
    if not r:
        return None
    return {"id": r["id"], "tool": r["tool"], "args": json.loads(r["args"]), "targetId": r["target_id"],
            "state": r["state"], "expiresAt": r["expires_at"], "createdAt": r["created_at"], "messageId": r["message_id"]}


def claim_action(con: sqlite3.Connection, aid: str, now_iso: str) -> bool:
    """pending → running 을 한 문장으로 (만료 전인 것만). 두 번 누른 확인 중 하나만 True 다."""
    cur = con.execute("UPDATE actions SET state = 'running' WHERE id = ? AND state = 'pending' AND expires_at > ?",
                      (aid, now_iso))
    return cur.rowcount == 1


def cancel_action(con: sqlite3.Connection, aid: str) -> bool:
    cur = con.execute("UPDATE actions SET state = 'canceled' WHERE id = ? AND state = 'pending'", (aid,))
    return cur.rowcount == 1


def set_action(con: sqlite3.Connection, aid: str, state: str, args: Optional[dict] = None) -> None:
    if args is None:
        con.execute("UPDATE actions SET state = ? WHERE id = ?", (state, aid))
    else:
        con.execute("UPDATE actions SET state = ?, args = ? WHERE id = ?", (state, json.dumps(args, ensure_ascii=False), aid))
