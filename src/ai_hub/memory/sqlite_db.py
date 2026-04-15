import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ai_hub.config import DB_PATH
from ai_hub.state import ApprovalStatus, ThreadStatus


def _ensure_parent_dir() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    _ensure_parent_dir()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db() -> None:
    with get_connection() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL UNIQUE,
                subscription_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                agent TEXT,
                content TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS approval_requests (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                command_json TEXT NOT NULL,
                rationale TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                decision_note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                decided_at TEXT,
                executed_at TEXT,
                FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
            )
            """
        )
        con.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _thread_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "archived_at": row["archived_at"],
    }


def _message_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "thread_id": row["thread_id"],
        "role": row["role"],
        "agent": row["agent"],
        "content": row["content"],
        "meta": json.loads(row["meta_json"] or "{}"),
        "created_at": row["created_at"],
    }


def _approval_to_dict(row: sqlite3.Row) -> dict:
    result = json.loads(row["result_json"]) if row["result_json"] else None
    return {
        "id": row["id"],
        "thread_id": row["thread_id"],
        "agent_role": row["agent_role"],
        "tool_name": row["tool_name"],
        "command": json.loads(row["command_json"]),
        "rationale": row["rationale"],
        "status": row["status"],
        "result": result,
        "decision_note": row["decision_note"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "decided_at": row["decided_at"],
        "executed_at": row["executed_at"],
    }


def create_thread(title: str, thread_id: str | None = None) -> dict:
    now = _now()
    thread_id = thread_id or uuid4().hex

    with get_connection() as con:
        con.execute(
            """
            INSERT INTO threads (id, title, status, created_at, updated_at, archived_at)
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (thread_id, title.strip() or "Neuer Chat", ThreadStatus.ACTIVE.value, now, now),
        )
        con.commit()

    return get_thread(thread_id)


def get_thread(thread_id: str) -> dict | None:
    with get_connection() as con:
        row = con.execute(
            "SELECT * FROM threads WHERE id = ?",
            (thread_id,),
        ).fetchone()

    if row is None:
        return None

    return _thread_to_dict(row)


def list_threads(include_archived: bool = True) -> list[dict]:
    query = """
        SELECT t.*
        FROM threads AS t
    """
    params: tuple[object, ...] = ()
    if not include_archived:
        query += " WHERE t.status = ?"
        params = (ThreadStatus.ACTIVE.value,)
    query += " ORDER BY t.updated_at DESC, t.created_at DESC"

    with get_connection() as con:
        rows = con.execute(query, params).fetchall()

    return [_thread_to_dict(row) for row in rows]


def update_thread_title(thread_id: str, title: str) -> dict | None:
    now = _now()
    with get_connection() as con:
        con.execute(
            "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
            (title.strip() or "Neuer Chat", now, thread_id),
        )
        con.commit()
    return get_thread(thread_id)


def archive_thread(thread_id: str) -> dict | None:
    now = _now()
    with get_connection() as con:
        con.execute(
            """
            UPDATE threads
            SET status = ?, archived_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (ThreadStatus.ARCHIVED.value, now, now, thread_id),
        )
        con.commit()
    return get_thread(thread_id)


def delete_thread(thread_id: str) -> bool:
    with get_connection() as con:
        cur = con.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        con.commit()
    return cur.rowcount > 0


def reset_thread(thread_id: str) -> dict | None:
    now = _now()
    with get_connection() as con:
        con.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        con.execute(
            """
            UPDATE approval_requests
            SET status = ?, updated_at = ?, decided_at = ?, decision_note = ?
            WHERE thread_id = ? AND status = ?
            """,
            (
                ApprovalStatus.CANCELLED.value,
                now,
                now,
                "Thread wurde zurückgesetzt.",
                thread_id,
                ApprovalStatus.PENDING.value,
            ),
        )
        con.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?",
            (now, thread_id),
        )
        con.commit()
    return get_thread(thread_id)


def add_message(
    thread_id: str,
    role: str,
    content: str,
    agent: str | None = None,
    meta: dict | None = None,
) -> dict:
    now = _now()
    with get_connection() as con:
        cur = con.execute(
            """
            INSERT INTO messages (thread_id, role, agent, content, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                role,
                agent,
                content,
                json.dumps(meta or {}, separators=(",", ":")),
                now,
            ),
        )
        con.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?",
            (now, thread_id),
        )
        con.commit()
        message_id = cur.lastrowid

    with get_connection() as con:
        row = con.execute(
            "SELECT * FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
    return _message_to_dict(row)


def list_messages(thread_id: str) -> list[dict]:
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT *
            FROM messages
            WHERE thread_id = ?
            ORDER BY id ASC
            """,
            (thread_id,),
        ).fetchall()
    return [_message_to_dict(row) for row in rows]


def create_approval_request(
    thread_id: str,
    agent_role: str,
    tool_name: str,
    command: dict,
    rationale: str,
) -> dict:
    approval_id = uuid4().hex
    now = _now()

    with get_connection() as con:
        con.execute(
            """
            INSERT INTO approval_requests (
                id, thread_id, agent_role, tool_name, command_json, rationale, status,
                result_json, decision_note, created_at, updated_at, decided_at, executed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, NULL, NULL)
            """,
            (
                approval_id,
                thread_id,
                agent_role,
                tool_name,
                json.dumps(command, separators=(",", ":")),
                rationale,
                ApprovalStatus.PENDING.value,
                now,
                now,
            ),
        )
        con.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?",
            (now, thread_id),
        )
        con.commit()

    return get_approval_request(approval_id)


def get_approval_request(approval_id: str) -> dict | None:
    with get_connection() as con:
        row = con.execute(
            "SELECT * FROM approval_requests WHERE id = ?",
            (approval_id,),
        ).fetchone()

    if row is None:
        return None

    return _approval_to_dict(row)


def list_approval_requests(
    status: str | None = None,
    thread_id: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM approval_requests"
    clauses: list[str] = []
    params: list[object] = []

    if status:
        clauses.append("status = ?")
        params.append(status)
    if thread_id:
        clauses.append("thread_id = ?")
        params.append(thread_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY updated_at DESC, created_at DESC"

    with get_connection() as con:
        rows = con.execute(query, tuple(params)).fetchall()

    return [_approval_to_dict(row) for row in rows]


def update_approval_request(
    approval_id: str,
    status: ApprovalStatus,
    decision_note: str | None = None,
    result: dict | None = None,
) -> dict | None:
    now = _now()
    decided_at = now if status in {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.CANCELLED} else None
    executed_at = now if status in {ApprovalStatus.EXECUTED, ApprovalStatus.FAILED} else None

    with get_connection() as con:
        con.execute(
            """
            UPDATE approval_requests
            SET status = ?, result_json = ?, decision_note = ?, updated_at = ?,
                decided_at = COALESCE(?, decided_at),
                executed_at = COALESCE(?, executed_at)
            WHERE id = ?
            """,
            (
                status.value,
                json.dumps(result, separators=(",", ":")) if result is not None else None,
                decision_note,
                now,
                decided_at,
                executed_at,
                approval_id,
            ),
        )
        con.commit()
    return get_approval_request(approval_id)


def count_threads() -> int:
    with get_connection() as con:
        row = con.execute("SELECT COUNT(*) AS count FROM threads").fetchone()
    return int(row["count"])


def count_pending_approvals() -> int:
    with get_connection() as con:
        row = con.execute(
            "SELECT COUNT(*) AS count FROM approval_requests WHERE status = ?",
            (ApprovalStatus.PENDING.value,),
        ).fetchone()
    return int(row["count"])


def upsert_push_subscription(subscription: dict) -> None:
    endpoint = subscription["endpoint"]
    subscription_json = json.dumps(subscription, separators=(",", ":"))
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as con:
        con.execute(
            """
            INSERT INTO push_subscriptions (endpoint, subscription_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(endpoint) DO UPDATE SET
                subscription_json = excluded.subscription_json,
                updated_at = excluded.updated_at
            """,
            (endpoint, subscription_json, now, now),
        )
        con.commit()


def delete_push_subscription(endpoint: str) -> int:
    with get_connection() as con:
        cur = con.execute(
            "DELETE FROM push_subscriptions WHERE endpoint = ?",
            (endpoint,),
        )
        con.commit()
        return cur.rowcount


def list_push_subscriptions() -> list[dict]:
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT endpoint, subscription_json, created_at, updated_at
            FROM push_subscriptions
            ORDER BY updated_at DESC
            """
        ).fetchall()

    result = []
    for row in rows:
        result.append(
            {
                "endpoint": row["endpoint"],
                "subscription": json.loads(row["subscription_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return result


def count_push_subscriptions() -> int:
    with get_connection() as con:
        row = con.execute(
            "SELECT COUNT(*) AS count FROM push_subscriptions"
        ).fetchone()
    return int(row["count"])
