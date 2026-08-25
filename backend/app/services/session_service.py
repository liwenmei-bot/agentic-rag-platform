"""
会话（session）与消息（message）持久化服务。

用 SQLite 是因为：单文件、零配置、Python 内置 sqlite3 模块直接能用，
适合个人项目起步阶段。数据量大了、需要多进程并发写入时，再迁移到 PostgreSQL
（迁移只需要改这一个文件里的连接方式和 SQL 语法，上层 api 代码不用大改）。
"""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.core.config import settings

DB_PATH = Path(settings.upload_dir).parent / "app.db"


@contextmanager
def get_db():
    """每次请求开一个连接，用完自动关闭，避免连接泄漏。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 让查询结果能用列名访问，比 tuple 索引直观
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """启动时建表，如果表已存在则跳过（IF NOT EXISTS）。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)


def create_session(title: str = "新对话") -> dict:
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
            (session_id, title, created_at),
        )
    return {"id": session_id, "title": title, "created_at": created_at}


def list_sessions() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_session_messages(session_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_message(session_id: str, role: str, content: str, sources: str = "") -> None:
    """role 是 'user' 或 'assistant'。"""
    message_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, session_id, role, content, sources, created_at),
        )


def update_session_title(session_id: str, title: str) -> None:
    """首次提问后，自动把会话标题改成问题的前几个字，方便在侧边栏识别。"""
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, session_id),
        )


def delete_session(session_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
